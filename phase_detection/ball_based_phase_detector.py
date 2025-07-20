"""
Ball-based Phase Detector

Phase detection strategy based on ball size and distance.
This is the current implementation from basketball_shooting_analyzer.py
"""

from typing import Dict, List, Optional, Tuple
from .base_phase_detector import BasePhaseDetector


class BallBasedPhaseDetector(BasePhaseDetector):
    """
    Ball-based phase detection strategy.
    
    Uses ball size and distance to determine phase transitions.
    This is the current implementation from the analyzer.
    """
    
    def __init__(self, min_phase_duration: int = 3, noise_threshold: int = 4):
        super().__init__(min_phase_duration, noise_threshold)
    
    def check_phase_transition(self, 
                             current_phase: str, 
                             frame_idx: int,
                             pose_data: List[Dict],
                             ball_data: List[Dict],
                             **kwargs) -> str:
        """
        Check phase transition using ball-based logic.
        This is an exact copy of the original _check_phase_transition_original method.
        """
        """
        Check phase transition using ball-based logic.
        
        Args:
            current_phase: Current phase name
            frame_idx: Current frame index
            pose_data: List of pose data for all frames
            ball_data: List of ball data for all frames
            **kwargs: Additional parameters
            
        Returns:
            Next phase name
        """
        # Get current frame data
        pose = self.get_pose_info(frame_idx, pose_data)
        ball_info = self.get_ball_info(frame_idx, ball_data)
        
        # Extract ball information
        ball_x = ball_info.get('center_x', 0) if ball_info else 0
        ball_y = ball_info.get('center_y', 0) if ball_info else 0
        ball_detected = ball_info is not None
        
        # Get previous frame ball data
        prev_ball_info = None
        if frame_idx > 0:
            prev_ball_info = self.get_ball_info(frame_idx - 1, ball_data)
        
        # Calculate ball change amount
        d_ball_y = 0
        if prev_ball_info:
            prev_ball_y = prev_ball_info.get('center_y', 0)
            d_ball_y = ball_y - prev_ball_y
        
        # Extract keypoints
        left_shoulder = pose.get('left_shoulder', {'x': 0, 'y': 0})
        right_shoulder = pose.get('right_shoulder', {'x': 0, 'y': 0})
        left_elbow = pose.get('left_elbow', {'x': 0, 'y': 0})
        right_elbow = pose.get('right_elbow', {'x': 0, 'y': 0})
        left_wrist = pose.get('left_wrist', {'x': 0, 'y': 0})
        right_wrist = pose.get('right_wrist', {'x': 0, 'y': 0})
        
        # Calculate shoulder position
        left_shoulder_y = left_shoulder.get('y', 0)
        right_shoulder_y = right_shoulder.get('y', 0)
        shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        
        # Calculate elbow angles
        left_angle = self.calculate_angle(
            left_shoulder.get('x', 0), left_shoulder.get('y', 0),
            left_elbow.get('x', 0), left_elbow.get('y', 0),
            left_wrist.get('x', 0), left_wrist.get('y', 0)
        )
        right_angle = self.calculate_angle(
            right_shoulder.get('x', 0), right_shoulder.get('y', 0),
            right_elbow.get('x', 0), right_elbow.get('y', 0),
            right_wrist.get('x', 0), right_wrist.get('y', 0)
        )
        
        # Select closest wrist to ball
        wrist_x, wrist_y, selected_wrist = self.select_closest_wrist_to_ball(pose, ball_info)
        
        # Calculate ball-wrist distance
        ball_wrist_distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5 if ball_detected else float('inf')
        
        # Calculate movement changes
        d_wrist_y = 0
        d_hip_y = 0
        if frame_idx > 0:
            prev_pose = self.get_pose_info(frame_idx - 1, pose_data)
            prev_wrist_x, prev_wrist_y, _ = self.select_closest_wrist_to_ball(prev_pose, prev_ball_info)
            prev_hip_data = self.calculate_keypoint_averages(prev_pose, ['hip'])
            current_hip_data = self.calculate_keypoint_averages(pose, ['hip'])
            
            d_wrist_y = wrist_y - prev_wrist_y
            # Use the lower hip for current and previous frame
            current_left_hip = pose.get('left_hip', {'y': None})
            current_right_hip = pose.get('right_hip', {'y': None})
            current_left_hip_y = current_left_hip.get('y', None)
            current_right_hip_y = current_right_hip.get('y', None)
            if current_left_hip_y is not None and current_right_hip_y is not None:
                current_hip_y = max(current_left_hip_y, current_right_hip_y)
            elif current_left_hip_y is not None:
                current_hip_y = current_left_hip_y
            elif current_right_hip_y is not None:
                current_hip_y = current_right_hip_y
            else:
                current_hip_y = 0
                
            prev_left_hip = prev_pose.get('left_hip', {'y': None})
            prev_right_hip = prev_pose.get('right_hip', {'y': None})
            prev_left_hip_y = prev_left_hip.get('y', None)
            prev_right_hip_y = prev_right_hip.get('y', None)
            if prev_left_hip_y is not None and prev_right_hip_y is not None:
                prev_hip_y = max(prev_left_hip_y, prev_right_hip_y)
            elif prev_left_hip_y is not None:
                prev_hip_y = prev_left_hip_y
            elif prev_right_hip_y is not None:
                prev_hip_y = prev_right_hip_y
            else:
                prev_hip_y = 0
                
            d_hip_y = current_hip_y - prev_hip_y
        
        # 1. General → Set-up: Ball is held in hand
        if current_phase == "General":
            if ball_detected:
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                close_threshold = ball_radius * 1.3
                
                distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5
                
                if distance < close_threshold:
                    return "Set-up"
        
        # 2. Set-up → Loading: Hip AND shoulder moving downward
        if current_phase == "Set-up":
            conditions = []
            
            # Calculate hip and shoulder positions
            left_hip = pose.get('left_hip', {'y': None})
            right_hip = pose.get('right_hip', {'y': None})
            left_hip_y = left_hip.get('y', None)
            right_hip_y = right_hip.get('y', None)
            if left_hip_y is not None and right_hip_y is not None:
                hip_y = max(left_hip_y, right_hip_y)
            elif left_hip_y is not None:
                hip_y = left_hip_y
            elif right_hip_y is not None:
                hip_y = right_hip_y
            else:
                hip_y = 0
            
            # Calculate changes from previous frame
            if frame_idx > 0:
                prev_pose = self.get_pose_info(frame_idx - 1, pose_data)
                prev_left_hip = prev_pose.get('left_hip', {'y': None})
                prev_right_hip = prev_pose.get('right_hip', {'y': None})
                prev_left_hip_y = prev_left_hip.get('y', None)
                prev_right_hip_y = prev_right_hip.get('y', None)
                if prev_left_hip_y is not None and prev_right_hip_y is not None:
                    prev_hip_y = max(prev_left_hip_y, prev_right_hip_y)
                elif prev_left_hip_y is not None:
                    prev_hip_y = prev_left_hip_y
                elif prev_right_hip_y is not None:
                    prev_hip_y = prev_right_hip_y
                else:
                    prev_hip_y = 0
                
                prev_hip_data = self.calculate_keypoint_averages(prev_pose, ['hip'])
                prev_shoulder_data = self.calculate_keypoint_averages(prev_pose, ['shoulder'])
                
                # Use the lower hip for previous frame
                prev_left_hip = prev_pose.get('left_hip', {'y': None})
                prev_right_hip = prev_pose.get('right_hip', {'y': None})
                prev_left_hip_y = prev_left_hip.get('y', None)
                prev_right_hip_y = prev_right_hip.get('y', None)
                if prev_left_hip_y is not None and prev_right_hip_y is not None:
                    prev_hip_y = max(prev_left_hip_y, prev_right_hip_y)
                elif prev_left_hip_y is not None:
                    prev_hip_y = prev_left_hip_y
                elif prev_right_hip_y is not None:
                    prev_hip_y = prev_right_hip_y
                else:
                    prev_hip_y = 0
                prev_shoulder_y = prev_shoulder_data['shoulder']['y']
                
                d_hip_y = hip_y - prev_hip_y
                d_shoulder_y = shoulder_y - prev_shoulder_y
                
                # Hip moving downward (y-coordinate increasing) - normalized coordinates
                if d_hip_y > 0.003:  # Adjusted for normalized coordinates (much smaller threshold)
                    conditions.append("hip_down")
                
                # Shoulder moving downward - normalized coordinates
                if d_shoulder_y > 0.003:  # Adjusted for normalized coordinates (much smaller threshold)
                    conditions.append("shoulder_down")
                
                # BOTH hip AND shoulder must be moving down
                if len(conditions) == 2:
                    return "Loading"
        
        # 3. Loading → Rising: Wrist and elbow moving upward
        if current_phase == "Loading":
            conditions = []
            
            # Calculate hip position for relative movement
            left_hip = pose.get('left_hip', {'y': None})
            right_hip = pose.get('right_hip', {'y': None})
            left_hip_y = left_hip.get('y', None)
            right_hip_y = right_hip.get('y', None)
            if left_hip_y is not None and right_hip_y is not None:
                hip_y = max(left_hip_y, right_hip_y)
            elif left_hip_y is not None:
                hip_y = left_hip_y
            elif right_hip_y is not None:
                hip_y = right_hip_y
            else:
                hip_y = 0
            
            if frame_idx > 0:
                prev_pose = self.get_pose_info(frame_idx - 1, pose_data)
                prev_left_hip = prev_pose.get('left_hip', {'y': None})
                prev_right_hip = prev_pose.get('right_hip', {'y': None})
                prev_left_hip_y = prev_left_hip.get('y', None)
                prev_right_hip_y = prev_right_hip.get('y', None)
                if prev_left_hip_y is not None and prev_right_hip_y is not None:
                    prev_hip_y = max(prev_left_hip_y, prev_right_hip_y)
                elif prev_left_hip_y is not None:
                    prev_hip_y = prev_left_hip_y
                elif prev_right_hip_y is not None:
                    prev_hip_y = prev_right_hip_y
                else:
                    prev_hip_y = 0
                
                # Calculate elbow position
                elbow_data = self.calculate_keypoint_averages(pose, ['elbow'])
                prev_elbow_data = self.calculate_keypoint_averages(prev_pose, ['elbow'])
                
                elbow_y = elbow_data['elbow']['y']
                prev_elbow_y = prev_elbow_data['elbow']['y']
                
                # Calculate relative movement
                d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                d_wrist_relative = d_wrist_y - (hip_y - prev_hip_y)
                
                # Wrist moving upward relative to hip - normalized coordinates
                if d_wrist_relative < -0.003:  # Adjusted for normalized coordinates (much smaller threshold)
                    conditions.append("wrist_up_relative")
                
                # Elbow moving upward relative to hip - normalized coordinates
                if d_elbow_relative < -0.003:  # Adjusted for normalized coordinates (much smaller threshold)
                    conditions.append("elbow_up_relative")
                
                # Both conditions must be met
                if len(conditions) == 2:
                    return "Rising"
        
        # 3.5. Set-up → Rising: Skip Loading if Rising conditions met directly
        if current_phase == "Set-up":
            conditions = []
            
            left_hip = pose.get('left_hip', {'y': None})
            right_hip = pose.get('right_hip', {'y': None})
            left_hip_y = left_hip.get('y', None)
            right_hip_y = right_hip.get('y', None)
            if left_hip_y is not None and right_hip_y is not None:
                hip_y = max(left_hip_y, right_hip_y)
            elif left_hip_y is not None:
                hip_y = left_hip_y
            elif right_hip_y is not None:
                hip_y = right_hip_y
            else:
                hip_y = 0
            
            if frame_idx > 0:
                prev_pose = self.get_pose_info(frame_idx - 1, pose_data)
                prev_left_hip = prev_pose.get('left_hip', {'y': None})
                prev_right_hip = prev_pose.get('right_hip', {'y': None})
                prev_left_hip_y = prev_left_hip.get('y', None)
                prev_right_hip_y = prev_right_hip.get('y', None)
                if prev_left_hip_y is not None and prev_right_hip_y is not None:
                    prev_hip_y = max(prev_left_hip_y, prev_right_hip_y)
                elif prev_left_hip_y is not None:
                    prev_hip_y = prev_left_hip_y
                elif prev_right_hip_y is not None:
                    prev_hip_y = prev_right_hip_y
                else:
                    prev_hip_y = 0
                
                elbow_data = self.calculate_keypoint_averages(pose, ['elbow'])
                prev_elbow_data = self.calculate_keypoint_averages(prev_pose, ['elbow'])
                
                elbow_y = elbow_data['elbow']['y']
                prev_elbow_y = prev_elbow_data['elbow']['y']
                
                # Calculate relative movement
                d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                d_wrist_relative = d_wrist_y - (hip_y - prev_hip_y)
                d_ball_relative = d_ball_y - (hip_y - prev_hip_y) if ball_detected else 0
                
                # All three conditions must be met - normalized coordinates
                if d_wrist_relative < -0.003:  # Adjusted for normalized coordinates (much smaller threshold)
                    conditions.append("wrist_up_relative")
                if d_elbow_relative < -0.003:  # Adjusted for normalized coordinates (much smaller threshold)
                    conditions.append("elbow_up_relative")
                if ball_detected and d_ball_relative < -0.003:  # Adjusted for normalized coordinates (much smaller threshold)
                    conditions.append("ball_up_relative")
                
                if len(conditions) == 3:
                    return "Rising"
        
        # Check for ball missed conditions
        if current_phase in ["Set-up", "Loading"]:
            if ball_detected:
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                close_threshold = ball_radius * 1.3
                
                distance = abs(ball_y - wrist_y)
                
                if distance > close_threshold:
                    if frame_idx >= self.min_phase_duration:
                        return "General"
                    else:
                        return current_phase
        
        # 4. Rising → Release: Ball is released with proper form
        if current_phase == "Rising":
            # Check for cancellation first
            if frame_idx > 0:
                prev_pose = self.get_pose_info(frame_idx - 1, pose_data)
                prev_left_hip = prev_pose.get('left_hip', {'y': None})
                prev_right_hip = prev_pose.get('right_hip', {'y': None})
                prev_left_hip_y = prev_left_hip.get('y', None)
                prev_right_hip_y = prev_right_hip.get('y', None)
                if prev_left_hip_y is not None and prev_right_hip_y is not None:
                    prev_hip_y = max(prev_left_hip_y, prev_right_hip_y)
                elif prev_left_hip_y is not None:
                    prev_hip_y = prev_left_hip_y
                elif prev_right_hip_y is not None:
                    prev_hip_y = prev_right_hip_y
                else:
                    prev_hip_y = 0
                
                left_hip = pose.get('left_hip', {'y': None})
                right_hip = pose.get('right_hip', {'y': None})
                left_hip_y = left_hip.get('y', None)
                right_hip_y = right_hip.get('y', None)
                if left_hip_y is not None and right_hip_y is not None:
                    hip_y = max(left_hip_y, right_hip_y)
                elif left_hip_y is not None:
                    hip_y = left_hip_y
                elif right_hip_y is not None:
                    hip_y = right_hip_y
                else:
                    hip_y = 0
                
                elbow_data = self.calculate_keypoint_averages(pose, ['elbow'])
                prev_elbow_data = self.calculate_keypoint_averages(prev_pose, ['elbow'])
                
                elbow_y = elbow_data['elbow']['y']
                prev_elbow_y = prev_elbow_data['elbow']['y']
                
                # Calculate relative movement
                d_wrist_relative = d_wrist_y - (hip_y - prev_hip_y)
                d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                d_ball_relative = d_ball_y - (hip_y - prev_hip_y) if ball_detected else 0
                
                wrist_moving_down = d_wrist_relative > 0.003  # Adjusted for normalized coordinates (much smaller threshold)
                elbow_moving_down = d_elbow_relative > 0.003  # Adjusted for normalized coordinates (much smaller threshold)
                
                # Rising cancellation
                if ball_detected:
                    ball_moving_down = d_ball_relative > 0.003  # Adjusted for normalized coordinates
                    if wrist_moving_down and elbow_moving_down and ball_moving_down:
                        return "Set-up"
                else:
                    if wrist_moving_down and elbow_moving_down:
                        return "Set-up"
            
            # Normal Rising → Release transition
            if ball_detected:
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                close_threshold = ball_radius * 1.3
                
                distance = abs(ball_y - wrist_y)
                wrist_above_shoulder = wrist_y < shoulder_y
                ball_released = distance > close_threshold
                
                if ball_released:
                    ball_above_shoulder = ball_y < shoulder_y
                    
                    if (left_angle >= 110 or right_angle >= 110) and wrist_above_shoulder and ball_above_shoulder:
                        return "Release"
                    else:
                        return "Set-up"
                else:
                    ball_above_shoulder = ball_y < shoulder_y
                    
                    if (left_angle >= 110 or right_angle >= 110) and distance > close_threshold and ball_above_shoulder:
                        return "Release"
        
        # 5. Release → Follow-through: Ball has fully left the hand
        if current_phase == "Release":
            if ball_detected:
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                close_threshold = ball_radius * 1.5
                medium_threshold = ball_radius * 2.5
                far_threshold = ball_radius * 4.0
                
                if ball_wrist_distance > far_threshold:
                    return "Follow-through"
                elif ball_wrist_distance > medium_threshold:
                    return "Follow-through"
                elif ball_wrist_distance > close_threshold:
                    return "Follow-through"
        
        # 6. Follow-through → General: Wrist below eyes relative to hip
        if current_phase == "Follow-through":
            # Check if ball is caught
            if ball_detected:
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                close_threshold = ball_radius * 1.3
                
                if ball_wrist_distance <= close_threshold:
                    return "Set-up"
            
            # Check if wrist is below eyes relative to hip
            if frame_idx > 0:
                left_eye = pose.get('left_eye', {'y': 0})
                right_eye = pose.get('right_eye', {'y': 0})
                eye_y = max(left_eye.get('y', 0), right_eye.get('y', 0))
                
                left_wrist = pose.get('left_wrist', {'y': 0})
                right_wrist = pose.get('right_wrist', {'y': 0})
                wrist_y = min(left_wrist.get('y', 0), right_wrist.get('y', 0))
                
                left_hip = pose.get('left_hip', {'y': None})
                right_hip = pose.get('right_hip', {'y': None})
                left_hip_y = left_hip.get('y', None)
                right_hip_y = right_hip.get('y', None)
                if left_hip_y is not None and right_hip_y is not None:
                    hip_y = max(left_hip_y, right_hip_y)
                elif left_hip_y is not None:
                    hip_y = left_hip_y
                elif right_hip_y is not None:
                    hip_y = right_hip_y
                else:
                    hip_y = 0
                
                eye_relative_to_hip = eye_y - hip_y
                wrist_relative_to_hip = wrist_y - hip_y
                
                if wrist_relative_to_hip > eye_relative_to_hip:
                    if frame_idx >= self.min_phase_duration:
                        return "General"
                    else:
                        return current_phase
        
        # If no conditions are met, keep current phase
        return current_phase 