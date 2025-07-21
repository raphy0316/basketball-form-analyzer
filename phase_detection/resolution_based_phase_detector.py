"""
Resolution-based Phase Detector

Phase detection strategy based on video resolution.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .base_phase_detector import BasePhaseDetector


class ResolutionBasedPhaseDetector(BasePhaseDetector):
    """
    Resolution-based phase detection strategy.
    
    Uses video resolution to adjust thresholds dynamically.
    """
    
    def __init__(self, 
                 base_resolution: Tuple[int, int] = (1920, 1080),
                 min_phase_duration: int = 3, 
                 noise_threshold: int = 4):
        super().__init__(min_phase_duration, noise_threshold)
        self.base_resolution = base_resolution
        self.current_resolution = base_resolution
    
    def set_resolution(self, resolution: Tuple[int, int]):
        """
        Set current video resolution.
        
        Args:
            resolution: Current resolution (width, height)
        """
        self.current_resolution = resolution
    
    def calculate_resolution_factor(self) -> float:
        """
        Calculate resolution scaling factor.
        
        Returns:
            Resolution scaling factor
        """
        base_width, base_height = self.base_resolution
        current_width, current_height = self.current_resolution
        
        # Calculate ratios
        width_ratio = current_width / base_width
        height_ratio = current_height / base_height
        
        # Use average ratio
        resolution_factor = (width_ratio + height_ratio) / 2
        
        return resolution_factor
    
    def calculate_resolution_based_threshold(self, base_threshold: float) -> float:
        """
        Calculate threshold adjusted for current resolution.
        
        Args:
            base_threshold: Base threshold value
            
        Returns:
            Resolution-adjusted threshold
        """
        resolution_factor = self.calculate_resolution_factor()
        adjusted_threshold = base_threshold * resolution_factor
        
        return adjusted_threshold
    
    def check_phase_transition(self, 
                             current_phase: str, 
                             frame_idx: int,
                             pose_data: List[Dict],
                             ball_data: List[Dict],
                             **kwargs) -> str:
        """
        Check phase transition using resolution-based logic.
        
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
        
        # Base thresholds (for 1920x1080 resolution)
        base_thresholds = {
            "close": 2.0,
            "medium": 4.0,
            "far": 8.0,
            "very_far": 15.0
        }
        
        # Calculate resolution-adjusted thresholds
        adjusted_thresholds = {}
        for key, base_threshold in base_thresholds.items():
            adjusted_thresholds[key] = self.calculate_resolution_based_threshold(base_threshold)
        
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
            left_shoulder.get('x', 0) , left_shoulder.get('y', 0),
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
        
        # Calculate movement changes
        d_wrist_y = 0
        d_hip_y = 0
        if frame_idx > 0:
            prev_pose = self.get_pose_info(frame_idx - 1, pose_data)
            prev_wrist_x, prev_wrist_y, _ = self.select_closest_wrist_to_ball(prev_pose, None)
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
            if ball_info is not None:
                ball_x = ball_info.get('center_x', 0)
                ball_y = ball_info.get('center_y', 0)
                
                distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5
                
                if distance < adjusted_thresholds["close"]:
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
                
                # Use resolution-adjusted threshold for movement detection
                movement_threshold = adjusted_thresholds["close"]
                
                # Hip moving downward (y-coordinate increasing)
                if d_hip_y > movement_threshold:
                    conditions.append("hip_down")
                
                # Shoulder moving downward
                if d_shoulder_y > movement_threshold:
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
                
                # Calculate relative movement (compared to hip)
                d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                d_wrist_relative = d_wrist_y - (hip_y - prev_hip_y)
                
                # Use resolution-adjusted threshold for relative movement
                relative_threshold = -adjusted_thresholds["close"]  # Negative for upward movement
                
                # Wrist moving upward relative to hip
                if d_wrist_relative < relative_threshold:
                    conditions.append("wrist_up_relative")
                
                # Elbow moving upward relative to hip
                if d_elbow_relative < relative_threshold:
                    conditions.append("elbow_up_relative")
                
                # Both conditions must be met
                if len(conditions) == 2:
                    return "Rising"
        
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
                
                elbow_data = self.calculate_keypoint_averages(pose, ['elbow'])
                prev_elbow_data = self.calculate_keypoint_averages(prev_pose, ['elbow'])
                
                elbow_y = elbow_data['elbow']['y']
                prev_elbow_y = prev_elbow_data['elbow']['y']
                
                # Calculate relative movement
                d_wrist_relative = d_wrist_y - (hip_y - prev_hip_y)
                d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                
                movement_threshold = adjusted_thresholds["close"]
                wrist_moving_down = d_wrist_relative > movement_threshold
                elbow_moving_down = d_elbow_relative > movement_threshold
                
                # Rising cancellation
                if wrist_moving_down and elbow_moving_down:
                    return "Set-up"
            
            # Normal Rising → Release transition
            if ball_info is not None:
                ball_x = ball_info.get('center_x', 0)
                ball_y = ball_info.get('center_y', 0)
                
                distance = abs(ball_y - wrist_y)
                wrist_above_shoulder = wrist_y < shoulder_y
                ball_released = distance > adjusted_thresholds["close"]
                
                if ball_released:
                    ball_above_shoulder = ball_y < shoulder_y
                    
                    if (left_angle >= 110 or right_angle >= 110) and wrist_above_shoulder and ball_above_shoulder:
                        return "Release"
                    else:
                        return "Set-up"
                else:
                    ball_above_shoulder = ball_y < shoulder_y
                    
                    if (left_angle >= 110 or right_angle >= 110) and distance > adjusted_thresholds["close"] and ball_above_shoulder:
                        return "Release"
        
        # 5. Release → Follow-through: Ball has fully left the hand
        if current_phase == "Release":
            if ball_info is not None:
                ball_x = ball_info.get('center_x', 0)
                ball_y = ball_info.get('center_y', 0)
                
                ball_wrist_distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5
                
                # Use resolution-adjusted thresholds
                close_threshold = adjusted_thresholds["close"] * 1.5
                medium_threshold = adjusted_thresholds["medium"] * 1.5
                far_threshold = adjusted_thresholds["far"] * 1.5
                
                if ball_wrist_distance > far_threshold:
                    return "Follow-through"
                elif ball_wrist_distance > medium_threshold:
                    return "Follow-through"
                elif ball_wrist_distance > close_threshold:
                    return "Follow-through"
        
        # 6. Follow-through → General: Wrist below eyes relative to hip
        if current_phase == "Follow-through":
            # Check if ball is caught
            if ball_info is not None:
                ball_x = ball_info.get('center_x', 0)
                ball_y = ball_info.get('center_y', 0)
                
                ball_wrist_distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5
                
                if ball_wrist_distance <= adjusted_thresholds["close"]:
                    return "Set-up"
            
            # Check if wrist is below eyes relative to hip
            if frame_idx > 0:
                left_eye = pose.get('left_eye', {'y': 0})
                right_eye = pose.get('right_eye', {'y': 0})
                eye_y = max(left_eye.get('y', 0), right_eye.get('y', 0))
                
                left_wrist = pose.get('left_wrist', {'y': 0})
                right_wrist = pose.get('right_wrist', {'y': 0})
                wrist_y = min(left_wrist.get('y', 0), right_wrist.get('y', 0))
                
                hip_data = self.calculate_keypoint_averages(pose, ['hip'])
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