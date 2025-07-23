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
    
    def _get_previous_valid_frame(self, frame_idx: int, pose_data: List[Dict], ball_data: List[Dict]) -> Optional[int]:
        """
        Get the index of the most recent valid frame (not skipped due to missing data).
        
        Args:
            frame_idx: Current frame index
            pose_data: List of pose data
            ball_data: List of ball data
            
        Returns:
            Index of most recent valid frame, or None if no valid frame found
        """
        for i in range(frame_idx - 1, -1, -1):
            pose = self.get_pose_info(i, pose_data)
            ball_info = self.get_ball_info(i, ball_data)
            
            # Simple check: if pose data exists and has at least one keypoint
            if pose and len(pose) > 0:
                return i
        
        return None



    def check_phase_transition(self, 
                             current_phase: str, 
                             frame_idx: int,
                             pose_data: List[Dict],
                             ball_data: List[Dict],
                             **kwargs) -> str:
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
            # Get the most recent valid previous frame
            prev_frame_idx = self._get_previous_valid_frame(frame_idx, pose_data, ball_data)
            if prev_frame_idx is not None:
                prev_ball_info = self.get_ball_info(prev_frame_idx, ball_data)
        
        # Calculate ball change amount
        d_ball_y = 0
        if prev_ball_info:
            prev_ball_y = prev_ball_info.get('center_y', 0)
            d_ball_y = ball_y - prev_ball_y
        
        # Extract keypoints with safety checks
        left_shoulder = pose.get('left_shoulder')
        right_shoulder = pose.get('right_shoulder')
        left_elbow = pose.get('left_elbow')
        right_elbow = pose.get('right_elbow')
        left_wrist = pose.get('left_wrist')
        right_wrist = pose.get('right_wrist')
        
        # Check if required keypoints exist
        if not all([left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]):
            return current_phase
        
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
            # Get the most recent valid previous frame
            prev_frame_idx = self._get_previous_valid_frame(frame_idx, pose_data, ball_data)
            if prev_frame_idx is not None:
                prev_pose = self.get_pose_info(prev_frame_idx, pose_data)
                prev_wrist_x, prev_wrist_y, _ = self.select_closest_wrist_to_ball(prev_pose, prev_ball_info)
                # Use max hip position instead of average
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
                
                d_wrist_y = wrist_y - prev_wrist_y
                d_hip_y = current_hip_y - prev_hip_y
        
        # 1. General → Set-up: Ball is held in hand
        if current_phase == "General":
            # Check if required data exists for this condition
            if ball_detected and 'left_wrist' in pose and 'right_wrist' in pose:
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                close_threshold = ball_radius * 1.3
                
                if ball_wrist_distance < close_threshold:
                    return "Set-up"
        
        # 2. Set-up → Loading: Ball moving downward
        if current_phase == "Set-up":
            # Check if required data exists for this condition
            if ball_detected and frame_idx > 0:
                if d_ball_y > 0:  # Ball moving downward
                    return "Loading"
        
        # 3. Loading → Rising: Ball moving upward
        if current_phase == "Loading":
            # Check if required data exists for this condition
            if ball_detected and frame_idx > 0:
                if d_ball_y < 0:  # Ball moving upward
                    return "Rising"
        
        # 4. Rising → Release: Ball is released
        if current_phase == "Rising":
            # Check if required data exists for this condition
            if ball_detected and 'left_wrist' in pose and 'right_wrist' in pose:
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                release_threshold = ball_radius * 2.0
                
                if ball_wrist_distance > release_threshold:
                    return "Release"
        
        # 5. Release → Follow-through: Ball is far from wrist
        if current_phase == "Release":
            # Check if required data exists for this condition
            if ball_detected and 'left_wrist' in pose and 'right_wrist' in pose:
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                follow_through_threshold = ball_radius * 3.0
                
                if ball_wrist_distance > follow_through_threshold:
                    return "Follow-through"
        
        # 6. Follow-through → General: Ball is caught or far away
        if current_phase == "Follow-through":
            # Check if required data exists for this condition
            if ball_detected and 'left_wrist' in pose and 'right_wrist' in pose:
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                catch_threshold = ball_radius * 1.5
                
                if ball_wrist_distance < catch_threshold:
                    return "Set-up"
                elif ball_wrist_distance > ball_radius * 5.0:  # Ball is very far
                    return "General"
        
        return current_phase 