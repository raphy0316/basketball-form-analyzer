"""
Torso-based Phase Detector

Phase detection strategy based on body size (torso length).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .base_phase_detector import BasePhaseDetector


class TorsoBasedPhaseDetector(BasePhaseDetector):
    """
    Torso-based phase detection strategy.
    
    Uses body size (torso length) to determine phase transitions.
    More stable than ball-based detection.
    """
    
    def __init__(self, min_phase_duration: int = 3, noise_threshold: int = 4):
        super().__init__(min_phase_duration, noise_threshold)
        self.torso_length_cache = []
        self.cache_size = 30
    
    def calculate_torso_length(self, pose: Dict) -> float:
        """
        Calculate torso length from pose data.
        Uses left shoulder-left hip and right shoulder-right hip distances,
        returns the longer one.
        If keypoints are missing, uses the last valid torso length.
        
        Args:
            pose: Pose data
            
        Returns:
            Torso length in normalized units
        """
        # Check if required keypoints exist
        required_keypoints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        for keypoint in required_keypoints:
            if keypoint not in pose:
                # Use last valid torso length if available
                if hasattr(self, 'torso_length') and self.torso_length > 0:
                    return self.torso_length
                else:
                    return 0.1  # Default fallback value
            kp_data = pose[keypoint]
            if not isinstance(kp_data, dict) or 'x' not in kp_data or 'y' not in kp_data:
                # Use last valid torso length if available
                if hasattr(self, 'torso_length') and self.torso_length > 0:
                    return self.torso_length
                else:
                    return 0.1  # Default fallback value
        
        # Get shoulder and hip positions
        left_shoulder = pose['left_shoulder']
        right_shoulder = pose['right_shoulder']
        left_hip = pose['left_hip']
        right_hip = pose['right_hip']
        
        # Calculate left torso length (left shoulder to left hip)
        left_torso_length = np.sqrt(
            (left_shoulder.get('x', 0) - left_hip.get('x', 0))**2 + 
            (left_shoulder.get('y', 0) - left_hip.get('y', 0))**2
        )
        
        # Calculate right torso length (right shoulder to right hip)
        right_torso_length = np.sqrt(
            (right_shoulder.get('x', 0) - right_hip.get('x', 0))**2 + 
            (right_shoulder.get('y', 0) - right_hip.get('y', 0))**2
        )
        
        # Return the longer torso length
        torso_length = max(left_torso_length, right_torso_length)
        
        # Update stored torso length if valid
        if torso_length > 0:
            self.torso_length = torso_length
        
        return torso_length
    
    def get_stable_torso_length(self, pose: Dict) -> float:
        """
        Get stable torso length.
        Uses current frame's torso length if valid, otherwise uses last valid value.
        
        Args:
            pose: Current pose data
            
        Returns:
            Stable torso length
        """
        # Calculate current torso length (with fallback to last valid value)
        torso_length = self.calculate_torso_length(pose)
        
        # Store the torso length for future use
        if torso_length > 0:
            self.torso_length = torso_length
        
        return torso_length
    
    def calculate_torso_based_threshold(self, pose: Dict) -> float:
        """
        Calculate threshold based on torso length.
        
        Args:
            pose: Pose data
            
        Returns:
            Threshold value
        """
        # For normalized data, use fixed threshold values
        # Since data is already normalized by torso length, use relative thresholds
        return 0.15  # 15% of normalized torso length
    
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
        Check for phase transition based on torso-based movement patterns.
        
        Args:
            current_phase: Current phase
            frame_idx: Current frame index
            pose_data: List of pose data
            ball_data: List of ball data
            
        Returns:
            New phase or current phase if no transition
        """
        # Get current frame data
        pose = self.get_pose_info(frame_idx, pose_data)
        ball_info = self.get_ball_info(frame_idx, ball_data)
        
        # Simple check: if pose data exists and has at least one keypoint
        if not pose or len(pose) == 0:
            return current_phase
        
        # Get previous valid frame for comparison
        prev_frame_idx = self._get_previous_valid_frame(frame_idx, pose_data, ball_data)
        if prev_frame_idx is None:
            return current_phase
        
        prev_pose = self.get_pose_info(prev_frame_idx, pose_data)
        
        # Calculate movement deltas
        movement_deltas = self.calculate_movement_deltas(pose, prev_pose)
        
        # Get stable torso length for threshold calculation
        torso_length = self.get_stable_torso_length(pose)
        
        # Calculate torso-based threshold
        threshold = self.calculate_torso_based_threshold(pose)
        
        # Phase transition logic
        if current_phase == "Set-up":
            # Set-up → Loading: Significant upward movement
            if movement_deltas['wrist_y'] < -threshold:
                return "Loading"
        
        elif current_phase == "Loading":
            # Loading → Rising: Continued upward movement
            if movement_deltas['wrist_y'] < -threshold * 0.5:
                return "Rising"
        
        elif current_phase == "Rising":
            # Rising → Release: Peak reached, slight downward movement
            if movement_deltas['wrist_y'] > threshold * 0.3:
                return "Release"
        
        elif current_phase == "Release":
            # Release → Follow-through: Ball released, wrist moving down
            if movement_deltas['wrist_y'] > threshold * 0.5:
                return "Follow-through"
        
        elif current_phase == "Follow-through":
            # Follow-through → General: Return to general state
            return "General"
        
        return current_phase 