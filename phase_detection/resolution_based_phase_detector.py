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
        Check for phase transition based on resolution-based movement patterns.
        
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
        
        # Calculate resolution-based threshold
        threshold = self.calculate_resolution_based_threshold(pose)
        
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