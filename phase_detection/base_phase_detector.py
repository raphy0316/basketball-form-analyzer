"""
Base Phase Detector

Abstract base class for all phase detection strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np


class BasePhaseDetector(ABC):
    """
    Abstract base class for phase detection strategies.
    
    All phase detectors must implement the check_phase_transition method.
    """
    
    def __init__(self, min_phase_duration: int = 3, noise_threshold: int = 4):
        """
        Initialize the phase detector.
        
        Args:
            min_phase_duration: Minimum frames a phase must last
            noise_threshold: Threshold for noise filtering
        """
        self.min_phase_duration = min_phase_duration
        self.noise_threshold = noise_threshold
        self.phase_history = []
    
    @abstractmethod
    def check_phase_transition(self, 
                             current_phase: str, 
                             frame_idx: int,
                             pose_data: List[Dict],
                             ball_data: List[Dict],
                             **kwargs) -> str:
        """
        Check if phase should transition to next phase.
        
        Args:
            current_phase: Current phase name
            frame_idx: Current frame index
            pose_data: List of pose data for all frames
            ball_data: List of ball data for all frames
            **kwargs: Additional parameters
            
        Returns:
            Next phase name
        """
        pass
    
    def calculate_angle(self, ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
        """
        Calculate angle between three points.
        
        Args:
            ax, ay: First point coordinates
            bx, by: Second point coordinates (vertex)
            cx, cy: Third point coordinates
            
        Returns:
            Angle in degrees
        """
        # Vector AB
        ab_x = ax - bx
        ab_y = ay - by
        
        # Vector CB
        cb_x = cx - bx
        cb_y = cy - by
        
        # Dot product
        dot_product = ab_x * cb_x + ab_y * cb_y
        
        # Magnitudes
        ab_magnitude = np.sqrt(ab_x**2 + ab_y**2)
        cb_magnitude = np.sqrt(cb_x**2 + cb_y**2)
        
        # Avoid division by zero
        if ab_magnitude == 0 or cb_magnitude == 0:
            return 0.0
        
        # Cosine of angle
        cos_angle = dot_product / (ab_magnitude * cb_magnitude)
        
        # Clamp to valid range
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def get_ball_info(self, frame_idx: int, ball_data: List[Dict]) -> Optional[Dict]:
        """
        Get ball information for a specific frame.
        
        Args:
            frame_idx: Frame index
            ball_data: List of ball data
            
        Returns:
            Ball information dict or None
        """
        if frame_idx < len(ball_data):
            ball_frame_data = ball_data[frame_idx]
            if isinstance(ball_frame_data, dict) and ball_frame_data.get('ball_detections'):
                ball_detections = ball_frame_data['ball_detections']
                if ball_detections and isinstance(ball_detections[0], dict):
                    return ball_detections[0]
        return None
    
    def get_pose_info(self, frame_idx: int, pose_data: List[Dict]) -> Dict:
        """
        Get pose information for a specific frame.
        
        Args:
            frame_idx: Frame index
            pose_data: List of pose data
            
        Returns:
            Pose information dict
        """
        if frame_idx < len(pose_data):
            return pose_data[frame_idx].get('pose', {})
        return {}
    
    def calculate_keypoint_averages(self, pose: Dict, keypoints: List[str]) -> Dict[str, float]:
        """
        Calculate average positions for keypoints.
        
        Args:
            pose: Pose data
            keypoints: List of keypoint pairs to average
            
        Returns:
            Dict with averaged keypoint positions
        """
        averages = {}
        
        for keypoint_pair in keypoints:
            left_key = f"left_{keypoint_pair}"
            right_key = f"right_{keypoint_pair}"
            
            left_pos = pose.get(left_key, {'x': 0, 'y': 0})
            right_pos = pose.get(right_key, {'x': 0, 'y': 0})
            
            avg_x = (left_pos.get('x', 0) + right_pos.get('x', 0)) / 2
            avg_y = (left_pos.get('y', 0) + right_pos.get('y', 0)) / 2
            
            averages[keypoint_pair] = {'x': avg_x, 'y': avg_y}
        
        return averages
    
    def select_closest_wrist_to_ball(self, pose: Dict, ball_info: Optional[Dict]) -> Tuple[float, float, str]:
        """
        Select the wrist closest to the ball.
        
        Args:
            pose: Pose data
            ball_info: Ball information
            
        Returns:
            Tuple of (wrist_x, wrist_y, selected_side)
        """
        left_wrist = pose.get('left_wrist', {'x': 0, 'y': 0})
        right_wrist = pose.get('right_wrist', {'x': 0, 'y': 0})
        
        if ball_info is None:
            # Default to left wrist if no ball info
            return left_wrist.get('x', 0), left_wrist.get('y', 0), "left"
        
        ball_x = ball_info.get('center_x', 0)
        ball_y = ball_info.get('center_y', 0)
        
        left_distance = ((ball_x - left_wrist.get('x', 0))**2 + 
                       (ball_y - left_wrist.get('y', 0))**2)**0.5
        right_distance = ((ball_x - right_wrist.get('x', 0))**2 + 
                         (ball_y - right_wrist.get('y', 0))**2)**0.5
        
        if left_distance <= right_distance:
            return left_wrist.get('x', 0), left_wrist.get('y', 0), "left"
        else:
            return right_wrist.get('x', 0), right_wrist.get('y', 0), "right"
    
    def is_trend_based_transition(self, frame_idx: int, current_phase: str, next_phase: str) -> bool:
        """
        Check if transition is based on trend (always returns True for now).
        
        Args:
            frame_idx: Frame index
            current_phase: Current phase
            next_phase: Next phase
            
        Returns:
            True if transition should occur
        """
        return True
    
    def update_phase_history(self, phase: str, start_frame: int, end_frame: int):
        """
        Update phase history.
        
        Args:
            phase: Phase name
            start_frame: Start frame index
            end_frame: End frame index
        """
        self.phase_history.append((phase, start_frame, end_frame)) 