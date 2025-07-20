"""
Hybrid Phase Detector

Phase detection strategy that combines multiple approaches.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .base_phase_detector import BasePhaseDetector
from .ball_based_phase_detector import BallBasedPhaseDetector
from .torso_based_phase_detector import TorsoBasedPhaseDetector


class HybridPhaseDetector(BasePhaseDetector):
    """
    Hybrid phase detection strategy.
    
    Combines ball-based and torso-based approaches with adaptive weights.
    """
    
    def __init__(self, 
                 ball_weight: float = 0.6,
                 torso_weight: float = 0.4,
                 min_phase_duration: int = 3, 
                 noise_threshold: int = 4):
        super().__init__(min_phase_duration, noise_threshold)
        self.ball_weight = ball_weight
        self.torso_weight = torso_weight
        
        # Initialize sub-detectors
        self.ball_detector = BallBasedPhaseDetector(min_phase_duration, noise_threshold)
        self.torso_detector = TorsoBasedPhaseDetector(min_phase_duration, noise_threshold)
    
    def get_adaptive_weights(self, 
                           ball_confidence: float,
                           torso_confidence: float) -> Tuple[float, float]:
        """
        Calculate adaptive weights based on confidence.
        
        Args:
            ball_confidence: Ball detection confidence (0~1)
            torso_confidence: Torso detection confidence (0~1)
            
        Returns:
            Tuple of (ball_weight, torso_weight)
        """
        total_confidence = ball_confidence + torso_confidence
        
        if total_confidence > 0:
            ball_weight = ball_confidence / total_confidence
            torso_weight = torso_confidence / total_confidence
        else:
            # Use default weights if both confidences are low
            ball_weight = self.ball_weight
            torso_weight = self.torso_weight
        
        return ball_weight, torso_weight
    
    def calculate_hybrid_threshold(self, 
                                 pose: Dict, 
                                 ball_info: Optional[Dict],
                                 ball_confidence: float = 1.0,
                                 torso_confidence: float = 1.0) -> float:
        """
        Calculate hybrid threshold combining ball and torso approaches.
        
        Args:
            pose: Pose data
            ball_info: Ball information
            ball_confidence: Ball detection confidence
            torso_confidence: Torso detection confidence
            
        Returns:
            Hybrid threshold value
        """
        # Get adaptive weights
        ball_weight, torso_weight = self.get_adaptive_weights(ball_confidence, torso_confidence)
        
        # Calculate ball-based threshold
        ball_threshold = 0
        if ball_info is not None:
            ball_width = ball_info.get('width', 0)
            ball_height = ball_info.get('height', 0)
            ball_radius = (ball_width + ball_height) / 4
            ball_threshold = ball_radius * 1.3
        
        # Calculate torso-based threshold
        torso_threshold = self.torso_detector.calculate_torso_based_threshold(pose)
        
        # Combine thresholds with weights
        hybrid_threshold = (ball_weight * ball_threshold + 
                          torso_weight * torso_threshold)
        
        return hybrid_threshold
    
    def check_phase_transition(self, 
                             current_phase: str, 
                             frame_idx: int,
                             pose_data: List[Dict],
                             ball_data: List[Dict],
                             **kwargs) -> str:
        """
        Check phase transition using hybrid logic.
        
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
        
        # Calculate confidences
        ball_confidence = ball_info.get('confidence', 0.5) if ball_info else 0.0
        torso_confidence = 0.8  # Assume torso detection is generally stable
        
        # Calculate hybrid threshold
        hybrid_threshold = self.calculate_hybrid_threshold(
            pose, ball_info, ball_confidence, torso_confidence
        )
        
        # Use ball-based detector for detailed logic
        ball_result = self.ball_detector.check_phase_transition(
            current_phase, frame_idx, pose_data, ball_data, **kwargs
        )
        
        # Use torso-based detector as backup
        torso_result = self.torso_detector.check_phase_transition(
            current_phase, frame_idx, pose_data, ball_data, **kwargs
        )
        
        # Combine results based on confidence
        if ball_confidence > 0.7:
            # High ball confidence - use ball-based result
            return ball_result
        elif torso_confidence > 0.7:
            # High torso confidence - use torso-based result
            return torso_result
        else:
            # Low confidence - use hybrid approach
            # For now, prefer ball-based result but with hybrid threshold
            return ball_result
    
    def get_phase_specific_weights(self, current_phase: str) -> Tuple[float, float]:
        """
        Get phase-specific weights for ball and torso approaches.
        
        Args:
            current_phase: Current phase name
            
        Returns:
            Tuple of (ball_weight, torso_weight)
        """
        phase_weights = {
            "General": (0.6, 0.4),      # Ball detection important for ball holding
            "Set-up": (0.5, 0.5),       # Balanced approach
            "Loading": (0.3, 0.7),      # Body movement more important
            "Rising": (0.4, 0.6),       # Body movement important
            "Release": (0.7, 0.3),      # Ball release critical
            "Follow-through": (0.6, 0.4) # Ball trajectory important
        }
        
        return phase_weights.get(current_phase, (self.ball_weight, self.torso_weight)) 