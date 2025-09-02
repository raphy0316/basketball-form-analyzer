"""
Release Posture Similarity Scorer

Calculates release phase similarity based on posture features instead of DTW.
Uses existing release_analysis features with weighted scoring system.
Handles multiple release frames by averaging values with type-safe error handling.
"""

import numpy as np
from typing import Dict, Optional, List, Union
import sys
import os

# Add parent directory to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shooting_comparison import config

class ReleasePostureScorer:
    """
    Calculate release phase similarity based on posture features.
    
    Target scores:
    - Same person different shots: ~90 points
    - Different posture but still shooting form: ~50 points
    
    Handles 1-2 frame release phases with averaging and interpolation for missing data.
    """
    
    def __init__(self):
        
        # Feature weights for release posture similarity
        self.feature_weights = {
            'arm_angles': 0.30,        # Most important - shooting arm position
            'ball_position': 0.30,     # Critical for release accuracy
            'body_tilt': 0.25,         # Upper body positioning
            'release_timing': 0.15     # Release timing relative to max jump
        }
        
        # Thresholds for similarity calculation (from existing config)
        self.thresholds = {
            'release_timing': {
                'base_threshold': 0.1,  # seconds
                'low': config.RELEASE_TIMING_DIFF_LOW,
                'medium': config.RELEASE_TIMING_DIFF_MEDIUM,
                'high': config.RELEASE_TIMING_DIFF_HIGH
            },
            'body_tilt': {
                'base_threshold': 10.0,  # degrees
                'low': config.RELEASE_TILT_DIFF_LOW,
                'medium': config.RELEASE_TILT_DIFF_MEDIUM,
                'high': config.RELEASE_TILT_DIFF_HIGH
            },
            'ball_position_x': {
                'base_threshold': 0.1,
                'low': config.RELEASE_BALL_X_DIFF_LOW,
                'medium': config.RELEASE_BALL_X_DIFF_MEDIUM,
                'high': config.RELEASE_BALL_X_DIFF_HIGH
            },
            'ball_position_y': {
                'base_threshold': 0.1,
                'low': config.RELEASE_BALL_Y_DIFF_LOW,
                'medium': config.RELEASE_BALL_Y_DIFF_MEDIUM,
                'high': config.RELEASE_BALL_Y_DIFF_HIGH
            },
            'arm_angles': {
                'base_threshold': 15.0,  # degrees
                'low': config.RELEASE_TORSO_LOW,
                'medium': config.RELEASE_TORSO_MEDIUM,
                'high': config.RELEASE_TORSO_HIGH
            }
        }
    
    def calculate_release_similarity(self, release_analysis1: Dict, release_analysis2: Dict) -> Dict:
        """
        Calculate release posture similarity between two videos.
        
        Args:
            release_analysis1: Release analysis results from video 1
            release_analysis2: Release analysis results from video 2
            
        Returns:
            Dictionary containing similarity score and breakdown
        """
        if not release_analysis1 or not release_analysis2:
            return {
                'similarity_score': 0.0,
                'feature_scores': {},
                'error': 'Missing release analysis data'
            }
        
        feature_scores = {}
        
        # 1. Arm angles similarity (shooting arm position)
        arm_score = self._calculate_arm_angles_similarity(
            release_analysis1.get('arm_angles', {}),
            release_analysis2.get('arm_angles', {})
        )
        feature_scores['arm_angles'] = arm_score
        
        # 2. Ball position similarity (relative to eyes)
        ball_score = self._calculate_ball_position_similarity(
            release_analysis1.get('ball_position', {}),
            release_analysis2.get('ball_position', {})
        )
        feature_scores['ball_position'] = ball_score
        
        # 3. Body tilt similarity (upper body positioning)
        body_score = self._calculate_body_tilt_similarity(
            release_analysis1.get('body_analysis', {}),
            release_analysis2.get('body_analysis', {})
        )
        feature_scores['body_tilt'] = body_score
        
        # 4. Release timing similarity
        timing_score = self._calculate_release_timing_similarity(
            release_analysis1.get('release_timing', {}),
            release_analysis2.get('release_timing', {})
        )
        feature_scores['release_timing'] = timing_score
        
        # Calculate weighted overall similarity
        overall_similarity = 0.0
        total_weight = 0.0
        
        for feature, score in feature_scores.items():
            if score > 0:  # Only include features with valid scores
                weight = self.feature_weights[feature]
                overall_similarity += score * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_similarity /= total_weight
        
        return {
            'similarity_score': round(float(overall_similarity), 1),
            'feature_scores': {k: round(float(v), 1) for k, v in feature_scores.items()},
            'feature_weights': self.feature_weights,
            'total_features_used': len([s for s in feature_scores.values() if s > 0])
        }
    
    def _calculate_arm_angles_similarity(self, arm_angles1: Dict, arm_angles2: Dict) -> float:
        """Calculate similarity for arm angles (shooting arm position)"""
        if not arm_angles1 or not arm_angles2:
            return 0.0
        
        # Get left and right arm angles (focus on shooting arm)
        left_angles1 = arm_angles1.get('left_arm', {})
        right_angles1 = arm_angles1.get('right_arm', {})
        left_angles2 = arm_angles2.get('left_arm', {})
        right_angles2 = arm_angles2.get('right_arm', {})
        
        if not all([left_angles1, right_angles1, left_angles2, right_angles2]):
            return 0.0
        
        # Compare torso angles (most important for shooting form)
        left_torso1 = self._safe_average(left_angles1.get('torso_angle', {}))
        right_torso1 = self._safe_average(right_angles1.get('torso_angle', {}))
        left_torso2 = self._safe_average(left_angles2.get('torso_angle', {}))
        right_torso2 = self._safe_average(right_angles2.get('torso_angle', {}))
        
        if all(v is not None for v in [left_torso1, right_torso1, left_torso2, right_torso2]):
            left_diff = abs(left_torso1 - left_torso2)
            right_diff = abs(right_torso1 - right_torso2)
            avg_diff = (left_diff + right_diff) / 2
            
            return self._difference_to_similarity(avg_diff, 'arm_angles')
        
        return 0.0
    
    def _calculate_ball_position_similarity(self, ball_pos1: Dict, ball_pos2: Dict) -> float:
        """Calculate similarity for ball position relative to eyes"""
        if not ball_pos1 or not ball_pos2:
            return 0.0
        
        # Get ball position relative to eyes (averaged across release frames)
        rel_x1 = self._safe_float(ball_pos1.get('average_relative_x', 0))
        rel_y1 = self._safe_float(ball_pos1.get('average_relative_y', 0))
        rel_x2 = self._safe_float(ball_pos2.get('average_relative_x', 0))
        rel_y2 = self._safe_float(ball_pos2.get('average_relative_y', 0))
        
        if all(v != 0 for v in [rel_x1, rel_y1, rel_x2, rel_y2]):
            x_diff = abs(rel_x1 - rel_x2)
            y_diff = abs(rel_y1 - rel_y2)
            
            x_score = self._difference_to_similarity(x_diff, 'ball_position_x')
            y_score = self._difference_to_similarity(y_diff, 'ball_position_y')
            
            # Average horizontal and vertical scores
            return (x_score + y_score) / 2
        
        return 0.0
    
    def _calculate_body_tilt_similarity(self, body_analysis1: Dict, body_analysis2: Dict) -> float:
        """Calculate similarity for body tilt (upper body positioning)"""
        if not body_analysis1 or not body_analysis2:
            return 0.0
        
        # Get body tilt measurements
        tilt1_data = body_analysis1.get('body_tilt', {})
        tilt2_data = body_analysis2.get('body_tilt', {})
        
        if not tilt1_data or not tilt2_data:
            return 0.0
        
        tilt1 = self._safe_average(tilt1_data)
        tilt2 = self._safe_average(tilt2_data)
        
        if tilt1 is not None and tilt2 is not None:
            tilt_diff = abs(tilt1 - tilt2)
            return self._difference_to_similarity(tilt_diff, 'body_tilt')
        
        return 0.0
    
    def _calculate_release_timing_similarity(self, timing1: Dict, timing2: Dict) -> float:
        """Calculate similarity for release timing relative to max jump"""
        if not timing1 or not timing2:
            return 0.0
        
        # Get relative timing (seconds relative to max jump height)
        rel_timing1 = self._safe_float(timing1.get('relative_timing', 0))
        rel_timing2 = self._safe_float(timing2.get('relative_timing', 0))
        
        if rel_timing1 != 0 or rel_timing2 != 0:  # At least one has valid timing
            timing_diff = abs(rel_timing1 - rel_timing2)
            return self._difference_to_similarity(timing_diff, 'release_timing')
        
        return 0.0
    
    def _safe_average(self, data: Union[Dict, float, int, str]) -> Optional[float]:
        """
        Safely extract average value from release data with type checking.
        Handles both single values and dictionaries with 'average' key.
        If all values are missing, attempts interpolation from neighboring frames.
        """
        if data is None or data == 'Undefined' or data == '':
            return None
        
        # Handle direct numeric values
        if isinstance(data, (int, float)):
            return float(data)
        
        # Handle dictionary with average, min, max structure
        if isinstance(data, dict):
            if 'average' in data:
                avg_val = data['average']
                if avg_val is not None and avg_val != 'Undefined' and avg_val != '':
                    try:
                        return float(avg_val)
                    except (ValueError, TypeError):
                        pass
            
            # Fallback to min/max if average not available
            min_val = data.get('min')
            max_val = data.get('max')
            if (min_val is not None and min_val != 'Undefined' and 
                max_val is not None and max_val != 'Undefined'):
                try:
                    min_f = float(min_val)
                    max_f = float(max_val)
                    return (min_f + max_f) / 2  # Use midpoint as estimate
                except (ValueError, TypeError):
                    pass
        
        # Try direct conversion as last resort
        try:
            return float(data)
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float, return 0.0 if invalid"""
        if value is None or value == 'Undefined' or value == '':
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _difference_to_similarity(self, difference: float, feature_type: str) -> float:
        """
        Convert difference to similarity score (0-100).
        
        Target mapping:
        - 0% difference = 100 points
        - 50% of threshold = 90 points (same person different shots)
        - 100% of threshold = 70 points
        - 200% of threshold = 50 points (different posture but still shooting form)
        - 400% of threshold = 20 points
        """
        threshold_config = self.thresholds.get(feature_type, {})
        base_threshold = threshold_config.get('base_threshold', 1.0)
        
        if difference == 0:
            return 100.0
        
        # Calculate similarity using exponential decay
        normalized_diff = difference / base_threshold
        
        if normalized_diff <= 0.5:
            # Linear interpolation from 100 to 90 for very small differences
            similarity = 100 - (normalized_diff * 20)  # 100 -> 90
        elif normalized_diff <= 1.0:
            # Linear interpolation from 90 to 70 for small to medium differences
            similarity = 90 - ((normalized_diff - 0.5) * 40)  # 90 -> 70
        elif normalized_diff <= 2.0:
            # Linear interpolation from 70 to 50 for medium to large differences
            similarity = 70 - ((normalized_diff - 1.0) * 20)  # 70 -> 50
        elif normalized_diff <= 4.0:
            # Linear interpolation from 50 to 20 for large to very large differences
            similarity = 50 - ((normalized_diff - 2.0) * 15)  # 50 -> 20
        else:
            # Very large differences get low scores
            similarity = max(10, 20 - (normalized_diff - 4.0) * 5)
        
        return max(0.0, min(100.0, similarity))