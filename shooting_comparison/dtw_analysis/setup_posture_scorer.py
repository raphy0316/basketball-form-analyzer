"""
Setup Posture Similarity Scorer

Calculates setup phase similarity based on posture features instead of DTW.
Uses existing setup_analysis features with weighted scoring system.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import sys
import os

# Add parent directory to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shooting_comparison import config

class SetupPostureScorer:
    """
    Calculate setup phase similarity based on posture features.
    
    Target scores:
    - Same person different shots: ~90 points
    - Different posture but still shooting form: ~50 points
    """
    
    def __init__(self):
        
        # Feature weights for setup posture similarity
        self.feature_weights = {
            'leg_angles': 0.35,        # Hip-knee-ankle angles (most important - basic posture)
            'ball_position': 0.25,     # Ball-hip distances (shooting preparation)  
            'stance_position': 0.25,   # Foot positions (stability and stance)
            'upper_body_lean': 0.15    # Shoulder tilt (posture)
        }
        
        # Thresholds for similarity calculation (from existing config)
        self.thresholds = {
            'leg_angles': {
                'base_threshold': 10.0,  # degrees
                'low': config.SETUP_HIP_KNEE_ANKLE_ANGLES_LOW,
                'medium': config.SETUP_HIP_KNEE_ANKLE_ANGLES_MEDIUM,
                'high': config.SETUP_HIP_KNEE_ANKLE_ANGLES_HIGH
            },
            'ball_position_vertical': {
                'base_threshold': 0.1,
                'low': getattr(config, 'SETUP_POINT_BALL_EYE_DIFF_Y_LOW', 0.15),
                'medium': getattr(config, 'SETUP_POINT_BALL_EYE_DIFF_Y_MEDIUM', 0.25),
                'high': getattr(config, 'SETUP_POINT_BALL_EYE_DIFF_Y_HIGH', 0.35)
            },
            'ball_position_horizontal': {
                'base_threshold': 0.1,
                'low': getattr(config, 'SETUP_POINT_BALL_EYE_DIFF_X_LOW', 0.15),
                'medium': getattr(config, 'SETUP_POINT_BALL_EYE_DIFF_X_MEDIUM', 0.25),
                'high': getattr(config, 'SETUP_POINT_BALL_EYE_DIFF_X_HIGH', 0.35)
            },
            'stance_width': {
                'base_threshold': 0.05,
                'low': config.SETUP_STANCE_WIDTH_DIST_LOW,
                'medium': config.SETUP_STANCE_WIDTH_DIST_MEDIUM,
                'high': config.SETUP_STANCE_WIDTH_DIST_HIGH
            },
            'foot_front_back': {
                'base_threshold': 0.05,  # Similar to stance width
                'low': 0.1,
                'medium': 0.2,
                'high': 0.3
            },
            'shoulder_tilt': {
                'base_threshold': 5.0,  # degrees
                'low': config.SETUP_TILT_DIFF_LOW,
                'medium': config.SETUP_TILT_DIFF_MEDIUM,
                'high': config.SETUP_TILT_DIFF_HIGH
            }
        }
    
    def calculate_setup_similarity(self, setup_analysis1: Dict, setup_analysis2: Dict) -> Dict:
        """
        Calculate setup posture similarity between two videos.
        
        Args:
            setup_analysis1: Setup analysis results from video 1
            setup_analysis2: Setup analysis results from video 2
            
        Returns:
            Dictionary containing similarity score and breakdown
        """
        if not setup_analysis1 or not setup_analysis2:
            return {
                'similarity_score': 0.0,
                'feature_scores': {},
                'error': 'Missing setup analysis data'
            }
        
        feature_scores = {}
        
        # 1. Leg angles similarity
        leg_score = self._calculate_leg_angles_similarity(
            setup_analysis1.get('hip_knee_ankle_angles', {}),
            setup_analysis2.get('hip_knee_ankle_angles', {})
        )
        feature_scores['leg_angles'] = leg_score
        
        # 2. Ball position similarity
        ball_score = self._calculate_ball_position_similarity(
            setup_analysis1.get('ball_hip_distances', {}),
            setup_analysis2.get('ball_hip_distances', {})
        )
        feature_scores['ball_position'] = ball_score
        
        # 3. Stance position similarity
        stance_score = self._calculate_stance_position_similarity(
            setup_analysis1.get('foot_positions', {}),
            setup_analysis2.get('foot_positions', {})
        )
        feature_scores['stance_position'] = stance_score
        
        # 4. Upper body lean similarity
        lean_score = self._calculate_upper_body_lean_similarity(
            setup_analysis1.get('shoulder_tilt', {}),
            setup_analysis2.get('shoulder_tilt', {})
        )
        feature_scores['upper_body_lean'] = lean_score
        
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
    
    def _calculate_leg_angles_similarity(self, angles1: Dict, angles2: Dict) -> float:
        """Calculate similarity for hip-knee-ankle angles"""
        if not angles1 or not angles2:
            return 0.0
        
        left1 = self._safe_float(angles1.get('left', {}).get('average', 0))
        right1 = self._safe_float(angles1.get('right', {}).get('average', 0))
        left2 = self._safe_float(angles2.get('left', {}).get('average', 0))
        right2 = self._safe_float(angles2.get('right', {}).get('average', 0))
        
        if left1 == 0 or right1 == 0 or left2 == 0 or right2 == 0:
            return 0.0
        
        left_diff = abs(left1 - left2)
        right_diff = abs(right1 - right2)
        avg_diff = (left_diff + right_diff) / 2
        
        return self._difference_to_similarity(avg_diff, 'leg_angles')
    
    def _calculate_ball_position_similarity(self, ball_dist1: Dict, ball_dist2: Dict) -> float:
        """Calculate similarity for ball-hip distances"""
        if not ball_dist1 or not ball_dist2:
            return 0.0
        
        vert1 = self._safe_float(ball_dist1.get('average_vertical', 0))
        horiz1 = self._safe_float(ball_dist1.get('average_horizontal', 0))
        vert2 = self._safe_float(ball_dist2.get('average_vertical', 0))
        horiz2 = self._safe_float(ball_dist2.get('average_horizontal', 0))
        
        if vert1 == 0 or horiz1 == 0 or vert2 == 0 or horiz2 == 0:
            return 0.0
        
        vert_diff = abs(vert1 - vert2)
        horiz_diff = abs(horiz1 - horiz2)
        
        vert_score = self._difference_to_similarity(vert_diff, 'ball_position_vertical')
        horiz_score = self._difference_to_similarity(horiz_diff, 'ball_position_horizontal')
        
        # Average vertical and horizontal scores
        return (vert_score + horiz_score) / 2
    
    def _calculate_stance_position_similarity(self, foot_pos1: Dict, foot_pos2: Dict) -> float:
        """Calculate similarity for foot positions (stance width + front-back positioning)"""
        if not foot_pos1 or not foot_pos2:
            return 0.0
        
        left_foot1 = foot_pos1.get('left_foot', {})
        right_foot1 = foot_pos1.get('right_foot', {})
        left_foot2 = foot_pos2.get('left_foot', {})
        right_foot2 = foot_pos2.get('right_foot', {})
        
        if not all([left_foot1, right_foot1, left_foot2, right_foot2]):
            return 0.0
        
        # Get foot positions
        left_x1 = self._safe_float(left_foot1.get('average_x', 0))
        right_x1 = self._safe_float(right_foot1.get('average_x', 0))
        left_y1 = self._safe_float(left_foot1.get('average_y', 0))
        right_y1 = self._safe_float(right_foot1.get('average_y', 0))
        
        left_x2 = self._safe_float(left_foot2.get('average_x', 0))
        right_x2 = self._safe_float(right_foot2.get('average_x', 0))
        left_y2 = self._safe_float(left_foot2.get('average_y', 0))
        right_y2 = self._safe_float(right_foot2.get('average_y', 0))
        
        if any(v == 0 for v in [left_x1, right_x1, left_y1, right_y1, left_x2, right_x2, left_y2, right_y2]):
            return 0.0
        
        # Calculate stance width (Y-axis distance between feet)
        stance_width1 = abs(left_y1 - right_y1)
        stance_width2 = abs(left_y2 - right_y2)
        width_diff = abs(stance_width1 - stance_width2)
        width_score = self._difference_to_similarity(width_diff, 'stance_width')
        
        # Calculate front-back positioning (X-axis difference between feet)
        front_back_diff1 = left_x1 - right_x1  # Which foot is more forward
        front_back_diff2 = left_x2 - right_x2
        front_back_similarity = abs(front_back_diff1 - front_back_diff2)
        front_back_score = self._difference_to_similarity(front_back_similarity, 'foot_front_back')
        
        # Average stance width and front-back positioning scores
        return (width_score + front_back_score) / 2
    
    def _calculate_upper_body_lean_similarity(self, shoulder_tilt1: Dict, shoulder_tilt2: Dict) -> float:
        """Calculate similarity for shoulder tilt (upper body lean)"""
        if not shoulder_tilt1 or not shoulder_tilt2:
            return 0.0
        
        tilt1 = self._safe_float(shoulder_tilt1.get('average', 0))
        tilt2 = self._safe_float(shoulder_tilt2.get('average', 0))
        
        if tilt1 == 0 or tilt2 == 0:
            return 0.0
        
        tilt_diff = abs(tilt1 - tilt2)
        return self._difference_to_similarity(tilt_diff, 'shoulder_tilt')
    
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
        # Designed to give ~90 at 50% threshold, ~70 at 100% threshold, ~50 at 200% threshold
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
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float, return 0.0 if invalid"""
        if value is None or value == 'Undefined' or value == '':
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0