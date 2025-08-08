"""
DTW Analysis Configuration

Configuration parameters for DTW-based shooting motion analysis.
"""

# DTW Feature weights based on shooting importance
DTW_FEATURE_WEIGHTS = {
    'ball_wrist_trajectory': 0.30,      # Most critical for accuracy
    'shooting_arm_kinematics': 0.25,    # Core shooting mechanics
    'lower_body_stability': 0.15,       # Foundation and balance
    'phase_timing_patterns': 0.15,      # Timing consistency
    'body_alignment': 0.15              # Overall posture
}

# Phase importance for shooting analysis
PHASE_IMPORTANCE_WEIGHTS = {
    'Setup': 0.10,          # Basic stance
    'Loading': 0.20,        # Power preparation
    'Rising': 0.25,         # Motion initiation  
    'Release': 0.35,        # Most critical moment
    'Follow-through': 0.10  # Consistency and finish
}

# DTW constraints for different feature types
DTW_CONSTRAINTS = {
    'trajectory_2d': {
        'window': 0.35,         # 0.2에서 0.35로 증가 (35% Sakoe-Chiba band) - ball_wrist_trajectory를 위해 더 관대하게
        'max_dist': 4.0,        # 3.0에서 4.0으로 증가
        'max_step': 5,          # 4에서 5로 증가
        'max_length_diff': 0.5  # 0.4에서 0.5로 증가
    },
    'ball_wrist_special': {
        'window': 0.1,          # 10% band for ball-wrist trajectory (극도로 엄격하게)
        'max_dist': 1.0,        # 극도로 엄격한 거리 제약
        'max_step': 1,          # 극도로 엄격한 스텝 제약
        'max_length_diff': 0.2  # 극도로 엄격한 길이 차이 제약
    },
    'kinematics': {
        'window': 0.15,         # 15% band for kinematics - 극도로 엄격하게
        'max_dist': 1.0,        # 2.5에서 1.0으로 감소
        'max_step': 1,          # 3에서 1로 감소
        'max_length_diff': 0.2  # 0.3에서 0.2로 감소
    },
    'stability': {
        'window': 0.3,          # 0.2에서 0.3으로 증가 (30% band for stability features)
        'max_dist': 1.5,        # 1.0에서 1.5로 증가
        'max_step': 3,          # 2에서 3으로 증가
        'max_length_diff': 0.5  # 0.4에서 0.5로 증가
    },
    'timing': {
        'window': 0.15,         # 15% band for timing patterns - phase_timing_patterns를 위해 극도로 엄격하게
        'max_dist': 0.5,        # 0.8에서 0.5로 감소
        'max_step': 1,          # 2에서 1로 감소
        'max_length_diff': 0.2  # 0.4에서 0.2로 감소
    }
}

# Similarity score conversion parameters
SIMILARITY_CONVERSION = {
    'trajectory_2d': {
        'max_expected_dist': 8.0,        # 5.0에서 8.0으로 증가 (ball_wrist_trajectory를 위해 더 관대하게)
        'scaling_factor': 0.4            # 0.6에서 0.4로 감소
    },
    'ball_wrist_special': {
        'max_expected_dist': 1.5,        # ball-wrist trajectory를 위한 극도로 엄격한 설정
        'scaling_factor': 1.5            # 극도로 엄격한 스케일링
    },
    'kinematics': {
        'max_expected_dist': 40.0,       # 60.0에서 40.0으로 감소 - 극도로 엄격한 설정
        'scaling_factor': 1.5            # 1.2에서 1.5로 증가 - 극도로 엄격한 스케일링
    },
    'stability': {
        'max_expected_dist': 1.5,        # 더욱 엄격한 설정
        'scaling_factor': 1.0            # 더욱 엄격한 스케일링
    },
    'timing': {
        'max_expected_dist': 0.4,        # 0.8에서 0.4로 감소 (phase_timing_patterns를 위해 극도로 엄격하게)
        'scaling_factor': 1.8            # 1.2에서 1.8로 증가
    }
}

# Subfeature weights for combined analysis
SUBFEATURE_WEIGHTS = {
    'ball_wrist_trajectory': {
        'ball_trajectory': 0.4,
        'wrist_trajectory': 0.35,
        'ball_wrist_distance': 0.25
    },
    'shooting_arm_kinematics': {
        'elbow_angles': 0.35,
        'shoulder_trajectory': 0.25,
        'elbow_trajectory': 0.25,
        'wrist_trajectory': 0.15
    },
    'lower_body_stability': {
        'hip_trajectory': 0.4,
        'knee_angles': 0.35,
        'stance_stability': 0.25
    },
    'phase_timing_patterns': {
        'phase_durations': 0.6,
        'transition_timing': 0.4
    },
    'body_alignment': {
        'shoulder_tilt': 0.4,
        'torso_angle': 0.35,
        'head_stability': 0.25
    }
}

# Similarity grade thresholds with enhanced differentiation
SIMILARITY_GRADES = {
    'A+': 95,    # 매우 유사한 경우
    'A': 90,     # 유사한 경우
    'A-': 85,    # 약간 유사한 경우
    'B+': 80,    # 중간 유사한 경우
    'B': 75,     # 보통 유사한 경우
    'B-': 70,    # 약간 다른 경우
    'C+': 65,    # 다른 경우
    'C': 60,     # 많이 다른 경우
    'C-': 55,    # 매우 다른 경우
    'D+': 50,    # 거의 다른 경우
    'D': 45,     # 완전히 다른 경우
    'D-': 40,    # 매우 다른 경우
    'F+': 35,    # 극도로 다른 경우
    'F': 30,     # 완전히 다른 경우
    'F-': 25     # 전혀 다른 경우
}

# Analysis confidence levels
CONFIDENCE_THRESHOLDS = {
    'high': 0.85,       # 85%+ successful analysis
    'medium': 0.70,     # 70%+ successful analysis
    'low': 0.50         # 50%+ successful analysis
}