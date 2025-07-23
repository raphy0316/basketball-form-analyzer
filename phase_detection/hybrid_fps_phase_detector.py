# -*- coding: utf-8 -*-
"""
Hybrid FPS Phase Detector
Uses original data with ball-size based Set-up detection and torso + FPS proportional thresholds
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_phase_detector import BasePhaseDetector

class HybridFPSPhaseDetector(BasePhaseDetector):
    """
    Hybrid FPS Phase Detector that combines torso-based and ball-size-based thresholds.
    """
    
    # Base thresholds (relative to torso length)
    BASE_THRESHOLDS = {
        "movement": 0.040,      # 2% of torso length for Set-up→Loading (hip/shoulder movement)
        "elbow_relative": 0.015, # 1% of torso length for elbow relative movement
        "wrist_relative": 0.040, # 1.5% of torso length for wrist relative movement
        "ball_relative": 0.045,  # 1.8% of torso length for ball relative movement
        "ball_size_multiplier": 1.6,  # Ball radius multiplier for threshold
        "min_elbow_angle": 110,  # Minimum elbow angle (degrees)
        "wrist_absolute": 0.065, # 2% of torso length for wrist absolute movement (Rising detection)
        "rise_cancellation": 0.010, # 1.5% of torso length for shoulder/hip rising cancellation (optimized)
        "rising_cancel_relative": 0.025, # 0.75% of torso length for rising cancellation relative movement
        "rising_cancel_absolute": 0.040, # 1.5% of torso length for rising cancellation absolute movement
        "ball_cancel_relative": 0.030, # 1% of torso length for ball rising cancellation relative movement
        "ball_cancel_absolute": 0.045, # 2% of torso length for ball rising cancellation absolute movement
    }
    
    def __init__(self, min_phase_duration: int = 1, noise_threshold: int = 4):
        """
        Initialize hybrid FPS phase detector.
        
        Args:
            min_phase_duration: Minimum frames a phase must last
            noise_threshold: Threshold for noise filtering
        """
        super().__init__(min_phase_duration, noise_threshold)
        self.fps = 30.0  # Default FPS, will be updated during detection
        self.torso_length = 1.0  # Default torso length, will be updated during detection
        
        # Cancellation tracking variables
        self.ball_drop_frames = 0  # Track consecutive frames where ball is dropped
        self.shoulder_hip_rise_frames = 0  # Track consecutive frames where shoulder/hip rise
        
    def set_fps(self, fps: float):
        """Set FPS for threshold calculations"""
        self.fps = fps
        
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
    
    def calculate_fps_adjusted_threshold(self, base_threshold: float) -> float:
        """
        Calculate FPS-adjusted threshold.
        
        Args:
            base_threshold: Base threshold value
            
        Returns:
            FPS-adjusted threshold
        """
        # FPS adjustment factor
        # Higher FPS = more frames per second = smaller movement per frame
        # Lower FPS = fewer frames per second = larger movement per frame
        fps_factor = 30.0 / self.fps  # Normalize to 30fps
        
        return base_threshold * fps_factor
    
    def calculate_hybrid_threshold(self, pose: Dict, threshold_type: str = "movement") -> float:
        """
        Calculate hybrid threshold (torso + FPS proportional).
        
        Args:
            pose: Pose data
            threshold_type: Type of threshold ("movement", "wrist_relative", "elbow_relative", "ball_relative", "ball_distance")
            
        Returns:
            Hybrid threshold value
        """
        # Get stable torso length
        torso_length = self.get_stable_torso_length(pose)
        
        # Get base threshold
        base_threshold = self.BASE_THRESHOLDS.get(threshold_type, 0.15)
        
        # Calculate threshold relative to torso length
        torso_relative_threshold = base_threshold * torso_length
        
        # Apply FPS adjustment for movement and relative thresholds
        if threshold_type in ["movement", "wrist_relative", "elbow_relative", "ball_relative"]:
            adjusted_threshold = self.calculate_fps_adjusted_threshold(torso_relative_threshold)
        else:
            # For ball_distance, use torso-relative threshold without FPS adjustment
            adjusted_threshold = torso_relative_threshold
        
        return adjusted_threshold
    
    def calculate_ball_wrist_threshold(self, pose: Dict, ball_info: Optional[Dict]) -> float:
        """
        Calculate ball-wrist distance threshold based on ball size only.
        
        Args:
            pose: Pose data
            ball_info: Ball information
            
        Returns:
            Threshold for ball-wrist distance
        """
        # If ball info is available, use ball size based threshold
        if ball_info is not None:
            ball_width = ball_info.get('width', 0)
            ball_height = ball_info.get('height', 0)
            ball_radius = (ball_width + ball_height) / 4
            
            # Get ball size multiplier from base thresholds (not affected by torso length)
            ball_size_multiplier = self.BASE_THRESHOLDS.get("ball_size_multiplier", 1.3)
            
            # Use ball size based threshold
            ball_size_threshold = ball_radius * ball_size_multiplier
            
            return ball_size_threshold
        
        # Fallback to a default threshold if ball info is not available
        return 50.0  # Default value
    
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
            **kwargs: Additional parameters (fps, selected_hand, etc.)
            
        Returns:
            Next phase name
        """
        # Update FPS if provided
        if 'fps' in kwargs:
            self.set_fps(kwargs['fps'])
        
        # Get selected hand (default to left if not provided)
        selected_hand = kwargs.get('selected_hand', 'left')
        
        # Get current frame data
        pose = self.get_pose_info(frame_idx, pose_data)
        ball_info = self.get_ball_info(frame_idx, ball_data)
        
        # Check for cancellation conditions first (like original)
        cancellation_result = self._is_cancellation_condition(current_phase, frame_idx, pose_data, ball_data, selected_hand)
        if cancellation_result:
            # Reset tracking variables when phase changes
            self.ball_drop_frames = 0
            self.shoulder_hip_rise_frames = 0
            return cancellation_result
        
        # Calculate hybrid thresholds
        movement_threshold = self.calculate_hybrid_threshold(pose, "movement")
        ball_distance_threshold = self.calculate_ball_wrist_threshold(pose, ball_info)
        
        # Get selected hand keypoints
        selected_shoulder, selected_elbow, selected_wrist = self.get_selected_hand_keypoints(pose, selected_hand)
        
        # Check if required keypoints exist
        if not all([selected_shoulder, selected_elbow, selected_wrist]):
            return current_phase
        
        # Calculate shoulder position
        shoulder_y = selected_shoulder.get('y', 0)
        
        # Calculate elbow angles for selected hand
        # Elbow angle calculation validation
        if (selected_shoulder and selected_elbow and selected_wrist and
            'x' in selected_shoulder and 'y' in selected_shoulder and
            'x' in selected_elbow and 'y' in selected_elbow and
            'x' in selected_wrist and 'y' in selected_wrist):
            
            left_angle = self.calculate_angle(
                selected_shoulder.get('x', 0), selected_shoulder.get('y', 0),
                selected_elbow.get('x', 0), selected_elbow.get('y', 0),
                selected_wrist.get('x', 0), selected_wrist.get('y', 0)
            )
        else:
            left_angle = 0  # Default value if invalid
        
        # Get selected hand position
        wrist_x, wrist_y = self.get_selected_hand_position(pose, selected_hand)
        
        # Calculate movement changes
        d_wrist_y = 0
        d_hip_y = 0
        if frame_idx > 0:
            # Get the most recent valid previous frame
            prev_frame_idx = self._get_previous_valid_frame(frame_idx, pose_data, ball_data)
            if prev_frame_idx is not None:
                prev_pose = self.get_pose_info(prev_frame_idx, pose_data)
                prev_wrist_x, prev_wrist_y = self.get_selected_hand_position(prev_pose, selected_hand)
                
                d_wrist_y = wrist_y - prev_wrist_y
                
                # Calculate hip movement
                left_hip = pose.get('left_hip', {'y': None})
                right_hip = pose.get('right_hip', {'y': None})
                left_hip_y = left_hip.get('y', None)
                right_hip_y = right_hip.get('y', None)
                if left_hip_y is not None and right_hip_y is not None:
                    hip_y = (left_hip_y + right_hip_y) / 2  # 평균값 사용
                elif left_hip_y is not None:
                    hip_y = left_hip_y
                elif right_hip_y is not None:
                    hip_y = right_hip_y
                else:
                    hip_y = 0
                    
                prev_left_hip = prev_pose.get('left_hip', {'y': None})
                prev_right_hip = prev_pose.get('right_hip', {'y': None})
                prev_left_hip_y = prev_left_hip.get('y', None)
                prev_right_hip_y = prev_right_hip.get('y', None)
                if prev_left_hip_y is not None and prev_right_hip_y is not None:
                    prev_hip_y = (prev_left_hip_y + prev_right_hip_y) / 2  # 평균값 사용
                elif prev_left_hip_y is not None:
                    prev_hip_y = prev_left_hip_y
                elif prev_right_hip_y is not None:
                    prev_hip_y = prev_right_hip_y
                else:
                    prev_hip_y = 0
                    
                d_hip_y = hip_y - prev_hip_y
        
        # 1. General → Set-up: Ball is held in hand (torso + ball size based)
        if current_phase == "General":
            # 필요한 값만 검사: 공 정보와 선택된 손 키포인트
            if (ball_info is not None and 
                selected_wrist and 'x' in selected_wrist and 'y' in selected_wrist):
                
                ball_x = ball_info.get('center_x', 0)
                ball_y = ball_info.get('center_y', 0)
                
                # Combined torso and ball-size based threshold
                ball_distance_threshold = self.calculate_ball_wrist_threshold(pose, ball_info)
                
                distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5
                
                if distance < ball_distance_threshold:
                    # Reset tracking variables when phase changes
                    self.ball_drop_frames = 0
                    self.shoulder_hip_rise_frames = 0
                    return "Set-up"
        
        # 2. Set-up → Loading: Hip AND shoulder moving downward
        if current_phase == "Set-up":
            # 필요한 값만 검사: 엉덩이와 어깨 키포인트
            if ('left_hip' in pose and 'right_hip' in pose and 
                'left_shoulder' in pose and 'right_shoulder' in pose and
                frame_idx > 0):
                
                # 엉덩이와 어깨 데이터 유효성 검사
                left_hip = pose.get('left_hip', {})
                right_hip = pose.get('right_hip', {})
                left_shoulder = pose.get('left_shoulder', {})
                right_shoulder = pose.get('right_shoulder', {})
                
                if (isinstance(left_hip, dict) and 'y' in left_hip and
                    isinstance(right_hip, dict) and 'y' in right_hip and
                    isinstance(left_shoulder, dict) and 'y' in left_shoulder and
                    isinstance(right_shoulder, dict) and 'y' in right_shoulder):
                    
                    conditions = []
                    
                    # Calculate hip and shoulder positions
                    left_hip_y = left_hip.get('y', None)
                    right_hip_y = right_hip.get('y', None)
                    if left_hip_y is not None and right_hip_y is not None:
                        hip_y = (left_hip_y + right_hip_y) / 2  # Use average
                    elif left_hip_y is not None:
                        hip_y = left_hip_y
                    elif right_hip_y is not None:
                        hip_y = right_hip_y
                    else:
                        hip_y = 0
                    
                    # Calculate changes from previous frame
                    prev_frame_idx = self._get_previous_valid_frame(frame_idx, pose_data, ball_data)
                    if prev_frame_idx is not None:
                        prev_pose = self.get_pose_info(prev_frame_idx, pose_data)
                        prev_left_hip = prev_pose.get('left_hip', {'y': None})
                        prev_right_hip = prev_pose.get('right_hip', {'y': None})
                        prev_left_hip_y = prev_left_hip.get('y', None)
                        prev_right_hip_y = prev_right_hip.get('y', None)
                        if prev_left_hip_y is not None and prev_right_hip_y is not None:
                            prev_hip_y = (prev_left_hip_y + prev_right_hip_y) / 2  # Use average
                        elif prev_left_hip_y is not None:
                            prev_hip_y = prev_left_hip_y
                        elif prev_right_hip_y is not None:
                            prev_hip_y = prev_right_hip_y
                        else:
                            prev_hip_y = 0
                        
                        prev_shoulder_data = self.calculate_keypoint_averages(prev_pose, ['shoulder'])
                        prev_shoulder_y = prev_shoulder_data['shoulder']['y']
                        
                        d_hip_y = hip_y - prev_hip_y
                        d_shoulder_y = shoulder_y - prev_shoulder_y
                        
                        # Hip moving downward (y-coordinate increasing)
                        if d_hip_y > movement_threshold:
                            conditions.append("hip_down")
                        
                        # Shoulder moving downward
                        if d_shoulder_y > movement_threshold:
                            conditions.append("shoulder_down")
                        
                        # Knee angles decreasing (squatting motion)
                        # Knee angle calculation validation
                        if ('left_hip' in pose and 'right_hip' in pose and
                            'left_knee' in pose and 'right_knee' in pose and
                            'left_ankle' in pose and 'right_ankle' in pose and
                            'left_hip' in prev_pose and 'right_hip' in prev_pose and
                            'left_knee' in prev_pose and 'right_knee' in prev_pose and
                            'left_ankle' in prev_pose and 'right_ankle' in prev_pose):
                            
                            # Calculate current knee angles
                            left_knee_angle = self.calculate_angle(
                                pose.get('left_hip', {}).get('x', 0), pose.get('left_hip', {}).get('y', 0),
                                pose.get('left_knee', {}).get('x', 0), pose.get('left_knee', {}).get('y', 0),
                                pose.get('left_ankle', {}).get('x', 0), pose.get('left_ankle', {}).get('y', 0)
                            )
                            right_knee_angle = self.calculate_angle(
                                pose.get('right_hip', {}).get('x', 0), pose.get('right_hip', {}).get('y', 0),
                                pose.get('right_knee', {}).get('x', 0), pose.get('right_knee', {}).get('y', 0),
                                pose.get('right_ankle', {}).get('x', 0), pose.get('right_ankle', {}).get('y', 0)
                            )
                            
                            # Calculate previous knee angles
                            prev_left_knee_angle = self.calculate_angle(
                                prev_pose.get('left_hip', {}).get('x', 0), prev_pose.get('left_hip', {}).get('y', 0),
                                prev_pose.get('left_knee', {}).get('x', 0), prev_pose.get('left_knee', {}).get('y', 0),
                                prev_pose.get('left_ankle', {}).get('x', 0), prev_pose.get('left_ankle', {}).get('y', 0)
                            )
                            prev_right_knee_angle = self.calculate_angle(
                                prev_pose.get('right_hip', {}).get('x', 0), prev_pose.get('right_hip', {}).get('y', 0),
                                prev_pose.get('right_knee', {}).get('x', 0), prev_pose.get('right_knee', {}).get('y', 0),
                                prev_pose.get('right_ankle', {}).get('x', 0), prev_pose.get('right_ankle', {}).get('y', 0)
                            )
                        else:
                            # Set default values if invalid
                            left_knee_angle = 180
                            right_knee_angle = 180
                            prev_left_knee_angle = 180
                            prev_right_knee_angle = 180
                        
                        # Check if knee angles are decreasing (squatting)
                        left_knee_decreasing = left_knee_angle < prev_left_knee_angle
                        right_knee_decreasing = right_knee_angle < prev_right_knee_angle
                        
                        if left_knee_decreasing and right_knee_decreasing:
                            conditions.append("knees_bending")
                        
                        # ALL THREE conditions must be met: hip down, shoulder down, knees bending
                        if len(conditions) == 3:
                            # Reset tracking variables when phase changes
                            self.ball_drop_frames = 0
                            self.shoulder_hip_rise_frames = 0
                            return "Loading"
        
        # 2.5. Set-up → Rising: Skip Loading if Rising conditions met directly
        if current_phase == "Set-up":
            # Check only required values: ball info, selected hand keypoints, hip
            if (ball_info is not None and 
                selected_elbow and selected_wrist and
                'x' in selected_elbow and 'y' in selected_elbow and
                'x' in selected_wrist and 'y' in selected_wrist and
                'left_hip' in pose and 'right_hip' in pose and
                frame_idx > 0):
                
                # Hip data validation
                left_hip = pose.get('left_hip', {})
                right_hip = pose.get('right_hip', {})
                
                if (isinstance(left_hip, dict) and 'y' in left_hip and
                    isinstance(right_hip, dict) and 'y' in right_hip):
                    
                    conditions = []
                    
                    # Calculate hip position
                    left_hip_y = left_hip.get('y', None)
                    right_hip_y = right_hip.get('y', None)
                    if left_hip_y is not None and right_hip_y is not None:
                        hip_y = (left_hip_y + right_hip_y) / 2  # Use average
                    elif left_hip_y is not None:
                        hip_y = left_hip_y
                    elif right_hip_y is not None:
                        hip_y = right_hip_y
                    else:
                        hip_y = 0
                    
                    prev_frame_idx = self._get_previous_valid_frame(frame_idx, pose_data, ball_data)
                    if prev_frame_idx is not None:
                        prev_pose = self.get_pose_info(prev_frame_idx, pose_data)
                        
                        # Get selected hand keypoints for previous frame
                        prev_selected_shoulder, prev_selected_elbow, prev_selected_wrist = self.get_selected_hand_keypoints(prev_pose, selected_hand)
                        
                        # Calculate previous hip position
                        prev_left_hip = prev_pose.get('left_hip', {'y': None})
                        prev_right_hip = prev_pose.get('right_hip', {'y': None})
                        prev_left_hip_y = prev_left_hip.get('y', None)
                        prev_right_hip_y = prev_right_hip.get('y', None)
                        if prev_left_hip_y is not None and prev_right_hip_y is not None:
                            prev_hip_y = (prev_left_hip_y + prev_right_hip_y) / 2  # Use average
                        elif prev_left_hip_y is not None:
                            prev_hip_y = prev_left_hip_y
                        elif prev_right_hip_y is not None:
                            prev_hip_y = prev_right_hip_y
                        else:
                            prev_hip_y = 0
                        
                        # Use selected hand keypoints instead of averages
                        elbow_y = selected_elbow.get('y', 0)
                        prev_elbow_y = prev_selected_elbow.get('y', 0)
                        
                        wrist_y = selected_wrist.get('y', 0)
                        prev_wrist_y = prev_selected_wrist.get('y', 0)
                        
                        # Calculate relative movement (compared to hip)
                        d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                        d_wrist_relative = (wrist_y - prev_wrist_y) - (hip_y - prev_hip_y)
                        
                        # Calculate absolute movement
                        d_wrist_absolute = wrist_y - prev_wrist_y
                        
                        # Wrist moving upward relative to hip
                        wrist_relative_threshold = self.calculate_hybrid_threshold(pose, "wrist_relative")
                        
                        # Wrist moving upward in absolute terms
                        wrist_absolute_threshold = self.calculate_hybrid_threshold(pose, "wrist_absolute")
                        
                        # Both relative and absolute conditions must be met
                        if d_wrist_relative < -wrist_relative_threshold:
                            conditions.append("wrist_up_relative")
                        
                        if d_wrist_absolute < -wrist_absolute_threshold:
                            conditions.append("wrist_up_absolute")
                        
                        # BOTH relative AND absolute conditions must be met
                        if len(conditions) == 2:
                            # Reset tracking variables when phase changes
                            self.ball_drop_frames = 0
                            self.shoulder_hip_rise_frames = 0
                            return "Rising"
        
        # 3. Loading → Rising: Wrist and elbow moving upward (relative to torso)
        if current_phase == "Loading":
            # Check only required values: selected hand keypoints and hip
            if (selected_elbow and selected_wrist and
                'x' in selected_elbow and 'y' in selected_elbow and
                'x' in selected_wrist and 'y' in selected_wrist and
                'left_hip' in pose and 'right_hip' in pose and
                frame_idx > 0):
                
                # Hip data validation
                left_hip = pose.get('left_hip', {})
                right_hip = pose.get('right_hip', {})
                
                if (isinstance(left_hip, dict) and 'y' in left_hip and
                    isinstance(right_hip, dict) and 'y' in right_hip):
                    
                    conditions = []
                
                prev_frame_idx = self._get_previous_valid_frame(frame_idx, pose_data, ball_data)
                if prev_frame_idx is not None:
                    prev_pose = self.get_pose_info(prev_frame_idx, pose_data)
                    
                    # Get selected hand keypoints for previous frame
                    prev_selected_shoulder, prev_selected_elbow, prev_selected_wrist = self.get_selected_hand_keypoints(prev_pose, selected_hand)
                    
                    # Use selected hand keypoints instead of averages
                    elbow_y = selected_elbow.get('y', 0)
                    prev_elbow_y = prev_selected_elbow.get('y', 0)
                    
                    wrist_y = selected_wrist.get('y', 0)
                    prev_wrist_y = prev_selected_wrist.get('y', 0)
                    
                    # Calculate relative movement (compared to hip)
                    d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                    d_wrist_relative = (wrist_y - prev_wrist_y) - (hip_y - prev_hip_y)
                    
                    # Calculate absolute movement
                    d_wrist_absolute = wrist_y - prev_wrist_y
                    
                    # Wrist moving upward relative to hip
                    wrist_relative_threshold = self.calculate_hybrid_threshold(pose, "wrist_relative")
                    
                    # Wrist moving upward in absolute terms
                    wrist_absolute_threshold = self.calculate_hybrid_threshold(pose, "wrist_absolute")
                    
                    # Both relative and absolute conditions must be met
                    if d_wrist_relative < -wrist_relative_threshold:
                        conditions.append("wrist_up_relative")
                    
                    if d_wrist_absolute < -wrist_absolute_threshold:
                        conditions.append("wrist_up_absolute")
                    
                    # BOTH relative AND absolute conditions must be met
                    if len(conditions) == 2:
                        # Reset tracking variables when phase changes
                        self.ball_drop_frames = 0
                        self.shoulder_hip_rise_frames = 0
                        return "Rising"
        
        # 4. Rising → Release: Ball is released with proper form
        if current_phase == "Rising":
            # Check only required values: ball info and selected hand keypoints
            if (ball_info is not None and 
                selected_wrist and selected_shoulder and selected_elbow and
                'x' in selected_wrist and 'y' in selected_wrist and
                'x' in selected_shoulder and 'y' in selected_shoulder and
                'x' in selected_elbow and 'y' in selected_elbow):
                
                # Check if ball is released from hand
                ball_x = ball_info.get('center_x', 0)
                ball_y = ball_info.get('center_y', 0)
                
                # Use Euclidean distance with ball-size based threshold
                distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5
                ball_distance_threshold = self.calculate_ball_wrist_threshold(pose, ball_info)
                ball_released = distance > ball_distance_threshold
                
                if ball_released:
                    # Check angle and posture conditions when ball is released
                    # Calculate elbow angle for selected hand
                    # Elbow angle calculation validation
                    if (selected_shoulder and selected_elbow and selected_wrist and
                        'x' in selected_shoulder and 'y' in selected_shoulder and
                        'x' in selected_elbow and 'y' in selected_elbow and
                        'x' in selected_wrist and 'y' in selected_wrist):
                        
                        left_angle = self.calculate_angle(
                            selected_shoulder.get('x', 0), selected_shoulder.get('y', 0),
                            selected_elbow.get('x', 0), selected_elbow.get('y', 0),
                            selected_wrist.get('x', 0), selected_wrist.get('y', 0)
                        )
                    else:
                        left_angle = 0  # Default value if invalid
                    
                    # Angle condition: elbow angle must be >= 110 degrees
                    angle_ok = left_angle >= self.BASE_THRESHOLDS['min_elbow_angle']
                    
                    # Posture condition: wrist must be above shoulder
                    wrist_above_shoulder = wrist_y < shoulder_y
                    
                    # Ball position condition: ball must be above shoulder
                    ball_above_shoulder = ball_y < shoulder_y
                    
                    # If all conditions are met, Release; otherwise cancel to General
                    if angle_ok and wrist_above_shoulder and ball_above_shoulder:
                        # Reset tracking variables when phase changes
                        self.ball_drop_frames = 0
                        self.shoulder_hip_rise_frames = 0
                        return "Release"
                    else:
                        # Reset tracking variables when phase changes
                        self.ball_drop_frames = 0
                        self.shoulder_hip_rise_frames = 0
                        return "General"  # Cancel to General if conditions not met
        
        # 5. Release → Follow-through: Ball is released and wrist is above shoulder
        if current_phase == "Release":
            # Check only required values: ball info and selected hand keypoints
            if (ball_info is not None and 
                selected_wrist and 'x' in selected_wrist and 'y' in selected_wrist):
                
                # Calculate distance between ball and wrist
                ball_x = ball_info.get('center_x', 0)
                ball_y = ball_info.get('center_y', 0)
                wrist_x = selected_wrist.get('x', 0)
                wrist_y = selected_wrist.get('y', 0)
                
                distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5
                
                # Follow-through uses longer distance threshold than Release
                ball_distance_threshold = self.calculate_ball_wrist_threshold(pose, ball_info)
                follow_through_threshold = ball_distance_threshold * 1.5  # 1.5x longer distance than Release
                
                if distance > follow_through_threshold:
                    # Reset tracking variables when phase changes
                    self.ball_drop_frames = 0
                    self.shoulder_hip_rise_frames = 0
                    return "Follow-through"
        
        # 6. Follow-through → General: Transition to General when wrist goes below eyes
        if current_phase == "Follow-through":
            # Check only required values: selected wrist, both eyes
            if (selected_wrist and 'y' in selected_wrist and
                'left_eye' in pose and 'right_eye' in pose and
                'y' in pose['left_eye'] and 'y' in pose['right_eye']):
                wrist_y = selected_wrist['y']
                left_eye_y = pose['left_eye']['y']
                right_eye_y = pose['right_eye']['y']
                eye_y = (left_eye_y + right_eye_y) / 2
                if wrist_y > eye_y:  # When wrist goes below eyes
                    self.ball_drop_frames = 0
                    self.shoulder_hip_rise_frames = 0
                    return "General"
            # Maintain phase if conditions not met
            return current_phase
        
        return current_phase
    
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



    def _is_cancellation_condition(self, current_phase: str, frame_idx: int, pose_data: List[Dict], ball_data: List[Dict], selected_hand: str):
        """Check if current phase should be cancelled and return to Set-up (like original)"""
        
        # Get current frame data
        pose = self.get_pose_info(frame_idx, pose_data)
        ball_info = self.get_ball_info(frame_idx, ball_data)
        
        # Get selected hand keypoints
        selected_shoulder, selected_elbow, selected_wrist = self.get_selected_hand_keypoints(pose, selected_hand)
        
        # Check if required keypoints exist
        if not all([selected_shoulder, selected_elbow, selected_wrist]):
            return False
        
        # Calculate shoulder position
        shoulder_y = selected_shoulder.get('y', 0)
        
        # Select closest wrist to ball
        wrist_x, wrist_y = self.get_selected_hand_position(pose, selected_hand)
        
        # Calculate ball position
        ball_x = ball_info.get('center_x', 0) if ball_info else 0
        ball_y = ball_info.get('center_y', 0) if ball_info else 0
        ball_detected = ball_info is not None
        
        # Calculate Euclidean distance between ball and wrist
        ball_wrist_distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5 if ball_detected else float('inf')
        
        # Check cancellation conditions based on current phase
        if current_phase == "Loading":
            # Loading cancellation: Two conditions
            # 1. Ball dropped (with minimum frame requirement) -> General
            # 2. Shoulder/hip rising -> Set-up
            
            # Calculate minimum frames based on FPS (3 frames for 30fps)
            min_frames = max(1, round(self.fps * 4 / 30))
            
            # Condition 1: Ball dropped (minimum N frames)
            if (ball_detected and 
                selected_wrist and 'x' in selected_wrist and 'y' in selected_wrist):
                
                # Use combined torso and ball-size based threshold
                ball_distance_threshold = self.calculate_ball_wrist_threshold(pose, ball_info)
                
                if ball_wrist_distance > ball_distance_threshold:
                    self.ball_drop_frames += 1
                else:
                    self.ball_drop_frames = 0  # Reset counter
                
                # Return to General if ball dropped for minimum frames
                if self.ball_drop_frames >= min_frames:  # Dynamic application based on FPS
                    self.ball_drop_frames = 0  # Reset counter
                    return "General"
            
            # Condition 2: Shoulder/hip rising (immediate cancellation)
            if frame_idx > 0:
                prev_pose = self.get_pose_info(frame_idx - 1, pose_data)
                
                # Get shoulder and hip positions
                shoulder_y = selected_shoulder.get('y', 0)
                prev_shoulder_y = prev_pose.get(selected_hand + '_shoulder', {}).get('y', 0)
                
                # Calculate hip position (average of left and right)
                left_hip = pose.get('left_hip', {})
                right_hip = pose.get('right_hip', {})
                left_hip_y = left_hip.get('y', 0)
                right_hip_y = right_hip.get('y', 0)
                hip_y = (left_hip_y + right_hip_y) / 2 if left_hip_y != 0 and right_hip_y != 0 else (left_hip_y or right_hip_y or 0)
                
                prev_left_hip = prev_pose.get('left_hip', {})
                prev_right_hip = prev_pose.get('right_hip', {})
                prev_left_hip_y = prev_left_hip.get('y', 0)
                prev_right_hip_y = prev_right_hip.get('y', 0)
                prev_hip_y = (prev_left_hip_y + prev_right_hip_y) / 2 if prev_left_hip_y != 0 and prev_right_hip_y != 0 else (prev_left_hip_y or prev_right_hip_y or 0)
                
                # Calculate movement
                d_shoulder_y = shoulder_y - prev_shoulder_y
                d_hip_y = hip_y - prev_hip_y
                
                # Calculate threshold for rising cancellation (separate from movement threshold)
                rise_cancellation_threshold = self.calculate_hybrid_threshold(pose, "rise_cancellation")
                
                # Check if shoulder and hip are rising (negative values mean going up)
                shoulder_rising = d_shoulder_y < -rise_cancellation_threshold
                hip_rising = d_hip_y < -rise_cancellation_threshold
                
                if shoulder_rising and hip_rising:
                    return "Set-up"
        
        elif current_phase == "Rising":
            # Rising cancellation: Hand moving down relative to hip
            # Check only required values: selected hand keypoints and hip
            if (frame_idx > 0 and
                selected_elbow and selected_wrist and
                'x' in selected_elbow and 'y' in selected_elbow and
                'x' in selected_wrist and 'y' in selected_wrist and
                'left_hip' in pose and 'right_hip' in pose):
                
                prev_pose = self.get_pose_info(frame_idx - 1, pose_data)
                
                # Hip data validation
                left_hip = pose.get('left_hip', {})
                right_hip = pose.get('right_hip', {})
                prev_left_hip = prev_pose.get('left_hip', {})
                prev_right_hip = prev_pose.get('right_hip', {})
                
                if (isinstance(left_hip, dict) and 'y' in left_hip and
                    isinstance(right_hip, dict) and 'y' in right_hip and
                    isinstance(prev_left_hip, dict) and 'y' in prev_left_hip and
                    isinstance(prev_right_hip, dict) and 'y' in prev_right_hip):
                    
                    # Get selected hand keypoints for previous frame
                    prev_selected_shoulder, prev_selected_elbow, prev_selected_wrist = self.get_selected_hand_keypoints(prev_pose, selected_hand)
                    
                    # Check if required keypoints exist
                    if not all([prev_selected_shoulder, prev_selected_elbow, prev_selected_wrist]):
                        return False # Should not happen if _get_previous_valid_frame works correctly
                    
                    # Calculate hip position
                    left_hip_y = left_hip.get('y', None)
                    right_hip_y = right_hip.get('y', None)
                    if left_hip_y is not None and right_hip_y is not None:
                        hip_y = (left_hip_y + right_hip_y) / 2  # Use average
                    elif left_hip_y is not None:
                        hip_y = left_hip_y
                    elif right_hip_y is not None:
                        hip_y = right_hip_y
                    else:
                        hip_y = 0
                    
                    prev_left_hip_y = prev_left_hip.get('y', None)
                    prev_right_hip_y = prev_right_hip.get('y', None)
                    if prev_left_hip_y is not None and prev_right_hip_y is not None:
                        prev_hip_y = (prev_left_hip_y + prev_right_hip_y) / 2  # Use average
                    elif prev_left_hip_y is not None:
                        prev_hip_y = prev_left_hip_y
                    elif prev_right_hip_y is not None:
                        prev_hip_y = prev_right_hip_y
                    else:
                        prev_hip_y = 0
                    
                    # Use selected hand keypoints instead of averages
                    elbow_y = selected_elbow.get('y', 0)
                    prev_elbow_y = prev_selected_elbow.get('y', 0)
                    
                    wrist_y = selected_wrist.get('y', 0)
                    prev_wrist_y = prev_selected_wrist.get('y', 0)
                    
                    # Calculate relative movement
                    d_wrist_relative = (wrist_y - prev_wrist_y) - (hip_y - prev_hip_y)
                    d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                    d_ball_relative = 0
                    if ball_detected and frame_idx > 0:
                        prev_ball_info = self.get_ball_info(frame_idx - 1, ball_data)
                        if prev_ball_info:
                            prev_ball_y = prev_ball_info.get('center_y', 0)
                            ball_y = ball_info.get('center_y', 0)
                            d_ball_relative = ball_y - prev_ball_y - (hip_y - prev_hip_y)
                    
                    # Calculate absolute movement
                    d_wrist_absolute = wrist_y - prev_wrist_y
                    d_ball_absolute = 0
                    if ball_detected and frame_idx > 0:
                        prev_ball_info = self.get_ball_info(frame_idx - 1, ball_data)
                        if prev_ball_info:
                            prev_ball_y = prev_ball_info.get('center_y', 0)
                            ball_y = ball_info.get('center_y', 0)
                            d_ball_absolute = ball_y - prev_ball_y
                    
                    # Rising cancellation: Different conditions based on wrist position relative to shoulder
                    wrist_above_shoulder = wrist_y < shoulder_y
                    
                    if wrist_above_shoulder:
                        # When above shoulder: cancel when wrist goes below shoulder
                        # Check if current wrist is below shoulder
                        wrist_below_shoulder = wrist_y >= shoulder_y
                        
                        if ball_detected:
                            # Check if ball is also below shoulder
                            ball_below_shoulder = ball_y >= shoulder_y
                            
                            if wrist_below_shoulder and ball_below_shoulder:
                                return "Set-up"
                        else:
                            if wrist_below_shoulder:
                                return "Set-up"
                    else:
                        # When below shoulder: use cancel-specific threshold (cancel when moving down)
                        wrist_cancel_relative_threshold = self.calculate_hybrid_threshold(pose, "rising_cancel_relative")
                        wrist_cancel_absolute_threshold = self.calculate_hybrid_threshold(pose, "rising_cancel_absolute")
                        ball_cancel_relative_threshold = self.calculate_hybrid_threshold(pose, "ball_cancel_relative")
                        ball_cancel_absolute_threshold = self.calculate_hybrid_threshold(pose, "ball_cancel_absolute")
                        
                        wrist_moving_down_relative = d_wrist_relative > wrist_cancel_relative_threshold
                        ball_moving_down_relative = d_ball_relative > ball_cancel_relative_threshold
                        wrist_moving_down_absolute = d_wrist_absolute > wrist_cancel_absolute_threshold
                        ball_moving_down_absolute = d_ball_absolute > ball_cancel_absolute_threshold
                        
                        if ball_detected:
                            if (wrist_moving_down_relative and ball_moving_down_relative and 
                                wrist_moving_down_absolute and ball_moving_down_absolute):
                                return "Set-up"
                        else:
                            if wrist_moving_down_relative and wrist_moving_down_absolute:
                                return "Set-up"
         
        return None 