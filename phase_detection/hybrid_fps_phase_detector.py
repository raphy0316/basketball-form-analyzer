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
    Hybrid phase detector using original data with:
    - Ball-size based Set-up detection
    - Torso + FPS proportional thresholds
    """
    
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
        self.frame_width = None
        self.frame_height = None

    def set_frame_dimensions(self, width: int, height: int):
        self.frame_width = width
        self.frame_height = height

    def get_aspect_ratio(self) -> Optional[float]:
        if self.frame_width and self.frame_height:
            return self.frame_width / self.frame_height
        return 1

    def set_fps(self, fps: float):
        """Set FPS for threshold calculations"""
        self.fps = fps
        
    def calculate_torso_length(self, pose: Dict) -> float:
        """
        Calculate torso length from pose data.
        Uses left shoulder-left hip and right shoulder-right hip distances,
        returns the longer one.
        
        Args:
            pose: Pose data
            
        Returns:
            Torso length in normalized units
        """
        # Get shoulder and hip positions
        left_shoulder = pose.get('left_shoulder', {'x': 0, 'y': 0})
        right_shoulder = pose.get('right_shoulder', {'x': 0, 'y': 0})
        left_hip = pose.get('left_hip', {'x': 0, 'y': 0})
        right_hip = pose.get('right_hip', {'x': 0, 'y': 0})
        
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
        
        return torso_length
    
    def get_stable_torso_length(self, pose: Dict) -> float:
        """
        Get stable torso length (average of multiple frames).
        
        Args:
            pose: Current pose data
            
        Returns:
            Stable torso length
        """
        # For now, use current frame's torso length
        # In the future, could average over multiple frames for stability
        torso_length = self.calculate_torso_length(pose)
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
            threshold_type: Type of threshold ("movement", "relative", "ball_distance")
            
        Returns:
            Hybrid threshold value
        """
        # Get stable torso length
        torso_length = self.get_stable_torso_length(pose)
        
        # Base thresholds
        base_thresholds = {
            "movement": 0.001,      # 1% of torso length for Set-up→Loading (hip/shoulder movement)
            "relative": 0.003,     # 0.5% of torso length for Loading→Rising (wrist/elbow relative movement)
            "ball_distance": 0.0003, # Fixed distance for ball-wrist separation (not relative to torso)
        }
        
        # Get base threshold
        base_threshold = base_thresholds.get(threshold_type, 0.15)
        
        # For ball_distance, use fixed value (not relative to torso)
        if threshold_type == "ball_distance":
            return base_threshold  # Fixed value, no FPS adjustment needed
        
        # Apply FPS adjustment for movement and relative thresholds
        adjusted_threshold = self.calculate_fps_adjusted_threshold(base_threshold)
        
        return adjusted_threshold
    
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
            **kwargs: Additional parameters (fps, etc.)
            
        Returns:
            Next phase name
        """
        # Update FPS if provided
        if 'fps' in kwargs:
            self.set_fps(kwargs['fps'])
        
        # Get current frame data
        pose = self.get_pose_info(frame_idx, pose_data)
        ball_info = self.get_ball_info(frame_idx, ball_data)
        
        # Check for cancellation conditions first (like original)
        if self._is_cancellation_condition(current_phase, frame_idx, pose_data, ball_data):
            return "Set-up"  # Always return to Set-up for cancellations
        
        # Calculate hybrid thresholds
        movement_threshold = self.calculate_hybrid_threshold(pose, "movement")
        relative_threshold = self.calculate_hybrid_threshold(pose, "relative")
        ball_distance_threshold = self.calculate_hybrid_threshold(pose, "ball_distance")
        
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
        aspect_ratio = self.get_aspect_ratio()
        print(aspect_ratio)
        # Calculate elbow angles
        left_angle = self.calculate_angle(
            left_shoulder.get('x', 0) * aspect_ratio, left_shoulder.get('y', 0),
            left_elbow.get('x', 0) * aspect_ratio, left_elbow.get('y', 0),
            left_wrist.get('x', 0) * aspect_ratio, left_wrist.get('y', 0)
        )
        right_angle = self.calculate_angle(
            right_shoulder.get('x', 0) * aspect_ratio * self.frame_width, right_shoulder.get('y', 0) * self.frame_height,
            right_elbow.get('x', 0) * aspect_ratio  * self.frame_width, right_elbow.get('y', 0) * self.frame_height,
            right_wrist.get('x', 0) * aspect_ratio * self.frame_width , right_wrist.get('y', 0) * self.frame_height
        )
    
        print(f"Frame Idx: {frame_idx}, Left angle: {left_angle}, Right angle: {right_angle}")
        # print(f"right shoulder: {right_shoulder}, right elbow: {right_elbow}, right wrist: {right_wrist}")
        # Select closest wrist to ball
        wrist_x, wrist_y, selected_wrist = self.select_closest_wrist_to_ball(pose, ball_info)
        
        # Calculate movement changes
        d_wrist_y = 0
        d_hip_y = 0
        if frame_idx > 0:
            prev_pose = self.get_pose_info(frame_idx - 1, pose_data)
            prev_wrist_x, prev_wrist_y, _ = self.select_closest_wrist_to_ball(prev_pose, None)
            
            d_wrist_y = wrist_y - prev_wrist_y
            
            # Calculate hip movement
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
                
            d_hip_y = hip_y - prev_hip_y
        
        # 1. General → Set-up: Ball is held in hand (ball-size based)
        if current_phase == "General":
            if ball_info is not None:
                ball_x = ball_info.get('center_x', 0)
                ball_y = ball_info.get('center_y', 0)
                
                # Ball-size based threshold
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                close_threshold = ball_radius * 1.3  # Ball-size based
                
                distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5
                
                if distance < close_threshold:
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
                prev_shoulder_y = prev_shoulder_data['shoulder']['y']
                
                d_hip_y = hip_y - prev_hip_y
                d_shoulder_y = shoulder_y - prev_shoulder_y
                
                # Hip moving downward (y-coordinate increasing)
                if d_hip_y > movement_threshold:
                    conditions.append("hip_down")
                
                # Shoulder moving downward
                if d_shoulder_y > movement_threshold:
                    conditions.append("shoulder_down")
                
                # BOTH hip AND shoulder must be moving down
                if len(conditions) == 2:
                    return "Loading"
        
        # 2.5. Set-up → Rising: Skip Loading if Rising conditions met directly
        if current_phase == "Set-up":
            conditions = []
            
            if frame_idx > 0:
                prev_pose = self.get_pose_info(frame_idx - 1, pose_data)
                
                # Calculate elbow position
                elbow_data = self.calculate_keypoint_averages(pose, ['elbow'])
                prev_elbow_data = self.calculate_keypoint_averages(prev_pose, ['elbow'])
                
                elbow_y = elbow_data['elbow']['y']
                prev_elbow_y = prev_elbow_data['elbow']['y']
                
                # Calculate wrist position
                wrist_data = self.calculate_keypoint_averages(pose, ['wrist'])
                prev_wrist_data = self.calculate_keypoint_averages(prev_pose, ['wrist'])
                
                wrist_y = wrist_data['wrist']['y']
                prev_wrist_y = prev_wrist_data['wrist']['y']
                
                # Calculate relative movement
                d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                d_wrist_relative = (wrist_y - prev_wrist_y) - (hip_y - prev_hip_y)
                d_ball_relative = 0
                if ball_info is not None and frame_idx > 0:
                    prev_ball_info = self.get_ball_info(frame_idx - 1, ball_data)
                    if prev_ball_info:
                        prev_ball_y = prev_ball_info.get('center_y', 0)
                        ball_y = ball_info.get('center_y', 0)
                        d_ball_relative = ball_y - prev_ball_y - (hip_y - prev_hip_y)
                
                # All three conditions must be met - normalized coordinates
                if d_wrist_relative < -relative_threshold:
                    conditions.append("wrist_up_relative")
                if d_elbow_relative < -relative_threshold:
                    conditions.append("elbow_up_relative")
                if ball_info is not None and d_ball_relative < -relative_threshold:
                    conditions.append("ball_up_relative")
                
                if len(conditions) == 3:
                    return "Rising"
        
        # 3. Loading → Rising: Wrist and elbow moving upward (relative to torso)
        if current_phase == "Loading":
            conditions = []
            
            if frame_idx > 0:
                prev_pose = self.get_pose_info(frame_idx - 1, pose_data)
                
                # Calculate elbow position
                elbow_data = self.calculate_keypoint_averages(pose, ['elbow'])
                prev_elbow_data = self.calculate_keypoint_averages(prev_pose, ['elbow'])
                
                elbow_y = elbow_data['elbow']['y']
                prev_elbow_y = prev_elbow_data['elbow']['y']
                
                # Calculate wrist position
                wrist_data = self.calculate_keypoint_averages(pose, ['wrist'])
                prev_wrist_data = self.calculate_keypoint_averages(prev_pose, ['wrist'])
                
                wrist_y = wrist_data['wrist']['y']
                prev_wrist_y = prev_wrist_data['wrist']['y']
                
                # Calculate relative movement (compared to hip)
                d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                d_wrist_relative = (wrist_y - prev_wrist_y) - (hip_y - prev_hip_y)
                
                # Wrist moving upward relative to hip
                if d_wrist_relative < -relative_threshold:
                    conditions.append("wrist_up_relative")
                
                # Elbow moving upward relative to hip
                if d_elbow_relative < -relative_threshold:
                    conditions.append("elbow_up_relative")
                
                # Both conditions must be met
                if len(conditions) == 2:
                    return "Rising"
        
        # 4. Rising → Release: Ball is released with proper form
        if current_phase == "Rising":
            
            # Normal Rising → Release transition
            if ball_info is not None:
                ball_x = ball_info.get('center_x', 0)
                ball_y = ball_info.get('center_y', 0)
                
                distance = abs(ball_y - wrist_y)
                wrist_above_shoulder = wrist_y < shoulder_y
                ball_released = distance > ball_distance_threshold
                
                if ball_released:
                    ball_above_shoulder = ball_y < shoulder_y
                    
                    if (left_angle >= 110 or right_angle >= 110) and wrist_above_shoulder and ball_above_shoulder:
                        return "Release"
                    else:
                        return "Set-up"
                else:
                    ball_above_shoulder = ball_y < shoulder_y
                    
                    if (left_angle >= 110 or right_angle >= 110) and distance > ball_distance_threshold and ball_above_shoulder:
                        return "Release"
        
        # 5. Release → Follow-through: Ball has fully left the hand
        if current_phase == "Release":
            if ball_info is not None:
                ball_x = ball_info.get('center_x', 0)
                ball_y = ball_info.get('center_y', 0)
                
                ball_wrist_distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5
                
                # Use ball_distance threshold for Release→Follow-through
                ball_distance_threshold = self.calculate_hybrid_threshold(pose, "ball_distance")
                
                if ball_wrist_distance > ball_distance_threshold:
                    return "Follow-through"
        
        # 6. Follow-through → General: Wrist below eyes relative to hip
        if current_phase == "Follow-through":
            # Check if ball is caught
            if ball_info is not None:
                ball_x = ball_info.get('center_x', 0)
                ball_y = ball_info.get('center_y', 0)
                
                ball_wrist_distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5
                
                # Use ball_distance threshold for ball catch detection
                ball_distance_threshold = self.calculate_hybrid_threshold(pose, "ball_distance")
                
                if ball_wrist_distance <= ball_distance_threshold:
                    return "Set-up"
            
            # Check if wrist is below eyes relative to hip
            if frame_idx > 0:
                left_eye = pose.get('left_eye', {'y': 0})
                right_eye = pose.get('right_eye', {'y': 0})
                eye_y = max(left_eye.get('y', 0), right_eye.get('y', 0))
                
                left_wrist = pose.get('left_wrist', {'y': 0})
                right_wrist = pose.get('right_wrist', {'y': 0})
                wrist_y = min(left_wrist.get('y', 0), right_wrist.get('y', 0))
                
                eye_relative_to_hip = eye_y - hip_y
                wrist_relative_to_hip = wrist_y - hip_y
                
                if wrist_relative_to_hip > eye_relative_to_hip:
                    if frame_idx >= self.min_phase_duration:
                        return "General"
                    else:
                        return current_phase
        
        # If no conditions are met, keep current phase
        return current_phase 
    
    def _is_cancellation_condition(self, current_phase: str, frame_idx: int, pose_data: List[Dict], ball_data: List[Dict]) -> bool:
        """Check if current phase should be cancelled and return to Set-up (like original)"""
        
        # Get current frame data
        pose = self.get_pose_info(frame_idx, pose_data)
        ball_info = self.get_ball_info(frame_idx, ball_data)
        
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
        
        # Select closest wrist to ball
        wrist_x, wrist_y, selected_wrist = self.select_closest_wrist_to_ball(pose, ball_info)
        
        # Calculate ball position
        ball_x = ball_info.get('center_x', 0) if ball_info else 0
        ball_y = ball_info.get('center_y', 0) if ball_info else 0
        ball_detected = ball_info is not None
        
        # Calculate Euclidean distance between ball and wrist
        ball_wrist_distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5 if ball_detected else float('inf')
        
        # Check cancellation conditions based on current phase
        if current_phase == "Loading":
            # Loading cancellation: Ball missed
            if ball_detected:
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                close_threshold = ball_radius * 1.3
                
                if ball_wrist_distance > close_threshold:
                    return True
        
        elif current_phase == "Rising":
            # Rising cancellation: Hand moving down relative to hip
            if frame_idx > 0:
                prev_pose = self.get_pose_info(frame_idx - 1, pose_data)
                
                # Calculate hip position
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
                
                # Calculate wrist position
                wrist_data = self.calculate_keypoint_averages(pose, ['wrist'])
                prev_wrist_data = self.calculate_keypoint_averages(prev_pose, ['wrist'])
                
                wrist_y = wrist_data['wrist']['y']
                prev_wrist_y = prev_wrist_data['wrist']['y']
                
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
                
                # Calculate hybrid threshold for relative movement
                relative_threshold = self.calculate_hybrid_threshold(pose, "relative")
                
                wrist_moving_down_relative = d_wrist_relative > relative_threshold
                elbow_moving_down_relative = d_elbow_relative > relative_threshold
                
                # Rising cancellation: Hand moving down relative to hip
                if ball_detected:
                    # When ball is detected: if ball, wrist, and elbow are all moving down relative to hip, return to Set-up
                    ball_moving_down_relative = d_ball_relative > relative_threshold
                    
                    if wrist_moving_down_relative and elbow_moving_down_relative and ball_moving_down_relative:
                        return True
                else:
                    # When ball is not detected: if wrist and elbow are moving down relative to hip, return to Set-up
                    if wrist_moving_down_relative and elbow_moving_down_relative:
                        return True
        
        elif current_phase == "Release":
            # Release cancellation: Ball released but improper form
            if ball_detected:
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                close_threshold = ball_radius * 1.3
                
                distance = abs(ball_y - wrist_y)
                ball_released = distance > close_threshold
                aspect_ratio = self.get_aspect_ratio()
                if ball_released:
                    # Calculate angles
                    left_angle = self.calculate_angle(
                        left_shoulder.get('x', 0) * aspect_ratio, left_shoulder.get('y', 0),
                        left_elbow.get('x', 0) * aspect_ratio, left_elbow.get('y', 0),
                        left_wrist.get('x', 0) * aspect_ratio, left_wrist.get('y', 0)
                    )
                    right_angle = self.calculate_angle(
                        right_shoulder.get('x', 0) * aspect_ratio * self.frame_width, right_shoulder.get('y', 0) * self.frame_height,
                        right_elbow.get('x', 0) * aspect_ratio  * self.frame_width, right_elbow.get('y', 0) * self.frame_height,
                        right_wrist.get('x', 0) * aspect_ratio * self.frame_width , right_wrist.get('y', 0) * self.frame_height
                    )
    
                    wrist_above_shoulder = wrist_y < shoulder_y
                    ball_above_shoulder = ball_y < shoulder_y
                    
                    # Improper form: return to Set-up
                    if not ((left_angle >= 110 or right_angle >= 110) and wrist_above_shoulder and ball_above_shoulder):
                        return True
         
        return False 