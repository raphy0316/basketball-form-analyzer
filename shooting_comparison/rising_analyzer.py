"""
Rising Phase Analyzer

This module analyzes the rising phase of basketball shooting form.
It extracts windup trajectory, jump height, and timing information.
Also includes setpoint detection functionality.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import os
import cv2
import tkinter as tk
from tkinter import filedialog


class RisingAnalyzer:
    """
    Analyzer for rising phase information.
    
    Extracts key measurements from Rising and Loading-Rising phases:
    - Windup trajectory (ball, wrist, elbow) with 20-frame interpolation
    - Jump height calculation
    - Setup timing and relative timing
    - Body tilt and leg angles
    - Windup and rising timing
    - Setup point arm angles and eye-level ball position
    """
    
    def __init__(self):
        self.rising_data = {}
    
    def analyze_rising_phase(self, video_data: Dict) -> Dict:
        """
        Analyze rising phase information from video data.
        
        Args:
            video_data: Video analysis data containing frames
            
        Returns:
            Dictionary containing rising phase analysis results
        """
        frames = video_data.get('frames', [])
        if not frames:
            return {"error": "No frames available for analysis"}
        
        # Get FPS from metadata (default to 30fps)
        fps = video_data.get('metadata', {}).get('fps', 30.0)
        
        # Find all Rising and Loading-Rising frames
        rising_frames = []
        loading_rising_frames = []
        
        for frame in frames:
            phase = frame.get('phase', '')
            if phase == 'Rising':
                rising_frames.append(frame)
            elif phase == 'Loading-Rising':
                loading_rising_frames.append(frame)
        
        # Combine all rising-related frames
        all_rising_frames = loading_rising_frames + rising_frames
        
        if not all_rising_frames:
            return {"error": "No Rising or Loading-Rising frames found"}
        
        # Get loading frames for baseline foot height calculation
        loading_frames = []
        for frame in frames:
            phase = frame.get('phase', '')
            if phase == 'Loading':
                loading_frames.append(frame)
        
        # Analyze rising phase
        rising_analysis = {
            'fps': fps,
            'total_rising_frames': len(all_rising_frames),
            'rising_frames': len(rising_frames),
            'loading_rising_frames': len(loading_rising_frames),
            'total_rising_time': len(all_rising_frames) / fps,
            'windup_trajectory': self._analyze_windup_trajectory(all_rising_frames, fps),
            'jump_analysis': self._analyze_jump_height(all_rising_frames, fps, loading_frames),
            'body_analysis': self._analyze_body_angles(all_rising_frames),
            'timing_analysis': self._analyze_timing(all_rising_frames, fps),
            'setup_point_analysis': self._analyze_setup_point(all_rising_frames)
        }
        
        return rising_analysis
    
    def _analyze_windup_trajectory(self, rising_frames: List[Dict], fps: float) -> Dict:
        """
        Analyze windup trajectory from dip to setup with 20-frame interpolation.
        
        Args:
            rising_frames: List of rising frame data
            fps: Frames per second
            
        Returns:
            Dictionary containing windup trajectory analysis
        """
        # Find dip point (lowest ball position)
        dip_frame = self._find_dip_point(rising_frames)
        if not dip_frame:
            return {"error": "Dip point not found"}
        
        # Find setup point using SetpointDetector
        setup_frame = self._find_setup_point_using_setpoint_detector(rising_frames)
        if not setup_frame:
            return {"error": "Setup point not found using SetpointDetector"}
        
        # Extract trajectory between dip and setup
        trajectory_frames = self._extract_trajectory_frames(rising_frames, dip_frame, setup_frame)
        
        # Interpolate to 20 frames
        interpolated_trajectory = self._interpolate_trajectory(trajectory_frames, 20)
        
        # Normalize coordinates relative to dip position
        normalized_trajectory = self._normalize_trajectory(interpolated_trajectory, dip_frame)
        
        # Calculate trajectory curvature and path length
        ball_trajectory = normalized_trajectory['ball']
        curvature = self._calculate_trajectory_curvature(ball_trajectory)
        path_length = self._calculate_trajectory_path_length(ball_trajectory)
        
        return {
            'dip_frame': dip_frame.get('frame_index', 0),
            'setup_frame': setup_frame.get('frame_index', 0),
            'trajectory_frames': len(trajectory_frames),
            'interpolated_frames': 20,
            'ball_trajectory': normalized_trajectory['ball'],
            'wrist_trajectory': normalized_trajectory['wrist'],
            'elbow_trajectory': normalized_trajectory['elbow'],
            'trajectory_curvature': curvature,
            'trajectory_path_length': path_length,
            'dip_position': {
                'ball': self._get_ball_position(dip_frame),
                'wrist': self._get_wrist_position(dip_frame),
                'elbow': self._get_elbow_position(dip_frame)
            }
        }
    
    def _find_dip_point(self, frames: List[Dict]) -> Optional[Dict]:
        """Find the frame with the lowest ball position (dip point)."""
        lowest_frame = None
        lowest_y = float('inf')
        
        for frame in frames:
            ball_pos = self._get_ball_position(frame)
            if ball_pos and ball_pos['y'] < lowest_y:
                lowest_y = ball_pos['y']
                lowest_frame = frame
        
        return lowest_frame
    
    def _find_setup_point_using_setpoint_detector(self, frames: List[Dict]) -> Optional[Dict]:
        """Find setup point using SetpointDetector instead of eye level detection."""
        if not frames:
            return None
        
        # Convert frames to pose_data and ball_data format for SetpointDetector
        pose_data = []
        ball_data = []
        
        for frame in frames:
            # Extract pose data
            pose_info = {
                'normalized_pose': frame.get('normalized_pose', {}),
                'phase': frame.get('phase', 'Rising')  # Assume rising phase
            }
            pose_data.append(pose_info)
            
            # Extract ball data
            ball_pos = self._get_ball_position(frame)
            ball_info = {
                'center_x': ball_pos.get('x', 0.0) if ball_pos else 0.0,
                'center_y': ball_pos.get('y', 0.0) if ball_pos else 0.0
            }
            ball_data.append(ball_info)
        
        # Use SetpointDetector to find setpoints
        setpoint_detector = SetpointDetector()
        setpoints = setpoint_detector.detect_setpoint(pose_data, ball_data)
        
        # Return the first setpoint frame (if any)
        if setpoints and len(setpoints) > 0:
            setpoint_frame_idx = setpoints[0]  # Use first detected setpoint
            if setpoint_frame_idx < len(frames):
                return frames[setpoint_frame_idx]
        
        return None
    
    def _extract_trajectory_frames(self, frames: List[Dict], dip_frame: Dict, setup_frame: Dict) -> List[Dict]:
        """Extract frames between dip and setup points."""
        dip_idx = frames.index(dip_frame)
        setup_idx = frames.index(setup_frame)
        
        if dip_idx > setup_idx:
            dip_idx, setup_idx = setup_idx, dip_idx
        
        return frames[dip_idx:setup_idx + 1]
    
    def _interpolate_trajectory(self, trajectory_frames: List[Dict], target_frames: int) -> List[Dict]:
        """Interpolate trajectory to target number of frames."""
        if len(trajectory_frames) <= 1:
            return trajectory_frames
        
        # Extract positions
        ball_positions = []
        wrist_positions = []
        elbow_positions = []
        
        for frame in trajectory_frames:
            ball_pos = self._get_ball_position(frame)
            wrist_pos = self._get_wrist_position(frame)
            elbow_pos = self._get_elbow_position(frame)
            
            if ball_pos:
                ball_positions.append([ball_pos['x'], ball_pos['y']])
            if wrist_pos:
                wrist_positions.append([wrist_pos['x'], wrist_pos['y']])
            if elbow_pos:
                elbow_positions.append([elbow_pos['x'], elbow_pos['y']])
        
        # Interpolate each trajectory
        interpolated_ball = self._interpolate_positions(ball_positions, target_frames)
        interpolated_wrist = self._interpolate_positions(wrist_positions, target_frames)
        interpolated_elbow = self._interpolate_positions(elbow_positions, target_frames)
        
        # Create interpolated frames
        interpolated_frames = []
        for i in range(target_frames):
            frame_data = {
                'ball': {'x': interpolated_ball[i][0], 'y': interpolated_ball[i][1]} if interpolated_ball else None,
                'wrist': {'x': interpolated_wrist[i][0], 'y': interpolated_wrist[i][1]} if interpolated_wrist else None,
                'elbow': {'x': interpolated_elbow[i][0], 'y': interpolated_elbow[i][1]} if interpolated_elbow else None
            }
            interpolated_frames.append(frame_data)
        
        return interpolated_frames
    
    def _interpolate_positions(self, positions: List[List[float]], target_frames: int) -> List[List[float]]:
        """Interpolate position list to target number of frames."""
        if len(positions) <= 1:
            return positions
        
        # Create interpolation points
        original_indices = np.linspace(0, len(positions) - 1, len(positions))
        target_indices = np.linspace(0, len(positions) - 1, target_frames)
        
        # Interpolate x and y separately
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        interpolated_x = np.interp(target_indices, original_indices, x_coords)
        interpolated_y = np.interp(target_indices, original_indices, y_coords)
        
        return [[x, y] for x, y in zip(interpolated_x, interpolated_y)]
    
    def _normalize_trajectory(self, trajectory_frames: List[Dict], dip_frame: Dict) -> Dict:
        """Normalize trajectory coordinates relative to dip position."""
        dip_ball = self._get_ball_position(dip_frame)
        dip_wrist = self._get_wrist_position(dip_frame)
        dip_elbow = self._get_elbow_position(dip_frame)
        
        if not dip_ball:
            return {'ball': [], 'wrist': [], 'elbow': []}
        
        normalized_ball = []
        normalized_wrist = []
        normalized_elbow = []
        
        for frame in trajectory_frames:
            # Normalize ball
            if frame.get('ball'):
                norm_ball = {
                    'x': frame['ball']['x'] - dip_ball['x'],
                    'y': frame['ball']['y'] - dip_ball['y']
                }
                normalized_ball.append(norm_ball)
            
            # Normalize wrist
            if frame.get('wrist') and dip_wrist:
                norm_wrist = {
                    'x': frame['wrist']['x'] - dip_wrist['x'],
                    'y': frame['wrist']['y'] - dip_wrist['y']
                }
                normalized_wrist.append(norm_wrist)
            
            # Normalize elbow
            if frame.get('elbow') and dip_elbow:
                norm_elbow = {
                    'x': frame['elbow']['x'] - dip_elbow['x'],
                    'y': frame['elbow']['y'] - dip_elbow['y']
                }
                normalized_elbow.append(norm_elbow)
        
        return {
            'ball': normalized_ball,
            'wrist': normalized_wrist,
            'elbow': normalized_elbow
        }
    
    def _analyze_jump_height(self, rising_frames: List[Dict], fps: float, loading_frames: List[Dict] = None) -> Dict:
        """Analyze jump height and timing based on hip height changes relative to loading phase foot height."""
        # Get loading phase foot height as baseline
        loading_foot_height = self._get_loading_foot_height(loading_frames) if loading_frames else None
        
        # Find hip positions throughout rising phase
        hip_positions = []
        frame_indices = []
        
        for i, frame in enumerate(rising_frames):
            pose = frame.get('normalized_pose', {})
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            
            if self._has_valid_coordinates(left_hip, right_hip):
                hip_y = (left_hip.get('y', 0) + right_hip.get('y', 0)) / 2
                hip_positions.append(hip_y)
                frame_indices.append(i)
        
        if not hip_positions:
            return {"error": "No valid hip positions found"}
        
        # Find maximum hip height (highest point)
        max_height_idx = np.argmax(hip_positions)
        max_hip_height = hip_positions[max_height_idx]
        max_height_frame_idx = frame_indices[max_height_idx]
        
        # Calculate jump height relative to loading foot height
        if loading_foot_height is not None:
            jump_height = max_hip_height - loading_foot_height
            baseline_height = loading_foot_height
        else:
            # Fallback to relative to initial hip height
            initial_height = hip_positions[0]
            jump_height = max_hip_height - initial_height
            baseline_height = initial_height
        
        # Check if jump is significant (more than 0.01 difference)
        foot_height_diff = abs(max_hip_height - baseline_height)
        has_significant_jump = bool(foot_height_diff > 0.01)
        
        # Find setup timing
        setup_frame = self._find_setup_point_using_setpoint_detector(rising_frames)
        setup_idx = rising_frames.index(setup_frame) if setup_frame else 0
        
        # Calculate relative timing
        max_height_time = max_height_frame_idx / fps
        setup_time = setup_idx / fps
        relative_timing = setup_time - max_height_time
        
        return {
            'max_jump_height': jump_height,
            'max_height_frame': max_height_frame_idx,
            'max_height_time': max_height_time,
            'setup_frame': setup_idx,
            'setup_time': setup_time,
            'relative_timing': relative_timing,
            'hip_positions': hip_positions,
            'baseline_height': baseline_height,
            'has_significant_jump': has_significant_jump,
            'foot_height_diff': foot_height_diff
        }
    
    def _analyze_body_angles(self, rising_frames: List[Dict]) -> Dict:
        """
        Analyze body tilt and leg angles at maximum jump height point.
        
        Args:
            rising_frames: List of rising frame data
            
        Returns:
            Dictionary containing body angle measurements at max jump height
        """
        # First, find the maximum jump height frame
        max_jump_frame = self._find_max_jump_frame(rising_frames)
        
        if not max_jump_frame:
            return {"error": "Could not find maximum jump height frame"}
        
        pose = max_jump_frame.get('normalized_pose', {})
        
        # Body tilt relative to hip vertical line at max jump height
        body_tilt = self._calculate_body_tilt(pose)
        
        # Leg angles at max jump height
        leg_angles = self._calculate_leg_angles(pose)
        
        return {
            'body_tilt': body_tilt,
            'leg_angles': leg_angles,
            'max_jump_frame_index': max_jump_frame.get('frame_index', 'Unknown')
        }
    
    def _get_loading_foot_height(self, loading_frames: List[Dict]) -> Optional[float]:
        """Get average foot height from loading phase frames."""
        if not loading_frames:
            return None
        
        foot_heights = []
        
        for frame in loading_frames:
            pose = frame.get('normalized_pose', {})
            left_ankle = pose.get('left_ankle', {})
            right_ankle = pose.get('right_ankle', {})
            
            if self._has_valid_coordinates(left_ankle, right_ankle):
                foot_height = (left_ankle.get('y', 0) + right_ankle.get('y', 0)) / 2
                foot_heights.append(foot_height)
        
        if not foot_heights:
            return None
        
        # Return average foot height from loading phase
        return np.mean(foot_heights)
    
    def _find_max_jump_frame(self, rising_frames: List[Dict]) -> Optional[Dict]:
        """Find the frame with maximum jump height based on hip height."""
        max_height = -float('inf')
        max_jump_frame = None
        
        for frame in rising_frames:
            pose = frame.get('normalized_pose', {})
            
            # Get hip positions for jump height calculation
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            
            if self._has_valid_coordinates(left_hip, right_hip):
                # Calculate average hip height
                hip_height = (left_hip.get('y', 0) + right_hip.get('y', 0)) / 2
                
                if hip_height > max_height:
                    max_height = hip_height
                    max_jump_frame = frame
        
        return max_jump_frame
    
    def _analyze_timing(self, rising_frames: List[Dict], fps: float) -> Dict:
        """Analyze windup and rising timing."""
        # Find dip and setup points
        dip_frame = self._find_dip_point(rising_frames)
        setup_frame = self._find_setup_point_using_setpoint_detector(rising_frames)
        
        if not dip_frame or not setup_frame:
            return {"error": "Dip or setup point not found"}
        
        dip_idx = rising_frames.index(dip_frame)
        setup_idx = rising_frames.index(setup_frame)
        
        windup_time = (setup_idx - dip_idx) / fps
        total_rising_time = len(rising_frames) / fps
        
        return {
            'windup_time': windup_time,
            'total_rising_time': total_rising_time,
            'windup_ratio': windup_time / total_rising_time if total_rising_time > 0 else 0
        }
    
    def _get_ball_position(self, frame: Dict) -> Optional[Dict]:
        """Get ball position from frame."""
        ball = frame.get('normalized_ball', {})
        if 'center_x' in ball and 'center_y' in ball:
            return {'x': ball['center_x'], 'y': ball['center_y']}
        return None
    
    def _get_wrist_position(self, frame: Dict) -> Optional[Dict]:
        """Get wrist position from frame."""
        pose = frame.get('normalized_pose', {})
        wrist = pose.get('right_wrist', {})  # Assuming right hand
        if 'x' in wrist and 'y' in wrist:
            return {'x': wrist['x'], 'y': wrist['y']}
        return None
    
    def _get_elbow_position(self, frame: Dict) -> Optional[Dict]:
        """Get elbow position from frame."""
        pose = frame.get('normalized_pose', {})
        elbow = pose.get('right_elbow', {})  # Assuming right hand
        if 'x' in elbow and 'y' in elbow:
            return {'x': elbow['x'], 'y': elbow['y']}
        return None
    
    def _calculate_body_tilt(self, pose: Dict) -> float:
        """Calculate body tilt relative to hip vertical line."""
        left_hip = pose.get('left_hip', {})
        right_hip = pose.get('right_hip', {})
        left_shoulder = pose.get('left_shoulder', {})
        right_shoulder = pose.get('right_shoulder', {})
        
        if not self._has_valid_coordinates(left_hip, right_hip, left_shoulder, right_shoulder):
            return 0.0
        
        # Calculate hip center
        hip_center_x = (left_hip.get('x', 0) + right_hip.get('x', 0)) / 2
        hip_center_y = (left_hip.get('y', 0) + right_hip.get('y', 0)) / 2
        
        # Calculate shoulder center
        shoulder_center_x = (left_shoulder.get('x', 0) + right_shoulder.get('x', 0)) / 2
        shoulder_center_y = (left_shoulder.get('y', 0) + right_shoulder.get('y', 0)) / 2
        
        # Calculate tilt angle
        dx = shoulder_center_x - hip_center_x
        dy = shoulder_center_y - hip_center_y
        tilt_angle = np.degrees(np.arctan2(dx, abs(dy)))
        
        return tilt_angle
    
    def _calculate_leg_angles(self, pose: Dict) -> Dict:
        """Calculate leg angles relative to hip vertical line."""
        left_hip = pose.get('left_hip', {})
        right_hip = pose.get('right_hip', {})
        left_knee = pose.get('left_knee', {})
        right_knee = pose.get('right_knee', {})
        left_ankle = pose.get('left_ankle', {})
        right_ankle = pose.get('right_ankle', {})
        
        angles = {}
        
        # Left leg angles
        if self._has_valid_coordinates(left_hip, left_knee):
            left_thigh_angle = self._calculate_angle_to_vertical(left_hip, left_knee)
            angles['left_thigh_angle'] = left_thigh_angle
        
        if self._has_valid_coordinates(left_hip, left_knee, left_ankle):
            left_leg_angle = self._calculate_angle(
                left_hip.get('x', 0), left_hip.get('y', 0),
                left_knee.get('x', 0), left_knee.get('y', 0),
                left_ankle.get('x', 0), left_ankle.get('y', 0)
            )
            angles['left_leg_angle'] = left_leg_angle
        
        # Right leg angles
        if self._has_valid_coordinates(right_hip, right_knee):
            right_thigh_angle = self._calculate_angle_to_vertical(right_hip, right_knee)
            angles['right_thigh_angle'] = right_thigh_angle
        
        if self._has_valid_coordinates(right_hip, right_knee, right_ankle):
            right_leg_angle = self._calculate_angle(
                right_hip.get('x', 0), right_hip.get('y', 0),
                right_knee.get('x', 0), right_knee.get('y', 0),
                right_ankle.get('x', 0), right_ankle.get('y', 0)
            )
            angles['right_leg_angle'] = right_leg_angle
        
        return angles
    
    def _calculate_angle_to_vertical(self, point1: Dict, point2: Dict) -> float:
        """Calculate angle between line and vertical axis."""
        dx = point2.get('x', 0) - point1.get('x', 0)
        dy = point2.get('y', 0) - point1.get('y', 0)
        angle = np.degrees(np.arctan2(dx, abs(dy)))
        return angle
    
    def _calculate_angle(self, ax: float, ay: float, bx: float, by: float, 
                        cx: float, cy: float) -> float:
        """Calculate angle between three points."""
        a = np.array([ax, ay])
        b = np.array([bx, by])
        c = np.array([cx, cy])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _has_valid_coordinates(self, *points) -> bool:
        """Check if all points have valid x, y coordinates."""
        for point in points:
            if not point or 'x' not in point or 'y' not in point:
                return False
        return True
    
    def _calculate_trajectory_curvature(self, trajectory: List[Dict]) -> float:
        """
        Calculate the average curvature of the trajectory.
        
        Args:
            trajectory: List of {'x': float, 'y': float} coordinates
            
        Returns:
            Average curvature value
        """
        if len(trajectory) < 3:
            return 0.0
        
        curvatures = []
        for i in range(1, len(trajectory) - 1):
            # Get three consecutive points
            prev_point = trajectory[i - 1]
            curr_point = trajectory[i]
            next_point = trajectory[i + 1]
            
            # Calculate curvature using the formula: |(x'y'' - x''y')| / (x'^2 + y'^2)^(3/2)
            # For discrete points, we use finite differences
            dx1 = curr_point['x'] - prev_point['x']
            dy1 = curr_point['y'] - prev_point['y']
            dx2 = next_point['x'] - curr_point['x']
            dy2 = next_point['y'] - curr_point['y']
            
            # Calculate curvature
            numerator = abs(dx1 * dy2 - dx2 * dy1)
            denominator = (dx1**2 + dy1**2)**1.5
            
            if denominator > 0:
                curvature = numerator / denominator
                curvatures.append(curvature)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def _calculate_trajectory_path_length(self, trajectory: List[Dict]) -> float:
        """
        Calculate the total path length of the trajectory.
        
        Args:
            trajectory: List of {'x': float, 'y': float} coordinates
            
        Returns:
            Total path length
        """
        if len(trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(trajectory) - 1):
            point1 = trajectory[i]
            point2 = trajectory[i + 1]
            
            dx = point2['x'] - point1['x']
            dy = point2['y'] - point1['y']
            segment_length = np.sqrt(dx**2 + dy**2)
            total_length += segment_length
        
        return total_length 


    def _analyze_setup_point(self, rising_frames: List[Dict]) -> Dict:
        """
        Analyze setup point specific information including arm angles and eye-level ball position.
        
        Args:
            rising_frames: List of rising frame data
            
        Returns:
            Dictionary containing setup point analysis
        """
        # Find setup point using SetpointDetector
        setup_frame = self._find_setup_point_using_setpoint_detector(rising_frames)
        if not setup_frame:
            return {"error": "Setup point not found"}
        
        # Analyze arm angles at setup point
        arm_angles = self._analyze_setup_arm_angles(setup_frame)
        
        # Analyze ball position relative to eyes at setup point
        ball_eye_position = self._analyze_setup_ball_eye_position(setup_frame)
        
        return {
            'setup_frame_index': setup_frame.get('frame_index', 0),
            'arm_angles': arm_angles,
            'ball_eye_position': ball_eye_position
        }
    
    def _analyze_setup_arm_angles(self, setup_frame: Dict) -> Dict:
        """
        Analyze arm angles at setup point.
        
        Args:
            setup_frame: Setup point frame data
            
        Returns:
            Dictionary containing arm angle measurements
        """
        pose = setup_frame.get('normalized_pose', {})
        
        # Get joint positions
        right_shoulder = pose.get('right_shoulder', {})
        right_elbow = pose.get('right_elbow', {})
        right_wrist = pose.get('right_wrist', {})
        left_shoulder = pose.get('left_shoulder', {})
        left_hip = pose.get('left_hip', {})
        right_hip = pose.get('right_hip', {})
        
        angles = {}
        
        # Right arm angles
        if self._has_valid_coordinates(right_shoulder, right_elbow, right_wrist):
            # Shoulder-elbow-wrist angle
            shoulder_elbow_wrist = self._calculate_angle(
                right_shoulder.get('x', 0), right_shoulder.get('y', 0),
                right_elbow.get('x', 0), right_elbow.get('y', 0),
                right_wrist.get('x', 0), right_wrist.get('y', 0)
            )
            angles['shoulder_elbow_wrist'] = shoulder_elbow_wrist
        
        if self._has_valid_coordinates(right_elbow, right_shoulder, right_hip):
            # Elbow-shoulder-hip angle (armpit angle)
            elbow_shoulder_hip = self._calculate_angle(
                right_elbow.get('x', 0), right_elbow.get('y', 0),
                right_shoulder.get('x', 0), right_shoulder.get('y', 0),
                right_hip.get('x', 0), right_hip.get('y', 0)
            )
            angles['elbow_shoulder_hip'] = elbow_shoulder_hip
        
        # Calculate torso angle (shoulder-hip line relative to vertical)
        if self._has_valid_coordinates(right_shoulder, right_hip):
            torso_angle = self._calculate_angle_to_vertical(right_shoulder, right_hip)
            angles['torso_angle'] = torso_angle
        
        # Calculate arm angle relative to torso
        if 'shoulder_elbow_wrist' in angles and 'torso_angle' in angles:
            arm_torso_angle = angles['shoulder_elbow_wrist'] - angles['torso_angle']
            angles['arm_torso_angle'] = arm_torso_angle
        
        return angles
    
    def _analyze_setup_ball_eye_position(self, setup_frame: Dict) -> Dict:
        """
        Analyze ball position relative to eyes at setup point.
        
        Args:
            setup_frame: Setup point frame data
            
        Returns:
            Dictionary containing ball-eye position measurements
        """
        pose = setup_frame.get('normalized_pose', {})
        ball = setup_frame.get('normalized_ball', {})
        
        # Get eye positions
        left_eye = pose.get('left_eye', {})
        right_eye = pose.get('right_eye', {})
        
        if not (self._has_valid_coordinates(left_eye, right_eye) and 
                'center_x' in ball and 'center_y' in ball):
            return {"error": "Invalid eye or ball coordinates"}
        
        # Calculate eye center
        eye_center_x = (left_eye.get('x', 0) + right_eye.get('x', 0)) / 2
        eye_center_y = (left_eye.get('y', 0) + right_eye.get('y', 0)) / 2
        
        # Get ball position
        ball_x = ball.get('center_x', 0)
        ball_y = ball.get('center_y', 0)
        
        # Calculate relative positions
        relative_x = ball_x - eye_center_x
        relative_y = ball_y - eye_center_y
        
        return {
            'relative_x': relative_x,
            'relative_y': relative_y,
            'ball_x': ball_x,
            'ball_y': ball_y,
            'eye_center_x': eye_center_x,
            'eye_center_y': eye_center_y
        }


class SetpointDetector:
    """
    Detector for identifying setpoint transitions in basketball shooting videos.
    Setpoint is the transition point from pulling the ball up during rising to throwing it.
    """
    
    def __init__(self):
        self.setpoint_thresholds = {
            'ball_vertical_velocity_change': 0.05,
            'ball_horizontal_velocity_change': 0.03,
            'wrist_angle_change': 15.0,
            'ball_position_change': 0.02,
            'ball_trajectory_curvature': 0.1,
            'wrist_acceleration': 0.05,
            'phase_transition_weight': 0.3
        }
    
    def detect_setpoint(self, pose_data: List[Dict], ball_data: List[Dict]) -> List[int]:
        """
        Detect setpoint frames using multiple metrics.
        Only detects setpoints during rising phase when hand is above shoulder.
        
        Args:
            pose_data: List of pose data dictionaries
            ball_data: List of ball data dictionaries
            
        Returns:
            List of frame indices where setpoints are detected
        """
        if len(pose_data) < 10 or len(ball_data) < 10:
            return []
        
        setpoints_with_scores = []  # Store (frame_idx, score) tuples
        
        # Analyze different metrics
        ball_velocity_changes = self._analyze_ball_velocity_changes(ball_data)
        wrist_angle_changes = self._analyze_wrist_angle_changes(pose_data)
        ball_position_changes = self._analyze_ball_position_changes(ball_data)
        ball_trajectory_curvatures = self._analyze_ball_trajectory_curvature(ball_data)
        wrist_accelerations = self._analyze_wrist_acceleration(pose_data)
        phase_transitions = [False] * len(pose_data)  # Phase info not in current data
        
        print("🔍 Setpoint 감지 상세 분석:")
        print("=" * 60)
        
        # Detect setpoints based on combined metrics with additional conditions
        for i in range(5, len(pose_data) - 5):
            # Check if current frame is in rising phase
            if not self._is_rising_phase(pose_data[i]):
                continue
            
            # Check if hand is above shoulder
            if not self._is_hand_above_shoulder(pose_data[i]):
                continue
            
            score = self._calculate_advanced_setpoint_score(
                i, ball_velocity_changes, wrist_angle_changes, 
                ball_position_changes, ball_trajectory_curvatures,
                wrist_accelerations, phase_transitions
            )
            
            if score > 0.6:  # Threshold for setpoint detection
                setpoints_with_scores.append((i, score))
                
                # Log detailed metrics for detected setpoint
                self._log_setpoint_details(i, ball_velocity_changes, wrist_angle_changes,
                                         ball_position_changes, ball_trajectory_curvatures,
                                         wrist_accelerations, score)
        
        # Sort by score (highest first) and filter
        setpoints_with_scores.sort(key=lambda x: x[1], reverse=True)
        setpoints = self._filter_setpoints_by_score(setpoints_with_scores, pose_data, ball_data)
        
        return setpoints
    
    def _log_setpoint_details(self, frame_idx: int, ball_velocity_changes: List[float],
                             wrist_angle_changes: List[float], ball_position_changes: List[float],
                             ball_trajectory_curvatures: List[float], wrist_accelerations: List[float],
                             total_score: float):
        """Log detailed metrics for detected setpoint."""
        
        if frame_idx < 2 or frame_idx >= len(ball_velocity_changes) - 2:
            return
        
        # Get raw metric values
        velocity_raw = ball_velocity_changes[frame_idx-2]
        angle_raw = wrist_angle_changes[frame_idx-2]
        position_raw = ball_position_changes[frame_idx-2]
        curvature_raw = ball_trajectory_curvatures[frame_idx-2]
        acceleration_raw = wrist_accelerations[frame_idx-2]
        
        # Check if this frame would be rejected due to direction
        direction_rejected = velocity_raw < 0 or position_raw < 0
        
        if direction_rejected:
            return
        
        # Calculate individual scores with direction consideration
        velocity_score = min(velocity_raw / self.setpoint_thresholds['ball_vertical_velocity_change'], 1.0) if velocity_raw > 0 else 0.0
        angle_score = min(abs(angle_raw) / self.setpoint_thresholds['wrist_angle_change'], 1.0)
        position_score = min(position_raw / self.setpoint_thresholds['ball_position_change'], 1.0) if position_raw > 0 else 0.0
        curvature_score = min(curvature_raw / self.setpoint_thresholds['ball_trajectory_curvature'], 1.0)
        acceleration_score = min(abs(acceleration_raw) / self.setpoint_thresholds['wrist_acceleration'], 1.0)
        
        # Logging removed - no print statements
    
    def _is_rising_phase(self, pose_frame: Dict) -> bool:
        """
        Check if the current frame is in rising phase.
        
        Args:
            pose_frame: Pose data for current frame
            
        Returns:
            True if in rising phase, False otherwise
        """
        phase = pose_frame.get('phase', 'General')
        return phase.lower() in ['rising', 'rise']
    
    def _is_hand_above_shoulder(self, pose_frame: Dict) -> bool:
        """
        Check if the hand (wrist) is above the shoulder.
        
        Args:
            pose_frame: Pose data for current frame
            
        Returns:
            True if hand is above shoulder, False otherwise
        """
        try:
            normalized_pose = pose_frame.get('normalized_pose', {})
            
            # Get wrist position
            wrist = normalized_pose.get('right_wrist', {})
            wrist_y = wrist.get('y', 0.0)
            
            # Get shoulder position
            shoulder = normalized_pose.get('right_shoulder', {})
            shoulder_y = shoulder.get('y', 0.0)
            
            # In normalized coordinates, lower y values are higher positions
            # So we check if wrist_y < shoulder_y (wrist is above shoulder)
            return wrist_y < shoulder_y
            
        except (KeyError, TypeError):
            return False
    
    def _analyze_ball_velocity_changes(self, ball_data: List[Dict]) -> List[float]:
        """Analyze changes in ball velocity over time."""
        velocity_changes = []
        
        for i in range(2, len(ball_data) - 2):
            try:
                # Calculate vertical velocity change
                prev_y = ball_data[i-2].get('center_y', 0.0)
                curr_y = ball_data[i].get('center_y', 0.0)
                next_y = ball_data[i+2].get('center_y', 0.0)
                
                # Calculate horizontal velocity change
                prev_x = ball_data[i-2].get('center_x', 0.0)
                curr_x = ball_data[i].get('center_x', 0.0)
                next_x = ball_data[i+2].get('center_x', 0.0)
                
                # Velocity changes
                vertical_change = abs(next_y - curr_y) - abs(curr_y - prev_y)
                horizontal_change = abs(next_x - curr_x) - abs(curr_x - prev_x)
                
                # Combined velocity change score
                velocity_change = (vertical_change + horizontal_change) / 2
                velocity_changes.append(velocity_change)
                
            except (KeyError, IndexError):
                velocity_changes.append(0.0)
        
        return velocity_changes
    
    def _analyze_wrist_angle_changes(self, pose_data: List[Dict]) -> List[float]:
        """Analyze changes in wrist angle over time."""
        angle_changes = []
        
        for i in range(2, len(pose_data) - 2):
            try:
                # Get wrist and elbow positions
                curr_frame = pose_data[i].get('normalized_pose', {})
                prev_frame = pose_data[i-2].get('normalized_pose', {})
                next_frame = pose_data[i+2].get('normalized_pose', {})
                
                # Calculate wrist angles
                curr_angle = self._calculate_angle(
                    curr_frame.get('right_elbow', {}).get('x', 0.0),
                    curr_frame.get('right_elbow', {}).get('y', 0.0),
                    curr_frame.get('right_wrist', {}).get('x', 0.0),
                    curr_frame.get('right_wrist', {}).get('y', 0.0)
                )
                
                prev_angle = self._calculate_angle(
                    prev_frame.get('right_elbow', {}).get('x', 0.0),
                    prev_frame.get('right_elbow', {}).get('y', 0.0),
                    prev_frame.get('right_wrist', {}).get('x', 0.0),
                    prev_frame.get('right_wrist', {}).get('y', 0.0)
                )
                
                next_angle = self._calculate_angle(
                    next_frame.get('right_elbow', {}).get('x', 0.0),
                    next_frame.get('right_elbow', {}).get('y', 0.0),
                    next_frame.get('right_wrist', {}).get('x', 0.0),
                    next_frame.get('right_wrist', {}).get('y', 0.0)
                )
                
                # Angle change
                angle_change = abs(next_angle - curr_angle) - abs(curr_angle - prev_angle)
                angle_changes.append(angle_change)
                
            except (KeyError, IndexError):
                angle_changes.append(0.0)
        
        return angle_changes
    
    def _analyze_ball_position_changes(self, ball_data: List[Dict]) -> List[float]:
        """Analyze changes in ball position relative to hip."""
        position_changes = []
        
        for i in range(2, len(ball_data) - 2):
            try:
                # Calculate position changes
                prev_pos = ball_data[i-2].get('center_y', 0.0)
                curr_pos = ball_data[i].get('center_y', 0.0)
                next_pos = ball_data[i+2].get('center_y', 0.0)
                
                # Position change
                position_change = abs(next_pos - curr_pos) - abs(curr_pos - prev_pos)
                position_changes.append(position_change)
                
            except (KeyError, IndexError):
                position_changes.append(0.0)
        
        return position_changes
    
    def _analyze_ball_trajectory_curvature(self, ball_data: List[Dict]) -> List[float]:
        """Analyze ball trajectory curvature."""
        curvatures = []
        
        for i in range(2, len(ball_data) - 2):
            try:
                # Get three consecutive points
                prev_point = (ball_data[i-2].get('center_x', 0.0), ball_data[i-2].get('center_y', 0.0))
                curr_point = (ball_data[i].get('center_x', 0.0), ball_data[i].get('center_y', 0.0))
                next_point = (ball_data[i+2].get('center_x', 0.0), ball_data[i+2].get('center_y', 0.0))
                
                # Calculate curvature
                curvature = self._calculate_curvature(prev_point, curr_point, next_point)
                curvatures.append(curvature)
                
            except (KeyError, IndexError):
                curvatures.append(0.0)
        
        return curvatures
    
    def _analyze_wrist_acceleration(self, pose_data: List[Dict]) -> List[float]:
        """Analyze wrist acceleration patterns."""
        accelerations = []
        
        for i in range(2, len(pose_data) - 2):
            try:
                # Get wrist positions
                prev_frame = pose_data[i-2].get('normalized_pose', {})
                curr_frame = pose_data[i].get('normalized_pose', {})
                next_frame = pose_data[i+2].get('normalized_pose', {})
                
                # Calculate wrist velocities
                prev_vel = np.sqrt(
                    (curr_frame.get('right_wrist', {}).get('x', 0.0) - prev_frame.get('right_wrist', {}).get('x', 0.0))**2 +
                    (curr_frame.get('right_wrist', {}).get('y', 0.0) - prev_frame.get('right_wrist', {}).get('y', 0.0))**2
                )
                
                next_vel = np.sqrt(
                    (next_frame.get('right_wrist', {}).get('x', 0.0) - curr_frame.get('right_wrist', {}).get('x', 0.0))**2 +
                    (next_frame.get('right_wrist', {}).get('y', 0.0) - curr_frame.get('right_wrist', {}).get('y', 0.0))**2
                )
                
                # Acceleration
                acceleration = next_vel - prev_vel
                accelerations.append(acceleration)
                
            except (KeyError, IndexError):
                accelerations.append(0.0)
        
        return accelerations
    
    def _calculate_advanced_setpoint_score(self, frame_idx: int, 
                                         ball_velocity_changes: List[float],
                                         wrist_angle_changes: List[float],
                                         ball_position_changes: List[float],
                                         ball_trajectory_curvatures: List[float],
                                         wrist_accelerations: List[float],
                                         phase_transitions: List[bool]) -> float:
        """Calculate advanced setpoint score using weighted metrics with direction consideration."""
        
        if frame_idx < 2 or frame_idx >= len(ball_velocity_changes) - 2:
            return 0.0
        
        # Get raw metric values
        velocity_raw = ball_velocity_changes[frame_idx-2]
        angle_raw = wrist_angle_changes[frame_idx-2]
        position_raw = ball_position_changes[frame_idx-2]
        curvature_raw = ball_trajectory_curvatures[frame_idx-2]
        acceleration_raw = wrist_accelerations[frame_idx-2]
        
        # Check direction - only positive changes indicate upward movement (setpoint)
        # Negative changes indicate downward movement (not setpoint)
        if velocity_raw < 0 or position_raw < 0:
            return 0.0  # Reject downward movements
        
        # Calculate individual scores with direction consideration
        velocity_score = min(velocity_raw / self.setpoint_thresholds['ball_vertical_velocity_change'], 1.0) if velocity_raw > 0 else 0.0
        angle_score = min(abs(angle_raw) / self.setpoint_thresholds['wrist_angle_change'], 1.0)  # Angle can be positive or negative
        position_score = min(position_raw / self.setpoint_thresholds['ball_position_change'], 1.0) if position_raw > 0 else 0.0
        curvature_score = min(curvature_raw / self.setpoint_thresholds['ball_trajectory_curvature'], 1.0)  # Curvature is always positive
        acceleration_score = min(abs(acceleration_raw) / self.setpoint_thresholds['wrist_acceleration'], 1.0)  # Acceleration can be positive or negative
        phase_score = 1.0 if phase_transitions[frame_idx] else 0.0
        
        # Weighted combination
        weights = {
            'velocity': 0.25,
            'angle': 0.20,
            'position': 0.15,
            'curvature': 0.20,
            'acceleration': 0.15,
            'phase': 0.05
        }
        
        total_score = (
            velocity_score * weights['velocity'] +
            angle_score * weights['angle'] +
            position_score * weights['position'] +
            curvature_score * weights['curvature'] +
            acceleration_score * weights['acceleration'] +
            phase_score * weights['phase']
        )
        
        return total_score
    
    def _calculate_angle(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate angle between two points."""
        return np.degrees(np.arctan2(y2 - y1, x2 - x1))
    
    def _calculate_curvature(self, p1: Tuple[float, float], 
                            p2: Tuple[float, float], 
                            p3: Tuple[float, float]) -> float:
        """Calculate curvature of three points."""
        try:
            # Vector calculations
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Cross product for curvature
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            
            # Magnitudes
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 * mag2 == 0:
                return 0.0
            
            # Curvature
            curvature = cross_product / (mag1 * mag2)
            return abs(curvature)
            
        except:
            return 0.0
    
    def _filter_setpoints_by_score(self, setpoints_with_scores: List[Tuple[int, float]], 
                                  pose_data: List[Dict], 
                                  ball_data: List[Dict]) -> List[int]:
        """Filter setpoints by score, keeping only the highest scoring one."""
        if len(setpoints_with_scores) == 0:
            return []
        
        # Return only the highest scoring setpoint
        return [setpoints_with_scores[0][0]]  # Frame index of highest score