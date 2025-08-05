"""
Rising Phase Analyzer

This module analyzes the rising phase of basketball shooting form.
It extracts windup trajectory, jump height, and timing information.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json


class RisingAnalyzer:
    """
    Analyzer for rising phase information.
    
    Extracts key measurements from Rising and Loading-Rising phases:
    - Windup trajectory (ball, wrist, elbow) with 20-frame interpolation
    - Jump height calculation
    - Setup timing and relative timing
    - Body tilt and leg angles
    - Windup and rising timing
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
            'timing_analysis': self._analyze_timing(all_rising_frames, fps)
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
        
        # Find setup point (ball reaches eye level or higher)
        setup_frame = self._find_setup_point(rising_frames)
        if not setup_frame:
            return {"error": "Setup point not found"}
        
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
    
    def _find_setup_point(self, frames: List[Dict]) -> Optional[Dict]:
        """Find the frame where ball reaches eye level or higher (setup point)."""
        # Get average eye height from frames
        eye_heights = []
        for frame in frames:
            pose = frame.get('normalized_pose', {})
            left_eye = pose.get('left_eye', {})
            right_eye = pose.get('right_eye', {})
            
            if self._has_valid_coordinates(left_eye, right_eye):
                eye_y = (left_eye.get('y', 0) + right_eye.get('y', 0)) / 2
                eye_heights.append(eye_y)
        
        if not eye_heights:
            return None
        
        avg_eye_height = np.mean(eye_heights)
        
        # Find first frame where ball is at or above eye level
        for frame in frames:
            ball_pos = self._get_ball_position(frame)
            if ball_pos and ball_pos['y'] <= avg_eye_height:
                return frame
        
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
        setup_frame = self._find_setup_point(rising_frames)
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
        setup_frame = self._find_setup_point(rising_frames)
        
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