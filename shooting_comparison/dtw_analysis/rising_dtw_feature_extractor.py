"""
Rising-specific DTW Feature Extractor

Extracts Rising phase specific DTW features for more accurate comparison.
Focuses on windup kinematics, jump dynamics, and timing patterns during Rising + Loading-Rising phases.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path to import SafeCoordinateMixin
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shooting_comparison.safe_coordinate_mixin import SafeCoordinateMixin

class RisingDTWFeatureExtractor(SafeCoordinateMixin):
    """
    Extracts Rising-specific DTW features for more accurate similarity comparison.
    
    Focuses on:
    A. rising_windup_kinematics (40%): trajectory patterns, velocity, acceleration, curvature changes
    B. rising_jump_dynamics (35%): jump mechanics, upper body/leg change patterns
    C. rising_timing_patterns (25%): timing patterns and phase-by-phase ratios
    """
    
    def __init__(self):
        # Feature weights
        self.feature_weights = {
            'rising_windup_kinematics': 0.40,
            'rising_jump_dynamics': 0.35,
            'rising_timing_patterns': 0.25
        }
        
        # Minimum data requirements
        self.min_trajectory_length = 5
        self.min_phase_frames = 3
        self.interpolation_frames = 20  # normalize windup trajectory to 20 frames
        
    def extract_rising_dtw_features(self, video_data: Dict, selected_hand: str = "right") -> Dict:
        """
        Extract Rising-specific DTW features from video data.
        
        Args:
            video_data: Video data with normalized coordinates
            selected_hand: Selected shooting hand
            
        Returns:
            Dictionary containing Rising DTW features for comparison
        """
        frames = video_data.get('frames', [])
        if not frames:
            return {'error': 'No frames available for Rising DTW feature extraction'}
        
        print(f"🔄 Extracting Rising DTW features from {len(frames)} frames...")
        self.selected_hand = selected_hand.lower()
        
        # Get Rising + Loading-Rising frames
        rising_frames = self._get_rising_frames(frames)
        
        if len(rising_frames) < self.min_phase_frames:
            return {'error': f'Insufficient Rising frames ({len(rising_frames)}) for DTW analysis'}
        
        print(f"   📊 Found {len(rising_frames)} Rising frames")
        
        # Extract Rising-specific DTW features
        rising_dtw_features = {}
        
        try:
            # A. Rising windup kinematics (40%)
            rising_dtw_features['rising_windup_kinematics'] = self._extract_windup_kinematics(rising_frames)
            
            # B. Rising jump dynamics (35%)
            rising_dtw_features['rising_jump_dynamics'] = self._extract_jump_dynamics(rising_frames)
            
            # C. Rising timing patterns (25%)
            rising_dtw_features['rising_timing_patterns'] = self._extract_timing_patterns(rising_frames)
            
            # Add metadata
            rising_dtw_features['metadata'] = {
                'total_rising_frames': len(rising_frames),
                'fps': video_data.get('metadata', {}).get('fps', 30.0),
                'selected_hand': selected_hand,
                'extraction_success': True
            }
            
            print(f"✅ Rising DTW features extracted successfully")
            
        except Exception as e:
            print(f"❌ Error extracting Rising DTW features: {e}")
            return {'error': f'Rising DTW feature extraction failed: {str(e)}'}
        
        return rising_dtw_features
    
    def _get_rising_frames(self, frames: List[Dict]) -> List[Dict]:
        """Get Rising + Loading-Rising frames"""
        rising_frames = []
        
        for frame in frames:
            phase = frame.get('phase', '')
            if phase in ['Rising', 'Loading-Rising']:
                rising_frames.append(frame)
        
        return rising_frames
    
    def _extract_windup_kinematics(self, rising_frames: List[Dict]) -> Dict:
        """
        Extract windup kinematics for Rising phase DTW analysis.
        
        Features:
        - ball trajectory change pattern (dip → setup)
        - wrist trajectory change pattern
        - elbow trajectory change pattern
        - trajectory velocity change (time series)
        - trajectory acceleration change (time series)
        - trajectory curvature change (time series)
        """
        # Find dip and setup points
        dip_frame = self._find_dip_point(rising_frames)
        setup_frame = self._find_setup_point(rising_frames)
        
        if not dip_frame or not setup_frame:
            print("⚠️ Could not find dip or setup points for windup analysis")
            return self._create_empty_windup_features()
        
        # Extract trajectory between dip and setup
        trajectory_frames = self._extract_trajectory_frames(rising_frames, dip_frame, setup_frame)
        
        if len(trajectory_frames) < 3:
            print("⚠️ Insufficient trajectory frames for windup analysis")
            return self._create_empty_windup_features()
        
        # Interpolate to standard length
        interpolated_trajectory = self._interpolate_trajectory(trajectory_frames, self.interpolation_frames)
        
        # Normalize relative to dip position
        normalized_trajectory = self._normalize_trajectory(interpolated_trajectory, dip_frame)
        
        # Extract trajectory patterns
        ball_trajectory = normalized_trajectory.get('ball', [])
        wrist_trajectory = normalized_trajectory.get('wrist', [])
        elbow_trajectory = normalized_trajectory.get('elbow', [])
        
        # Calculate velocities and accelerations
        ball_velocities = self._calculate_trajectory_velocity(ball_trajectory)
        wrist_velocities = self._calculate_trajectory_velocity(wrist_trajectory)
        elbow_velocities = self._calculate_trajectory_velocity(elbow_trajectory)
        
        ball_accelerations = self._calculate_trajectory_acceleration(ball_trajectory)
        wrist_accelerations = self._calculate_trajectory_acceleration(wrist_trajectory)
        elbow_accelerations = self._calculate_trajectory_acceleration(elbow_trajectory)
        
        # Calculate curvatures
        ball_curvatures = self._calculate_trajectory_curvature_series(ball_trajectory)
        wrist_curvatures = self._calculate_trajectory_curvature_series(wrist_trajectory)
        elbow_curvatures = self._calculate_trajectory_curvature_series(elbow_trajectory)
        
        return {
            'ball_trajectory': ball_trajectory,
            'wrist_trajectory': wrist_trajectory,
            'elbow_trajectory': elbow_trajectory,
            'ball_velocities': ball_velocities,
            'wrist_velocities': wrist_velocities,
            'elbow_velocities': elbow_velocities,
            'ball_accelerations': ball_accelerations,
            'wrist_accelerations': wrist_accelerations,
            'elbow_accelerations': elbow_accelerations,
            'ball_curvatures': ball_curvatures,
            'wrist_curvatures': wrist_curvatures,
            'elbow_curvatures': elbow_curvatures,
            'dip_frame_idx': rising_frames.index(dip_frame),
            'setup_frame_idx': rising_frames.index(setup_frame),
            'trajectory_length': len(trajectory_frames),
            'feature_type': 'rising_windup_kinematics'
        }
    
    def _extract_jump_dynamics(self, rising_frames: List[Dict]) -> Dict:
        """
        Extract jump dynamics for Rising phase DTW analysis.
        
        Features:
        - hip height change pattern
        - jump velocity change (time series)
        - jump acceleration change (time series)
        - upper body tilt change pattern
        - leg angle change pattern
        - jump timing pattern
        """
        hip_heights = []
        body_tilts = []
        left_leg_angles = []
        right_leg_angles = []
        
        for frame in rising_frames:
            pose = frame.get('normalized_pose', {})
            
            # Hip height
            hip_center = self._safe_hip_center(pose)
            hip_height = hip_center['y'] if hip_center else np.nan
            hip_heights.append(hip_height)
            
            # Body tilt
            body_tilt = self._calculate_body_tilt(pose)
            body_tilts.append(body_tilt if body_tilt is not None else np.nan)
            
            # Leg angles
            left_angle = self._calculate_leg_angle(pose, 'left')
            right_angle = self._calculate_leg_angle(pose, 'right')
            left_leg_angles.append(left_angle if left_angle is not None else np.nan)
            right_leg_angles.append(right_angle if right_angle is not None else np.nan)
        
        # Interpolate missing values
        hip_heights = self._interpolate_series_1d(hip_heights)
        body_tilts = self._interpolate_series_1d(body_tilts)
        left_leg_angles = self._interpolate_series_1d(left_leg_angles)
        right_leg_angles = self._interpolate_series_1d(right_leg_angles)
        
        # Calculate velocities and accelerations
        jump_velocities = self._calculate_velocity(hip_heights)
        jump_accelerations = self._calculate_acceleration(hip_heights)
        body_tilt_velocities = self._calculate_velocity(body_tilts)
        left_leg_velocities = self._calculate_velocity(left_leg_angles)
        right_leg_velocities = self._calculate_velocity(right_leg_angles)
        
        # Find jump timing patterns
        jump_timing = self._analyze_jump_timing_pattern(hip_heights, rising_frames)
        
        return {
            'hip_heights': hip_heights,
            'body_tilts': body_tilts,
            'left_leg_angles': left_leg_angles,
            'right_leg_angles': right_leg_angles,
            'jump_velocities': jump_velocities,
            'jump_accelerations': jump_accelerations,
            'body_tilt_velocities': body_tilt_velocities,
            'left_leg_velocities': left_leg_velocities,
            'right_leg_velocities': right_leg_velocities,
            'jump_timing_pattern': jump_timing,
            'feature_type': 'rising_jump_dynamics'
        }
    
    def _extract_timing_patterns(self, rising_frames: List[Dict]) -> Dict:
        """
        Extract timing patterns for Rising phase DTW analysis.
        
        Features:
        - dip_point → setup_point timing
        - setup_point → release timing  
        - overall rising timing
        - windup ratio change
        - phase-by-phase timing ratio
        """
        # Find key timing points
        dip_frame = self._find_dip_point(rising_frames)
        setup_frame = self._find_setup_point(rising_frames)
        
        total_frames = len(rising_frames)
        
        if dip_frame and setup_frame:
            dip_idx = rising_frames.index(dip_frame)
            setup_idx = rising_frames.index(setup_frame)
            
            # Timing ratios
            dip_to_setup_ratio = (setup_idx - dip_idx) / total_frames if total_frames > 0 else 0
            setup_to_end_ratio = (total_frames - setup_idx) / total_frames if total_frames > 0 else 0
            windup_ratio = dip_to_setup_ratio
            
        else:
            dip_idx = 0
            setup_idx = total_frames // 2  # Default to middle
            dip_to_setup_ratio = 0.5
            setup_to_end_ratio = 0.5
            windup_ratio = 0.5
        
        # Create timing pattern as multi-dimensional time series
        timing_pattern = []
        for i in range(total_frames):
            time_progress = i / total_frames if total_frames > 0 else 0
            
            # Phase markers
            is_dip = 1.0 if i == dip_idx else 0.0
            is_setup = 1.0 if i == setup_idx else 0.0
            
            # Progressive timing features
            dip_progress = max(0, (i - dip_idx) / (setup_idx - dip_idx)) if setup_idx > dip_idx else 0
            setup_progress = max(0, (i - setup_idx) / (total_frames - setup_idx)) if total_frames > setup_idx else 0
            
            timing_features = [
                time_progress,        # Overall progress
                is_dip,              # Dip point marker
                is_setup,            # Setup point marker
                dip_progress,        # Progress from dip to setup
                setup_progress,      # Progress from setup to end
                dip_to_setup_ratio,  # Dip to setup timing ratio
                setup_to_end_ratio,  # Setup to end timing ratio
                windup_ratio         # Overall windup ratio
            ]
            timing_pattern.append(timing_features)
        
        return {
            'timing_pattern': timing_pattern,
            'dip_to_setup_ratio': dip_to_setup_ratio,
            'setup_to_end_ratio': setup_to_end_ratio,
            'windup_ratio': windup_ratio,
            'total_rising_time': total_frames,
            'dip_frame_idx': dip_idx,
            'setup_frame_idx': setup_idx,
            'feature_type': 'rising_timing_patterns'
        }
    
    def _find_dip_point(self, frames: List[Dict]) -> Optional[Dict]:
        """Find the frame with the lowest ball position (dip point)"""
        lowest_frame = None
        lowest_y = float('-inf')
        
        for frame in frames:
            ball_pos = self._get_ball_position(frame)
            if ball_pos and ball_pos['y'] > lowest_y:
                lowest_y = ball_pos['y']
                lowest_frame = frame
        
        return lowest_frame
    
    def _find_setup_point(self, frames: List[Dict]) -> Optional[Dict]:
        """Find setup point using simple heuristic (can be enhanced with SetpointDetector)"""
        if not frames:
            return None
        
        # For now, use simple heuristic - find highest ball position after dip
        dip_frame = self._find_dip_point(frames)
        if not dip_frame:
            return frames[-1] if frames else None
        
        dip_idx = frames.index(dip_frame)
        setup_candidates = frames[dip_idx:]
        
        highest_frame = None
        highest_y = float('inf')
        
        for frame in setup_candidates:
            ball_pos = self._get_ball_position(frame)
            if ball_pos and ball_pos['y'] < highest_y:
                highest_y = ball_pos['y']
                highest_frame = frame
        
        return highest_frame or frames[-1]
    
    def _extract_trajectory_frames(self, frames: List[Dict], dip_frame: Dict, setup_frame: Dict) -> List[Dict]:
        """Extract frames between dip and setup points"""
        dip_idx = frames.index(dip_frame)
        setup_idx = frames.index(setup_frame)
        
        if dip_idx > setup_idx:
            dip_idx, setup_idx = setup_idx, dip_idx
        
        return frames[dip_idx:setup_idx + 1]
    
    def _interpolate_trajectory(self, trajectory_frames: List[Dict], target_frames: int) -> List[Dict]:
        """Interpolate trajectory to target number of frames"""
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
        """Interpolate position list to target number of frames"""
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
        """Normalize trajectory coordinates relative to dip position"""
        dip_ball = self._get_ball_position(dip_frame)
        dip_wrist = self._get_wrist_position(dip_frame)
        dip_elbow = self._get_elbow_position(dip_frame)
        
        if not dip_ball:
            return {'ball': [], 'wrist': [], 'elbow': []}
        
        normalized_ball = []
        normalized_wrist = []
        normalized_elbow = []
        
        for frame in trajectory_frames:
            # Ball normalization
            if frame.get('ball') and dip_ball:
                normalized_ball.append({
                    'x': frame['ball']['x'] - dip_ball['x'],
                    'y': frame['ball']['y'] - dip_ball['y']
                })
            
            # Wrist normalization
            if frame.get('wrist') and dip_wrist:
                normalized_wrist.append({
                    'x': frame['wrist']['x'] - dip_wrist['x'],
                    'y': frame['wrist']['y'] - dip_wrist['y']
                })
            
            # Elbow normalization
            if frame.get('elbow') and dip_elbow:
                normalized_elbow.append({
                    'x': frame['elbow']['x'] - dip_elbow['x'],
                    'y': frame['elbow']['y'] - dip_elbow['y']
                })
        
        return {
            'ball': normalized_ball,
            'wrist': normalized_wrist,
            'elbow': normalized_elbow
        }
    
    def _calculate_trajectory_velocity(self, trajectory: List[Dict]) -> List[Dict]:
        """Calculate velocity for trajectory points"""
        if len(trajectory) < 2:
            return []
        
        velocities = []
        for i in range(len(trajectory) - 1):
            if i == 0:
                # First frame has zero velocity
                velocities.append({'x': 0.0, 'y': 0.0})
            else:
                dx = trajectory[i]['x'] - trajectory[i-1]['x']
                dy = trajectory[i]['y'] - trajectory[i-1]['y']
                velocities.append({'x': dx, 'y': dy})
        
        # Add last frame velocity
        if len(trajectory) >= 2:
            dx = trajectory[-1]['x'] - trajectory[-2]['x']
            dy = trajectory[-1]['y'] - trajectory[-2]['y']
            velocities.append({'x': dx, 'y': dy})
        
        return velocities
    
    def _calculate_trajectory_acceleration(self, trajectory: List[Dict]) -> List[Dict]:
        """Calculate acceleration for trajectory points"""
        velocities = self._calculate_trajectory_velocity(trajectory)
        return self._calculate_trajectory_velocity(velocities)
    
    def _calculate_trajectory_curvature_series(self, trajectory: List[Dict]) -> List[float]:
        """Calculate curvature at each point in trajectory"""
        if len(trajectory) < 3:
            return [0.0] * len(trajectory)
        
        curvatures = []
        for i in range(len(trajectory)):
            if i == 0 or i == len(trajectory) - 1:
                curvatures.append(0.0)
            else:
                # Calculate curvature at point i
                prev_point = trajectory[i - 1]
                curr_point = trajectory[i]
                next_point = trajectory[i + 1]
                
                curvature = self._calculate_point_curvature(prev_point, curr_point, next_point)
                curvatures.append(curvature)
        
        return curvatures
    
    def _calculate_point_curvature(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate curvature at middle point"""
        try:
            # Vector calculations
            v1 = (p2['x'] - p1['x'], p2['y'] - p1['y'])
            v2 = (p3['x'] - p2['x'], p3['y'] - p2['y'])
            
            # Cross product for curvature
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            
            # Magnitudes
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 * mag2 == 0:
                return 0.0
            
            # Curvature
            curvature = abs(cross_product) / (mag1 * mag2)
            return curvature
            
        except:
            return 0.0
    
    def _analyze_jump_timing_pattern(self, hip_heights: List[float], rising_frames: List[Dict]) -> List[float]:
        """Analyze jump timing pattern"""
        if not hip_heights:
            return []
        
        # Find peak jump (minimum y value)
        peak_jump_idx = np.argmin(hip_heights) if len(hip_heights) > 0 else 0
        total_frames = len(hip_heights)
        
        # Create timing pattern
        timing_pattern = []
        for i in range(total_frames):
            # Distance from peak jump
            distance_from_peak = abs(i - peak_jump_idx) / total_frames if total_frames > 0 else 0
            
            # Jump phase indicator
            jump_phase = 0.0
            if i < peak_jump_idx:
                jump_phase = i / peak_jump_idx if peak_jump_idx > 0 else 0  # Rising
            else:
                jump_phase = 1.0  # At or after peak
            
            timing_pattern.append(jump_phase)
        
        return timing_pattern
    
    def _get_ball_position(self, frame: Dict) -> Optional[Dict]:
        """Get ball position from frame"""
        ball = frame.get('normalized_ball', {})
        if 'center_x' in ball and 'center_y' in ball:
            return {'x': ball['center_x'], 'y': ball['center_y']}
        return None
    
    def _get_wrist_position(self, frame: Dict) -> Optional[Dict]:
        """Get wrist position from frame"""
        pose = frame.get('normalized_pose', {})
        wrist = pose.get(f'{self.selected_hand}_wrist', {})
        if 'x' in wrist and 'y' in wrist:
            return {'x': wrist['x'], 'y': wrist['y']}
        return None
    
    def _get_elbow_position(self, frame: Dict) -> Optional[Dict]:
        """Get elbow position from frame"""
        pose = frame.get('normalized_pose', {})
        elbow = pose.get(f'{self.selected_hand}_elbow', {})
        if 'x' in elbow and 'y' in elbow:
            return {'x': elbow['x'], 'y': elbow['y']}
        return None
    
    def _calculate_body_tilt(self, pose: Dict) -> Optional[float]:
        """Calculate body tilt relative to vertical"""
        try:
            shoulder_center = self._safe_shoulder_center(pose)
            hip_center = self._safe_hip_center(pose)
            
            if shoulder_center and hip_center:
                dx = shoulder_center['x'] - hip_center['x']
                dy = shoulder_center['y'] - hip_center['y']
                tilt_angle = np.degrees(np.arctan2(dx, abs(dy)))
                return tilt_angle
            else:
                return None
        except:
            return None
    
    def _calculate_leg_angle(self, pose: Dict, side: str) -> Optional[float]:
        """Calculate hip-knee-ankle angle for specified side"""
        try:
            hip = pose.get(f'{side}_hip', {})
            knee = pose.get(f'{side}_knee', {})
            ankle = pose.get(f'{side}_ankle', {})
            
            if self._has_valid_coordinates(hip, knee, ankle):
                return self._calculate_joint_angle(hip, knee, ankle)
            else:
                return None
        except:
            return None
    
    def _calculate_joint_angle(self, point1: Dict, point2: Dict, point3: Dict) -> float:
        """Calculate angle at point2 formed by point1-point2-point3"""
        try:
            x1, y1 = float(point1['x']), float(point1['y'])
            x2, y2 = float(point2['x']), float(point2['y'])
            x3, y3 = float(point3['x']), float(point3['y'])
            
            # Vectors
            v1 = np.array([x1 - x2, y1 - y2])
            v2 = np.array([x3 - x2, y3 - y2])
            
            # Angle calculation
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            return angle
        except:
            return np.nan
    
    def _calculate_velocity(self, values: List[float]) -> List[float]:
        """Calculate velocity (first derivative) of time series"""
        if len(values) < 2:
            return [0.0] * len(values)
        
        velocities = [0.0]  # First frame has zero velocity
        
        for i in range(1, len(values)):
            velocity = values[i] - values[i-1]
            velocities.append(velocity)
        
        return velocities
    
    def _calculate_acceleration(self, values: List[float]) -> List[float]:
        """Calculate acceleration (second derivative) of time series"""
        velocities = self._calculate_velocity(values)
        return self._calculate_velocity(velocities)
    
    def _interpolate_series_1d(self, values: List[float]) -> List[float]:
        """Interpolate missing values in 1D time series"""
        if not values:
            return values
        
        values = np.array(values, dtype=float)
        mask = ~np.isnan(values)
        
        if not np.any(mask).item():
            return [0.0] * len(values)
        
        # Simple linear interpolation for missing values
        if np.all(mask).item():
            return values.tolist()
        
        indices = np.arange(len(values))
        values[~mask] = np.interp(indices[~mask], indices[mask], values[mask])
        
        return values.tolist()
    
    def _has_valid_coordinates(self, *points) -> bool:
        """Check if all points have valid x, y coordinates"""
        for point in points:
            if not point or 'x' not in point or 'y' not in point:
                return False
            if point['x'] is None or point['y'] is None:
                return False
        return True
    
    def _create_empty_windup_features(self) -> Dict:
        """Create empty windup features structure for error cases"""
        empty_trajectory = [{'x': 0.0, 'y': 0.0} for _ in range(self.interpolation_frames)]
        empty_series = [0.0] * self.interpolation_frames
        
        return {
            'ball_trajectory': empty_trajectory,
            'wrist_trajectory': empty_trajectory,
            'elbow_trajectory': empty_trajectory,
            'ball_velocities': empty_trajectory,
            'wrist_velocities': empty_trajectory,
            'elbow_velocities': empty_trajectory,
            'ball_accelerations': empty_trajectory,
            'wrist_accelerations': empty_trajectory,
            'elbow_accelerations': empty_trajectory,
            'ball_curvatures': empty_series,
            'wrist_curvatures': empty_series,
            'elbow_curvatures': empty_series,
            'dip_frame_idx': 0,
            'setup_frame_idx': 0,
            'trajectory_length': 0,
            'feature_type': 'rising_windup_kinematics'
        }