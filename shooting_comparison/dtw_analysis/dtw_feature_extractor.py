"""
DTW Feature Extractor

Extracts DTW-compatible features from normalized shooting data.
Works with pre-normalized data (torso-scaled, hip-centered, direction-normalized).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from .dtw_config import DTW_FEATURE_WEIGHTS, PHASE_IMPORTANCE_WEIGHTS

class DTWFeatureExtractor:
    """
    Extracts DTW-compatible features from normalized shooting data.
    
    Handles pre-normalized data:
    - Torso-based scale normalization
    - Hip position coordinate normalization
    - Left/right direction normalization
    - Screen aspect ratio normalization
    """
    
    def __init__(self):
        self.feature_weights = DTW_FEATURE_WEIGHTS.copy()
        self.phase_weights = PHASE_IMPORTANCE_WEIGHTS.copy()
        
        # Minimum data requirements
        self.min_trajectory_length = 3
        self.min_phase_frames = 2
        
    def extract_dtw_features(self, normalized_video_data: Dict, selected_hand: str) -> Dict:
        """
        Extract DTW features from normalized video data.
        
        Args:
            normalized_video_data: Video data with normalized coordinates
            selected_hand: 'left' or 'right' shooting hand
            
        Returns:
            Dictionary containing time series features for DTW analysis
        """
        frames = normalized_video_data.get('frames', [])
        if not frames:
            return {'error': 'No frames available for feature extraction'}
        
        print(f"🔄 Extracting DTW features from {len(frames)} frames...")
        
        # Organize frames by phase
        phase_frames = self._organize_frames_by_phase(frames)
        
        # Extract primary DTW features
        dtw_features = {}
        
        try:
            dtw_features['ball_wrist_trajectory'] = self._extract_ball_wrist_trajectory(frames, selected_hand)
            dtw_features['shooting_arm_kinematics'] = self._extract_arm_kinematics(frames, selected_hand)
            dtw_features['lower_body_stability'] = self._extract_lower_body_features(frames)
            dtw_features['phase_timing_patterns'] = self._extract_phase_timing(phase_frames)
            dtw_features['body_alignment'] = self._extract_body_alignment(frames)
            
            # Add metadata
            dtw_features['metadata'] = {
                'total_frames': len(frames),
                'selected_hand': selected_hand,
                'fps': normalized_video_data.get('metadata', {}).get('fps', 30.0),
                'phase_distribution': {phase: len(phase_frames.get(phase, [])) for phase in ['Setup', 'Loading', 'Rising', 'Release', 'Follow-through']},
                'extraction_success': True
            }
            
            print(f"✅ DTW features extracted successfully")
            
        except Exception as e:
            print(f"❌ Error extracting DTW features: {e}")
            return {'error': f'Feature extraction failed: {str(e)}'}
        
        return dtw_features
    
    def _organize_frames_by_phase(self, frames: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize frames by shooting phase"""
        phase_frames = {
            'Setup': [],
            'Loading': [], 
            'Rising': [],
            'Release': [],
            'Follow-through': [],
            'General': []
        }
        
        for frame in frames:
            phase = frame.get('phase', 'General')
            if phase in phase_frames:
                phase_frames[phase].append(frame)
            else:
                phase_frames['General'].append(frame)
        
        return phase_frames
    
    def _extract_ball_wrist_trajectory(self, frames: List[Dict], selected_hand: str) -> Dict:
        """Extract ball-wrist relationship trajectory for DTW"""
        ball_positions = []
        wrist_positions = []
        ball_wrist_distances = []
        
        wrist_key = f'{selected_hand}_wrist'
        
        for frame in frames:
            pose = frame.get('normalized_pose', {})
            ball = frame.get('normalized_ball', {})  # Using normalized_ball instead of ball_info
            
            # Debugging: Check ball_info structure
            if len(ball_positions) == 0:  # Only print for the first frame
                print(f"         🔍 Debug: frame keys = {list(frame.keys())}")
                print(f"         🔍 Debug: normalized_ball = {ball}")
                print(f"         🔍 Debug: normalized_ball type = {type(ball)}")
                if ball:
                    print(f"         🔍 Debug: normalized_ball keys = {list(ball.keys())}")
            
            # Ball position (already normalized)
            if ball and 'center_x' in ball and 'center_y' in ball:
                ball_pos = [float(ball['center_x']), float(ball['center_y'])]
                ball_positions.append(ball_pos)
            else:
                ball_positions.append([np.nan, np.nan])
            
            # Wrist position (already normalized)
            wrist = pose.get(wrist_key, {})
            if wrist and 'x' in wrist and 'y' in wrist:
                wrist_pos = [float(wrist['x']), float(wrist['y'])]
                wrist_positions.append(wrist_pos)
            else:
                wrist_positions.append([np.nan, np.nan])
            
            # Ball-wrist distance
            if not np.isnan(ball_positions[-1][0]) and not np.isnan(wrist_positions[-1][0]):
                distance = np.sqrt(
                    (ball_positions[-1][0] - wrist_positions[-1][0])**2 +
                    (ball_positions[-1][1] - wrist_positions[-1][1])**2
                )
                ball_wrist_distances.append(float(distance))
            else:
                ball_wrist_distances.append(np.nan)
        
        # Interpolate missing values
        ball_positions = self._interpolate_trajectory_2d(ball_positions)
        wrist_positions = self._interpolate_trajectory_2d(wrist_positions)
        ball_wrist_distances = self._interpolate_series_1d(ball_wrist_distances)
        
        return {
            'ball_trajectory': ball_positions,
            'wrist_trajectory': wrist_positions,
            'ball_wrist_distance': ball_wrist_distances,
            'feature_type': 'ball_wrist_special'  # Changed to a special type for more lenient processing
        }
    
    def _extract_arm_kinematics(self, frames: List[Dict], selected_hand: str) -> Dict:
        """Extract shooting arm kinematics for DTW"""
        elbow_angles = []
        shoulder_positions = []
        elbow_positions = []
        wrist_positions = []
        
        # Key names based on selected hand
        shoulder_key = f'{selected_hand}_shoulder'
        elbow_key = f'{selected_hand}_elbow'
        wrist_key = f'{selected_hand}_wrist'
        
        for frame in frames:
            pose = frame.get('normalized_pose', {})
            
            # Get joint positions (already normalized)
            shoulder = pose.get(shoulder_key, {})
            elbow = pose.get(elbow_key, {})
            wrist = pose.get(wrist_key, {})
            
            # Calculate elbow angle
            if all(joint.get('x') is not None and joint.get('y') is not None 
                   for joint in [shoulder, elbow, wrist]):
                angle = self._calculate_joint_angle(shoulder, elbow, wrist)
                elbow_angles.append(float(angle))
            else:
                elbow_angles.append(np.nan)
            
            # Store positions
            shoulder_positions.append([
                float(shoulder.get('x', np.nan)), 
                float(shoulder.get('y', np.nan))
            ])
            elbow_positions.append([
                float(elbow.get('x', np.nan)), 
                float(elbow.get('y', np.nan))
            ])
            wrist_positions.append([
                float(wrist.get('x', np.nan)), 
                float(wrist.get('y', np.nan))
            ])
        
        # Interpolate missing values
        elbow_angles = self._interpolate_series_1d(elbow_angles)
        shoulder_positions = self._interpolate_trajectory_2d(shoulder_positions)
        elbow_positions = self._interpolate_trajectory_2d(elbow_positions)
        wrist_positions = self._interpolate_trajectory_2d(wrist_positions)
        
        return {
            'elbow_angles': elbow_angles,
            'shoulder_trajectory': shoulder_positions,
            'elbow_trajectory': elbow_positions,
            'wrist_trajectory': wrist_positions,
            'feature_type': 'kinematics'
        }
    
    def _extract_lower_body_features(self, frames: List[Dict]) -> Dict:
        """Extract lower body stability features"""
        hip_positions = []
        left_knee_angles = []
        right_knee_angles = []
        stance_widths = []
        
        for frame in frames:
            pose = frame.get('normalized_pose', {})
            
            # Hip center position
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            
            if (left_hip.get('x') is not None and right_hip.get('x') is not None):
                hip_center_x = (float(left_hip['x']) + float(right_hip['x'])) / 2
                hip_center_y = (float(left_hip['y']) + float(right_hip['y'])) / 2
                hip_positions.append([hip_center_x, hip_center_y])
            else:
                hip_positions.append([np.nan, np.nan])
            
            # Knee angles
            left_knee_angle = self._calculate_knee_angle(pose, 'left')
            right_knee_angle = self._calculate_knee_angle(pose, 'right')
            
            left_knee_angles.append(float(left_knee_angle) if not np.isnan(left_knee_angle) else np.nan)
            right_knee_angles.append(float(right_knee_angle) if not np.isnan(right_knee_angle) else np.nan)
            
            # Stance width
            if (left_hip.get('x') is not None and right_hip.get('x') is not None):
                width = abs(float(right_hip['x']) - float(left_hip['x']))
                stance_widths.append(width)
            else:
                stance_widths.append(np.nan)
        
        # Interpolate missing values
        hip_positions = self._interpolate_trajectory_2d(hip_positions)
        left_knee_angles = self._interpolate_series_1d(left_knee_angles)
        right_knee_angles = self._interpolate_series_1d(right_knee_angles)
        stance_widths = self._interpolate_series_1d(stance_widths)
        
        return {
            'hip_trajectory': hip_positions,
            'left_knee_angles': left_knee_angles,
            'right_knee_angles': right_knee_angles,
            'stance_stability': stance_widths,
            'feature_type': 'stability'
        }
    
    def _extract_phase_timing(self, phase_frames: Dict[str, List[Dict]]) -> Dict:
        """Extract phase timing patterns"""
        phase_durations = []
        transition_timings = []
        
        total_frames = sum(len(frames) for frames in phase_frames.values())
        
        # Calculate normalized phase durations
        phase_order = ['Setup', 'Loading', 'Rising', 'Release', 'Follow-through']
        
        for phase in phase_order:
            frames = phase_frames.get(phase, [])
            if total_frames > 0:
                duration_ratio = len(frames) / total_frames
                phase_durations.append(float(duration_ratio))
            else:
                phase_durations.append(0.0)
        
        # Calculate transition timing consistency
        # This is a simplified version - could be more sophisticated
        for i in range(len(phase_order) - 1):
            current_phase = phase_order[i]
            next_phase = phase_order[i + 1]
            
            current_frames = len(phase_frames.get(current_phase, []))
            next_frames = len(phase_frames.get(next_phase, []))
            
            if current_frames + next_frames > 0:
                transition_ratio = current_frames / (current_frames + next_frames)
                transition_timings.append(float(transition_ratio))
            else:
                transition_timings.append(0.5)  # Default equal split
        
        return {
            'phase_durations': phase_durations,
            'transition_timing': transition_timings,
            'feature_type': 'timing'
        }
    
    def _extract_body_alignment(self, frames: List[Dict]) -> Dict:
        """Extract body alignment features"""
        shoulder_tilts = []
        torso_angles = []
        head_positions = []
        
        for frame in frames:
            pose = frame.get('normalized_pose', {})
            
            # Shoulder tilt
            left_shoulder = pose.get('left_shoulder', {})
            right_shoulder = pose.get('right_shoulder', {})
            
            if (left_shoulder.get('y') is not None and right_shoulder.get('y') is not None):
                tilt = float(right_shoulder['y']) - float(left_shoulder['y'])
                shoulder_tilts.append(tilt)
            else:
                shoulder_tilts.append(np.nan)
            
            # Torso angle (simplified)
            left_hip = pose.get('left_hip', {})
            if (left_shoulder.get('x') is not None and left_hip.get('x') is not None):
                dx = float(left_shoulder['x']) - float(left_hip['x'])
                dy = float(left_shoulder['y']) - float(left_hip['y'])
                if dy != 0:
                    angle = np.degrees(np.arctan(dx / dy))
                    torso_angles.append(float(angle))
                else:
                    torso_angles.append(np.nan)
            else:
                torso_angles.append(np.nan)
            
            # Head position (if available)
            nose = pose.get('nose', {})
            if nose.get('x') is not None and nose.get('y') is not None:
                head_positions.append([float(nose['x']), float(nose['y'])])
            else:
                head_positions.append([np.nan, np.nan])
        
        # Interpolate missing values
        shoulder_tilts = self._interpolate_series_1d(shoulder_tilts)
        torso_angles = self._interpolate_series_1d(torso_angles)
        head_positions = self._interpolate_trajectory_2d(head_positions)
        
        return {
            'shoulder_tilt': shoulder_tilts,
            'torso_angle': torso_angles,
            'head_stability': head_positions,
            'feature_type': 'stability'
        }
    
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
    
    def _calculate_knee_angle(self, pose: Dict, side: str) -> float:
        """Calculate knee angle for given side"""
        try:
            hip = pose.get(f'{side}_hip', {})
            knee = pose.get(f'{side}_knee', {})
            ankle = pose.get(f'{side}_ankle', {})
            
            if all(joint.get('x') is not None and joint.get('y') is not None 
                   for joint in [hip, knee, ankle]):
                return self._calculate_joint_angle(hip, knee, ankle)
            else:
                return np.nan
        except:
            return np.nan
    
    def _interpolate_trajectory_2d(self, trajectory: List[List[float]]) -> List[List[float]]:
        """Interpolate 2D trajectory to handle missing data"""
        if not trajectory:
            return []
        
        # Convert to numpy arrays for easier handling
        trajectory = np.array(trajectory)
        
        # Handle NaN values more generously
        valid_mask = ~np.isnan(trajectory).any(axis=1)
        
        if np.sum(valid_mask).item() < 2:
            # If too few valid points, return original with NaN handling
            return trajectory.tolist()
        
        # Get valid indices
        valid_indices = np.where(valid_mask)[0]
        
        # If we have enough valid points, interpolate
        if len(valid_indices) >= 3:
            # Use more generous interpolation
            for i in range(trajectory.shape[1]):  # For each dimension (x, y)
                valid_values = trajectory[valid_indices, i]
                trajectory[:, i] = np.interp(
                    np.arange(len(trajectory)), 
                    valid_indices, 
                    valid_values,
                    left=valid_values[0] if len(valid_values) > 0 else 0,
                    right=valid_values[-1] if len(valid_values) > 0 else 0
                )
        else:
            # For very sparse data, use forward fill then backward fill
            for i in range(trajectory.shape[1]):
                series = trajectory[:, i]
                # Forward fill
                series = pd.Series(series).fillna(method='ffill')
                # Backward fill for remaining NaNs
                series = series.fillna(method='bfill')
                # Fill any remaining NaNs with 0
                series = series.fillna(0)
                trajectory[:, i] = series.values
        
        return trajectory.tolist()
    
    def _interpolate_series_1d(self, series: List[float]) -> List[float]:
        """Interpolate 1D series to handle missing data"""
        if not series:
            return []
        
        # Convert to numpy array
        series = np.array(series)
        
        # Handle NaN values more generously
        valid_mask = ~np.isnan(series)
        
        if np.sum(valid_mask).item() < 2:
            # If too few valid points, return original with NaN handling
            return series.tolist()
        
        # Get valid indices
        valid_indices = np.where(valid_mask)[0]
        
        # If we have enough valid points, interpolate
        if len(valid_indices) >= 3:
            # Use more generous interpolation
            interpolated = np.interp(
                np.arange(len(series)), 
                valid_indices, 
                series[valid_indices],
                left=series[valid_indices[0]] if len(valid_indices) > 0 else 0,
                right=series[valid_indices[-1]] if len(valid_indices) > 0 else 0
            )
        else:
            # For very sparse data, use forward fill then backward fill
            series_pd = pd.Series(series)
            # Forward fill
            series_pd = series_pd.fillna(method='ffill')
            # Backward fill for remaining NaNs
            series_pd = series_pd.fillna(method='bfill')
            # Fill any remaining NaNs with 0
            interpolated = series_pd.fillna(0).values
        
        return interpolated.tolist()