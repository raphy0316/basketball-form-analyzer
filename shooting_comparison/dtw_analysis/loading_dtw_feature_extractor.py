"""
Loading-specific DTW Feature Extractor

Extracts Loading phase specific DTW features for more accurate comparison.
Focuses on leg kinematics, upper body dynamics, and timing patterns during Loading + Loading-Rising phases.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path to import SafeCoordinateMixin
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shooting_comparison.safe_coordinate_mixin import SafeCoordinateMixin

class LoadingDTWFeatureExtractor(SafeCoordinateMixin):
    """
    Extracts Loading-specific DTW features for more accurate similarity comparison.
    
    Focuses on:
    A. loading_leg_kinematics (40%): Leg angle change patterns, rate of change, acceleration, asymmetry
    B. loading_upper_body_dynamics (35%): Shoulder tilt, upper body tilt, distance change, rotation
    C. loading_timing_patterns (25%): Timing patterns and transition point analysis
    """
    
    def __init__(self):
        # Feature weights
        self.feature_weights = {
            'loading_leg_kinematics': 0.40,
            'loading_upper_body_dynamics': 0.35, 
            'loading_timing_patterns': 0.25
        }
        
        # Minimum data requirements
        self.min_trajectory_length = 3
        self.min_phase_frames = 2
    
    def extract_loading_dtw_features(self, video_data: Dict) -> Dict:
        """
        Extract Loading-specific DTW features from video data.
        
        Args:
            video_data: Video data with normalized coordinates
            
        Returns:
            Dictionary containing Loading DTW features for comparison
        """
        frames = video_data.get('frames', [])
        if not frames:
            return {'error': 'No frames available for Loading DTW feature extraction'}
        
        print(f"🔄 Extracting Loading DTW features from {len(frames)} frames...")
        
        # Get Loading + Loading-Rising frames (same as existing loading analysis)
        loading_frames = self._get_loading_frames(frames)
        
        if len(loading_frames) < self.min_phase_frames:
            return {'error': f'Insufficient Loading frames ({len(loading_frames)}) for DTW analysis'}
        
        print(f"   📊 Found {len(loading_frames)} Loading frames")
        
        # Extract Loading-specific DTW features
        loading_dtw_features = {}
        
        try:
            # A. Loading leg kinematics (40%)
            loading_dtw_features['loading_leg_kinematics'] = self._extract_leg_kinematics(loading_frames)
            
            # B. Loading upper body dynamics (35%)
            loading_dtw_features['loading_upper_body_dynamics'] = self._extract_upper_body_dynamics(loading_frames)
            
            # C. Loading timing patterns (25%)
            loading_dtw_features['loading_timing_patterns'] = self._extract_timing_patterns(loading_frames)
            
            # Add metadata
            loading_dtw_features['metadata'] = {
                'total_loading_frames': len(loading_frames),
                'fps': video_data.get('metadata', {}).get('fps', 30.0),
                'extraction_success': True
            }
            
            print(f"✅ Loading DTW features extracted successfully")
            
        except Exception as e:
            print(f"❌ Error extracting Loading DTW features: {e}")
            return {'error': f'Loading DTW feature extraction failed: {str(e)}'}
        
        return loading_dtw_features
    
    def _get_loading_frames(self, frames: List[Dict]) -> List[Dict]:
        """Get Loading + Loading-Rising frames (same logic as existing loading analysis)"""
        loading_frames = []
        
        for frame in frames:
            phase = frame.get('phase', '')
            if phase in ['Loading', 'Loading-Rising']:
                loading_frames.append(frame)
        
        return loading_frames
    
    def _extract_leg_kinematics(self, loading_frames: List[Dict]) -> Dict:
        """
        Extract leg kinematics for Loading phase DTW analysis.
        
        Features:
        - Left and right leg angle change patterns (time series)
        - Leg angle rate of change (velocity)
        - Leg angle acceleration (acceleration)
        - Left-right leg angle ratio (asymmetry)
        """
        left_angles = []
        right_angles = []
        asymmetry_ratios = []
        
        for frame in loading_frames:
            pose = frame.get('normalized_pose', {})
            
            # Calculate leg angles (hip-knee-ankle)
            left_angle = self._calculate_leg_angle(pose, 'left')
            right_angle = self._calculate_leg_angle(pose, 'right')
            
            if left_angle is not None and right_angle is not None:
                left_angles.append(left_angle)
                right_angles.append(right_angle)
                
                # Calculate asymmetry ratio for this frame
                asymmetry = abs(left_angle - right_angle)
                avg_angle = (left_angle + right_angle) / 2
                asymmetry_ratio = asymmetry / avg_angle if avg_angle > 0 else 0
                asymmetry_ratios.append(asymmetry_ratio)
            else:
                left_angles.append(np.nan)
                right_angles.append(np.nan)
                asymmetry_ratios.append(np.nan)
        
        # Interpolate missing values
        left_angles = self._interpolate_series_1d(left_angles)
        right_angles = self._interpolate_series_1d(right_angles)
        asymmetry_ratios = self._interpolate_series_1d(asymmetry_ratios)
        
        # Calculate velocities (rate of change)
        left_velocities = self._calculate_velocity(left_angles)
        right_velocities = self._calculate_velocity(right_angles)
        
        # Calculate accelerations (acceleration)
        left_accelerations = self._calculate_acceleration(left_angles)
        right_accelerations = self._calculate_acceleration(right_angles)
        
        return {
            'left_leg_angles': left_angles,
            'right_leg_angles': right_angles,
            'left_leg_velocities': left_velocities,
            'right_leg_velocities': right_velocities,
            'left_leg_accelerations': left_accelerations,
            'right_leg_accelerations': right_accelerations,
            'asymmetry_ratios': asymmetry_ratios,
            'feature_type': 'loading_kinematics'
        }
    
    def _extract_upper_body_dynamics(self, loading_frames: List[Dict]) -> Dict:
        """
        Extract upper body dynamics for Loading phase DTW analysis.
        
        Features:
        - Shoulder tilt change pattern
        - Upper body tilt rate of change
        - Shoulder-hip distance change
        - Upper body rotation angle
        """
        shoulder_tilts = []
        shoulder_hip_distances = []
        torso_rotations = []
        
        for frame in loading_frames:
            pose = frame.get('normalized_pose', {})
            
            # Calculate shoulder tilt
            shoulder_tilt = self._calculate_shoulder_tilt(pose)
            shoulder_tilts.append(shoulder_tilt if shoulder_tilt is not None else np.nan)
            
            # Calculate shoulder-hip distance
            sh_distance = self._calculate_shoulder_hip_distance(pose)
            shoulder_hip_distances.append(sh_distance if sh_distance is not None else np.nan)
            
            # Calculate torso rotation (upper body rotation angle)
            torso_rotation = self._calculate_torso_rotation(pose)
            torso_rotations.append(torso_rotation if torso_rotation is not None else np.nan)
        
        # Interpolate missing values
        shoulder_tilts = self._interpolate_series_1d(shoulder_tilts)
        shoulder_hip_distances = self._interpolate_series_1d(shoulder_hip_distances)
        torso_rotations = self._interpolate_series_1d(torso_rotations)
        
        # Calculate velocities (rate of change)
        shoulder_tilt_velocities = self._calculate_velocity(shoulder_tilts)
        shoulder_hip_velocities = self._calculate_velocity(shoulder_hip_distances)
        torso_rotation_velocities = self._calculate_velocity(torso_rotations)
        
        return {
            'shoulder_tilts': shoulder_tilts,
            'shoulder_hip_distances': shoulder_hip_distances, 
            'torso_rotations': torso_rotations,
            'shoulder_tilt_velocities': shoulder_tilt_velocities,
            'shoulder_hip_velocities': shoulder_hip_velocities,
            'torso_rotation_velocities': torso_rotation_velocities,
            'feature_type': 'loading_dynamics'
        }
    
    def _extract_timing_patterns(self, loading_frames: List[Dict]) -> Dict:
        """
        Extract timing patterns for Loading phase DTW analysis.
        
        Features:
        - transition_start: Time taken to reach max leg angle during loading phase
        - transition_end: Time taken from max angle to the end of loading
        - transition_duration: Time maintained at max angle
        - Ratio of timing by phase
        """
        # Calculate leg angles for timing analysis
        left_angles = []
        right_angles = []
        
        for frame in loading_frames:
            pose = frame.get('normalized_pose', {})
            left_angle = self._calculate_leg_angle(pose, 'left')
            right_angle = self._calculate_leg_angle(pose, 'right')
            
            left_angles.append(left_angle if left_angle is not None else np.nan)
            right_angles.append(right_angle if right_angle is not None else np.nan)
        
        # Interpolate missing values
        left_angles = self._interpolate_series_1d(left_angles)
        right_angles = self._interpolate_series_1d(right_angles)
        
        # Find max angles and timing
        left_max_idx = np.argmax(left_angles) if len(left_angles) > 0 else 0
        right_max_idx = np.argmax(right_angles) if len(right_angles) > 0 else 0
        combined_max_idx = max(left_max_idx, right_max_idx)  # Both legs reach max
        
        total_frames = len(loading_frames)
        
        # Calculate timing patterns
        transition_start_ratio = combined_max_idx / total_frames if total_frames > 0 else 0
        transition_end_ratio = (total_frames - combined_max_idx) / total_frames if total_frames > 0 else 0
        
        # Calculate duration at max angle (frames where angle is close to max)
        max_threshold = 0.95  # 95% of max angle
        left_max_val = left_angles[left_max_idx] if len(left_angles) > 0 else 0
        right_max_val = right_angles[right_max_idx] if len(right_angles) > 0 else 0
        
        duration_frames = 0
        for i, (left_ang, right_ang) in enumerate(zip(left_angles, right_angles)):
            if (left_ang >= left_max_val * max_threshold or 
                right_ang >= right_max_val * max_threshold):
                duration_frames += 1
        
        duration_ratio = duration_frames / total_frames if total_frames > 0 else 0
        
        # Create timing pattern as time series for DTW
        timing_pattern = []
        for i in range(len(loading_frames)):
            # Normalized time progression with timing markers
            time_progress = i / total_frames if total_frames > 0 else 0
            
            # Add timing features as multi-dimensional signal
            timing_features = [
                time_progress,  # Basic time progression
                1.0 if i == combined_max_idx else 0.0,  # Max angle marker
                transition_start_ratio,  # Transition start ratio (constant)
                transition_end_ratio,   # Transition end ratio (constant)
                duration_ratio          # Duration ratio (constant)
            ]
            timing_pattern.append(timing_features)
        
        return {
            'timing_pattern': timing_pattern,
            'transition_start_ratio': transition_start_ratio,
            'transition_end_ratio': transition_end_ratio,
            'duration_ratio': duration_ratio,
            'max_angle_frame': combined_max_idx,
            'feature_type': 'loading_timing'
        }
    
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
    
    def _calculate_shoulder_tilt(self, pose: Dict) -> Optional[float]:
        """Calculate shoulder tilt relative to horizontal"""
        try:
            left_shoulder = pose.get('left_shoulder', {})
            right_shoulder = pose.get('right_shoulder', {})
            
            if self._has_valid_coordinates(left_shoulder, right_shoulder):
                dx = right_shoulder['x'] - left_shoulder['x']
                dy = right_shoulder['y'] - left_shoulder['y']
                angle = np.degrees(np.arctan2(dy, dx))
                return angle
            else:
                return None
        except:
            return None
    
    def _calculate_shoulder_hip_distance(self, pose: Dict) -> Optional[float]:
        """Calculate distance between shoulder center and hip center"""
        try:
            shoulder_center = self._safe_shoulder_center(pose)
            hip_center = self._safe_hip_center(pose)
            
            if shoulder_center and hip_center:
                distance = np.sqrt(
                    (shoulder_center['x'] - hip_center['x'])**2 + 
                    (shoulder_center['y'] - hip_center['y'])**2
                )
                return distance
            else:
                return None
        except:
            return None
    
    def _calculate_torso_rotation(self, pose: Dict) -> Optional[float]:
        """Calculate torso rotation (rotation angle between shoulder line and hip line)"""
        try:
            left_shoulder = pose.get('left_shoulder', {})
            right_shoulder = pose.get('right_shoulder', {})
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            
            if self._has_valid_coordinates(left_shoulder, right_shoulder, left_hip, right_hip):
                # Shoulder line angle
                shoulder_dx = right_shoulder['x'] - left_shoulder['x']
                shoulder_dy = right_shoulder['y'] - left_shoulder['y']
                shoulder_angle = np.degrees(np.arctan2(shoulder_dy, shoulder_dx))
                
                # Hip line angle
                hip_dx = right_hip['x'] - left_hip['x']
                hip_dy = right_hip['y'] - left_hip['y']
                hip_angle = np.degrees(np.arctan2(hip_dy, hip_dx))
                
                # Rotation difference
                rotation = shoulder_angle - hip_angle
                
                # Normalize to [-180, 180]
                while rotation > 180:
                    rotation -= 360
                while rotation < -180:
                    rotation += 360
                
                return rotation
            else:
                return None
        except:
            return None
    
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
    
    def _has_valid_coordinates(self, *points) -> bool:
        """Check if all points have valid x, y coordinates"""
        for point in points:
            if not point or 'x' not in point or 'y' not in point:
                return False
            if point['x'] is None or point['y'] is None:
                return False
        return True
    
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