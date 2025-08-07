"""
Release Phase Analyzer

This module analyzes the release phase of basketball shooting form.
It extracts key information from Release phase including arm angles, ball position, and body measurements.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json


class ReleaseAnalyzer:
    """
    Analyzer for release phase information.
    
    Extracts key measurements from Release phase:
    - Arm angles relative to torso and hip vertical line
    - Ball position relative to eyes at release point
    - Body tilt and leg angles at release
    - Release timing and positioning
    """
    
    def __init__(self):
        self.release_data = {}
        self.selected_hand = 'right'  # Default to right hand; can be set to 'left' as needed

    def analyze_release_phase(self, video_data: Dict, selected_hand) -> Dict:
        """
        Analyze release phase information from video data.
        
        Args:
            video_data: Video analysis data containing frames
            
        Returns:
            Dictionary containing release phase analysis results
        """
        self.selected_hand = selected_hand
        frames = video_data.get('frames', [])
        if not frames:
            return {"error": "No frames available for analysis"}
        
        # Get FPS from metadata (default to 30fps)
        fps = video_data.get('metadata', {}).get('fps', 30.0)
        
        # Find all Release frames
        release_frames = []
        
        for frame in frames:
            phase = frame.get('phase', '')
            if phase == 'Release':
                release_frames.append(frame)
        
        if not release_frames:
            return {"error": "No Release frames found"}
        
        # Analyze release phase
        release_analysis = {
            'fps': fps,
            'total_release_frames': len(release_frames),
            'total_release_time': len(release_frames) / fps,
            'arm_angles': self._analyze_arm_angles(release_frames),
            'ball_position': self._analyze_ball_position(release_frames),
            'body_analysis': self._analyze_body_angles(release_frames),
            'release_timing': self._analyze_release_timing(release_frames, fps, frames)
        }
        
        return release_analysis
    
    def _analyze_arm_angles(self, release_frames: List[Dict]) -> Dict:
        """
        Analyze arm angles relative to torso and hip vertical line.
        
        Args:
            release_frames: List of release frame data
            
        Returns:
            Dictionary containing arm angle measurements
        """
        arm_angles = []
        
        for frame in release_frames:
            pose = frame.get('normalized_pose', {})
            
            # Get shoulder, elbow, and wrist positions
            left_shoulder = pose.get('left_shoulder', {})
            right_shoulder = pose.get('right_shoulder', {})
            left_elbow = pose.get('left_elbow', {})
            right_elbow = pose.get('right_elbow', {})
            left_wrist = pose.get('left_wrist', {})
            right_wrist = pose.get('right_wrist', {})
            
            # Get hip positions for vertical reference
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            
            if (self._has_valid_coordinates(left_shoulder, left_elbow, left_wrist) and
                self._has_valid_coordinates(right_shoulder, right_elbow, right_wrist) and
                self._has_valid_coordinates(left_hip, right_hip)):
                
                # Calculate hip center for vertical reference
                hip_center_x = (left_hip.get('x', 0) + right_hip.get('x', 0)) / 2
                hip_center_y = (left_hip.get('y', 0) + right_hip.get('y', 0)) / 2
                
                # Calculate shoulder center for torso reference
                shoulder_center_x = (left_shoulder.get('x', 0) + right_shoulder.get('x', 0)) / 2
                shoulder_center_y = (left_shoulder.get('y', 0) + right_shoulder.get('y', 0)) / 2
                
                # Analyze left arm
                left_arm_angles = self._calculate_arm_angles(
                    left_shoulder, left_elbow, left_wrist,
                    shoulder_center_x, shoulder_center_y,
                    hip_center_x, hip_center_y
                )
                
                # Analyze right arm
                right_arm_angles = self._calculate_arm_angles(
                    right_shoulder, right_elbow, right_wrist,
                    shoulder_center_x, shoulder_center_y,
                    hip_center_x, hip_center_y
                )
                
                frame_angles = {
                    'left_arm': left_arm_angles,
                    'right_arm': right_arm_angles
                }
                arm_angles.append(frame_angles)
        
        # Calculate averages and ranges
        if arm_angles:
            left_torso_angles = [angle['left_arm']['torso_angle'] for angle in arm_angles if angle['left_arm']['torso_angle'] is not None]
            left_vertical_angles = [angle['left_arm']['vertical_angle'] for angle in arm_angles if angle['left_arm']['vertical_angle'] is not None]
            right_torso_angles = [angle['right_arm']['torso_angle'] for angle in arm_angles if angle['right_arm']['torso_angle'] is not None]
            right_vertical_angles = [angle['right_arm']['vertical_angle'] for angle in arm_angles if angle['right_arm']['vertical_angle'] is not None]
            
            return {
                'left_arm': {
                    'torso_angle': {
                        'average': np.mean(left_torso_angles) if left_torso_angles else None,
                        'min': min(left_torso_angles) if left_torso_angles else None,
                        'max': max(left_torso_angles) if left_torso_angles else None
                    },
                    'vertical_angle': {
                        'average': np.mean(left_vertical_angles) if left_vertical_angles else None,
                        'min': min(left_vertical_angles) if left_vertical_angles else None,
                        'max': max(left_vertical_angles) if left_vertical_angles else None
                    }
                },
                'right_arm': {
                    'torso_angle': {
                        'average': np.mean(right_torso_angles) if right_torso_angles else None,
                        'min': min(right_torso_angles) if right_torso_angles else None,
                        'max': max(right_torso_angles) if right_torso_angles else None
                    },
                    'vertical_angle': {
                        'average': np.mean(right_vertical_angles) if right_vertical_angles else None,
                        'min': min(right_vertical_angles) if right_vertical_angles else None,
                        'max': max(right_vertical_angles) if right_vertical_angles else None
                    }
                }
            }
        
        return {"error": "No valid arm angle data found"}
    
    def _calculate_arm_angles(self, shoulder: Dict, elbow: Dict, wrist: Dict,
                             shoulder_center_x: float, shoulder_center_y: float,
                             hip_center_x: float, hip_center_y: float) -> Dict:
        """Calculate arm angles relative to torso and hip vertical line."""
        angles = {}
        
        # Calculate upper arm angle (shoulder to elbow)
        if self._has_valid_coordinates(shoulder, elbow):
            # Torso angle (relative to shoulder center)
            dx_upper = elbow.get('x', 0) - shoulder.get('x', 0)
            dy_upper = elbow.get('y', 0) - shoulder.get('y', 0)
            upper_arm_angle = np.degrees(np.arctan2(dx_upper, -dy_upper))
            
            # Vertical angle (relative to hip vertical line)
            dx_upper_vert = elbow.get('x', 0) - hip_center_x
            dy_upper_vert = elbow.get('y', 0) - hip_center_y
            upper_vertical_angle = np.degrees(np.arctan2(dx_upper_vert, -dy_upper_vert))
            
            angles['upper_arm_torso_angle'] = upper_arm_angle
            angles['upper_arm_vertical_angle'] = upper_vertical_angle
        
        # Calculate forearm angle (elbow to wrist)
        if self._has_valid_coordinates(elbow, wrist):
            # Torso angle (relative to shoulder center)
            dx_forearm = wrist.get('x', 0) - elbow.get('x', 0)
            dy_forearm = wrist.get('y', 0) - elbow.get('y', 0)
            forearm_angle = np.degrees(np.arctan2(dx_forearm, -dy_forearm))
            
            # Vertical angle (relative to hip vertical line)
            dx_forearm_vert = wrist.get('x', 0) - hip_center_x
            dy_forearm_vert = wrist.get('y', 0) - hip_center_y
            forearm_vertical_angle = np.degrees(np.arctan2(dx_forearm_vert, -dy_forearm_vert))
            
            angles['forearm_torso_angle'] = forearm_angle
            angles['forearm_vertical_angle'] = forearm_vertical_angle
        
        # Calculate overall arm angle (shoulder to wrist)
        if self._has_valid_coordinates(shoulder, wrist):
            # Torso angle (relative to shoulder center)
            dx_arm = wrist.get('x', 0) - shoulder.get('x', 0)
            dy_arm = wrist.get('y', 0) - shoulder.get('y', 0)
            arm_angle = np.degrees(np.arctan2(dx_arm, -dy_arm))
            
            # Vertical angle (relative to hip vertical line)
            dx_arm_vert = wrist.get('x', 0) - hip_center_x
            dy_arm_vert = wrist.get('y', 0) - hip_center_y
            arm_vertical_angle = np.degrees(np.arctan2(dx_arm_vert, -dy_arm_vert))
            
            angles['torso_angle'] = arm_angle
            angles['vertical_angle'] = arm_vertical_angle
        
        return angles
    
    def _analyze_ball_position(self, release_frames: List[Dict]) -> Dict:
        """
        Analyze ball position relative to eyes and ball vector at release point.
        
        Args:
            release_frames: List of release frame data
            
        Returns:
            Dictionary containing ball position and vector measurements
        """
        ball_positions = []
        ball_vectors = []
        
        for i, frame in enumerate(release_frames):
            pose = frame.get('normalized_pose', {})
            ball = frame.get('normalized_ball', {})
            
            # Get eye positions
            left_eye = pose.get('left_eye', {})
            right_eye = pose.get('right_eye', {})
            
            if (self._has_valid_coordinates(left_eye, right_eye) and
                'center_x' in ball and 'center_y' in ball):
                
                # Calculate eye center
                eye_center_x = (left_eye.get('x', 0) + right_eye.get('x', 0)) / 2
                eye_center_y = (left_eye.get('y', 0) + right_eye.get('y', 0)) / 2
                
                # Calculate ball position relative to eyes
                ball_x = ball.get('center_x', 0)
                ball_y = ball.get('center_y', 0)
                
                relative_x = ball_x - eye_center_x
                relative_y = ball_y - eye_center_y
                
                ball_positions.append({
                    'relative_x': relative_x,
                    'relative_y': relative_y,
                    'ball_x': ball_x,
                    'ball_y': ball_y,
                    'eye_center_x': eye_center_x,
                    'eye_center_y': eye_center_y
                })
                
                # Calculate ball vector (velocity) if we have next frame
                if i < len(release_frames) - 1:
                    next_frame = release_frames[i + 1]
                    next_ball = next_frame.get('normalized_ball', {})
                    
                    if 'center_x' in next_ball and 'center_y' in next_ball:
                        next_ball_x = next_ball.get('center_x', 0)
                        next_ball_y = next_ball.get('center_y', 0)
                        
                        # Calculate velocity vector
                        velocity_x = next_ball_x - ball_x
                        velocity_y = next_ball_y - ball_y
                        
                        # Calculate vector magnitude and angle
                        magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
                        angle = np.degrees(np.arctan2(velocity_y, velocity_x))
                        
                        ball_vectors.append({
                            'velocity_x': velocity_x,
                            'velocity_y': velocity_y,
                            'magnitude': magnitude,
                            'angle': angle,
                            'frame_index': i
                        })
        
        result = {}
        
        if ball_positions:
            x_positions = [pos['relative_x'] for pos in ball_positions]
            y_positions = [pos['relative_y'] for pos in ball_positions]
            
            result.update({
                'average_relative_x': np.mean(x_positions),
                'average_relative_y': np.mean(y_positions),
                'min_relative_x': min(x_positions),
                'max_relative_x': max(x_positions),
                'min_relative_y': min(y_positions),
                'max_relative_y': max(y_positions),
                'positions': ball_positions
            })
        
        if ball_vectors:
            magnitudes = [vec['magnitude'] for vec in ball_vectors]
            angles = [vec['angle'] for vec in ball_vectors]
            velocity_x_values = [vec['velocity_x'] for vec in ball_vectors]
            velocity_y_values = [vec['velocity_y'] for vec in ball_vectors]
            
            result.update({
                'ball_vector': {
                    'average_magnitude': np.mean(magnitudes),
                    'average_angle': np.mean(angles),
                    'average_velocity_x': np.mean(velocity_x_values),
                    'average_velocity_y': np.mean(velocity_y_values),
                    'min_magnitude': min(magnitudes),
                    'max_magnitude': max(magnitudes),
                    'min_angle': min(angles),
                    'max_angle': max(angles),
                    'vectors': ball_vectors
                }
            })
        
        if not result:
            return {"error": "No valid ball position data found"}
        
        return result
    
    def _analyze_body_angles(self, release_frames: List[Dict]) -> Dict:
        """
        Analyze body tilt and leg angles at release point.
        
        Args:
            release_frames: List of release frame data
            
        Returns:
            Dictionary containing body angle measurements
        """
        body_measurements = []
        
        for frame in release_frames:
            pose = frame.get('normalized_pose', {})
            
            # Body tilt relative to hip vertical line
            body_tilt = self._calculate_body_tilt(pose)
            
            # Leg angles
            leg_angles = self._calculate_leg_angles(pose)
            
            # Additional angles for release phase
            upper_body_angle = self._calculate_upper_body_angle(pose)
            waist_angle = self._calculate_waist_angle(pose)
            thigh_angle = self._calculate_thigh_angle(pose)
            shoulder_elbow_wrist_angle = self._calculate_shoulder_elbow_wrist_angle(pose)
            wrist_shoulder_hip_angle = self._calculate_wrist_shoulder_hip_angle(pose)
            
            body_measurements.append({
                'body_tilt': body_tilt,
                'leg_angles': leg_angles,
                'upper_body_angle': upper_body_angle,
                'waist_angle': waist_angle,
                'thigh_angle': thigh_angle,
                'shoulder_elbow_wrist_angle': shoulder_elbow_wrist_angle,
                'wrist_shoulder_hip_angle': wrist_shoulder_hip_angle
            })
        
        if body_measurements:
            body_tilts = [m['body_tilt'] for m in body_measurements if m['body_tilt'] is not None]
            left_thigh_angles = [m['leg_angles'].get('left_thigh_angle') for m in body_measurements if m['leg_angles'].get('left_thigh_angle') is not None]
            left_leg_angles = [m['leg_angles'].get('left_leg_angle') for m in body_measurements if m['leg_angles'].get('left_leg_angle') is not None]
            right_thigh_angles = [m['leg_angles'].get('right_thigh_angle') for m in body_measurements if m['leg_angles'].get('right_thigh_angle') is not None]
            right_leg_angles = [m['leg_angles'].get('right_leg_angle') for m in body_measurements if m['leg_angles'].get('right_leg_angle') is not None]
            
            # Additional angles
            upper_body_angles = [m['upper_body_angle'] for m in body_measurements if m['upper_body_angle'] is not None]
            waist_angles = [m['waist_angle'] for m in body_measurements if m['waist_angle'] is not None]
            thigh_angles = [m['thigh_angle'] for m in body_measurements if m['thigh_angle'] is not None]
            shoulder_elbow_wrist_angles = [m['shoulder_elbow_wrist_angle'] for m in body_measurements if m['shoulder_elbow_wrist_angle'] is not None]
            wrist_shoulder_hip_angles = [m['wrist_shoulder_hip_angle'] for m in body_measurements if m['wrist_shoulder_hip_angle'] is not None]
            
            return {
                'body_tilt': {
                    'average': np.mean(body_tilts) if body_tilts else None,
                    'min': min(body_tilts) if body_tilts else None,
                    'max': max(body_tilts) if body_tilts else None
                },
                'leg_angles': {
                    'left_thigh_angle': {
                        'average': np.mean(left_thigh_angles) if left_thigh_angles else None,
                        'min': min(left_thigh_angles) if left_thigh_angles else None,
                        'max': max(left_thigh_angles) if left_thigh_angles else None
                    },
                    'left_leg_angle': {
                        'average': np.mean(left_leg_angles) if left_leg_angles else None,
                        'min': min(left_leg_angles) if left_leg_angles else None,
                        'max': max(left_leg_angles) if left_leg_angles else None
                    },
                    'right_thigh_angle': {
                        'average': np.mean(right_thigh_angles) if right_thigh_angles else None,
                        'min': min(right_thigh_angles) if right_thigh_angles else None,
                        'max': max(right_thigh_angles) if right_thigh_angles else None
                    },
                    'right_leg_angle': {
                        'average': np.mean(right_leg_angles) if right_leg_angles else None,
                        'min': min(right_leg_angles) if right_leg_angles else None,
                        'max': max(right_leg_angles) if right_leg_angles else None
                    }
                },
                'upper_body_angle': {
                    'average': np.mean(upper_body_angles) if upper_body_angles else None,
                    'min': min(upper_body_angles) if upper_body_angles else None,
                    'max': max(upper_body_angles) if upper_body_angles else None
                },
                'waist_angle': {
                    'average': np.mean(waist_angles) if waist_angles else None,
                    'min': min(waist_angles) if waist_angles else None,
                    'max': max(waist_angles) if waist_angles else None
                },
                'thigh_angle': {
                    'average': np.mean(thigh_angles) if thigh_angles else None,
                    'min': min(thigh_angles) if thigh_angles else None,
                    'max': max(thigh_angles) if thigh_angles else None
                },
                'shoulder_elbow_wrist_angle': {
                    'average': np.mean(shoulder_elbow_wrist_angles) if shoulder_elbow_wrist_angles else None,
                    'min': min(shoulder_elbow_wrist_angles) if shoulder_elbow_wrist_angles else None,
                    'max': max(shoulder_elbow_wrist_angles) if shoulder_elbow_wrist_angles else None
                },
                'wrist_shoulder_hip_angle': {
                    'average': np.mean(wrist_shoulder_hip_angles) if wrist_shoulder_hip_angles else None,
                    'min': min(wrist_shoulder_hip_angles) if wrist_shoulder_hip_angles else None,
                    'max': max(wrist_shoulder_hip_angles) if wrist_shoulder_hip_angles else None
                }
            }
        
        return {"error": "No valid body angle data found"}
    
    def _calculate_body_tilt(self, pose: Dict) -> Optional[float]:
        """Calculate body tilt relative to hip vertical line."""
        left_hip = pose.get('left_hip', {})
        right_hip = pose.get('right_hip', {})
        left_shoulder = pose.get('left_shoulder', {})
        right_shoulder = pose.get('right_shoulder', {})
        
        if not self._has_valid_coordinates(left_hip, right_hip, left_shoulder, right_shoulder):
            return None
        
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
    
    def _analyze_release_timing(self, release_frames: List[Dict], fps: float, all_frames: List[Dict]) -> Dict:
        """
        Analyze release timing information.
        
        Args:
            release_frames: List of release frame data
            fps: Frames per second
            
        Returns:
            Dictionary containing release timing measurements
        """
        if not release_frames:
            return {"error": "No release frames available"}
        
        total_release_time = len(release_frames) / fps
        
        # Find max jump frame from all frames to calculate relative timing
        max_jump_frame = self._find_max_jump_frame(all_frames)
        release_start_frame = release_frames[0] if release_frames else None
        
        relative_timing = 0
        if max_jump_frame and release_start_frame:
            max_jump_idx = all_frames.index(max_jump_frame)
            release_idx = all_frames.index(release_start_frame)
            relative_timing = (release_idx - max_jump_idx) / fps
        
        return {
            'total_release_time': total_release_time,
            'release_frames': len(release_frames),
            'average_frame_duration': 1.0 / fps,
            'relative_timing': relative_timing,  # Release timing relative to max jump
            'release_start_frame': release_start_frame.get('frame_index', 0) if release_start_frame else 0,
            'max_jump_frame': max_jump_frame.get('frame_index', 0) if max_jump_frame else 0
        }
    
    def _get_all_frames(self) -> List[Dict]:
        """Get all frames from the video data."""
        # This method should be implemented to get all frames from the video data
        # For now, we'll return an empty list - this should be passed from the main analyzer
        return []
    
    def _find_max_jump_frame(self, frames: List[Dict]) -> Optional[Dict]:
        """Find the frame with maximum jump height."""
        if not frames:
            return None
        
        max_jump_frame = None
        max_jump_height = float('inf')
        
        for frame in frames:
            # Calculate jump height based on foot position relative to baseline
            pose = frame.get('normalized_pose', {})
            left_ankle = pose.get('left_ankle', {})
            right_ankle = pose.get('right_ankle', {})
            
            if self._has_valid_coordinates(left_ankle, right_ankle):
                # Use average ankle height as jump height indicator
                ankle_height = (left_ankle.get('y', 0) + right_ankle.get('y', 0)) / 2
                
                if ankle_height < max_jump_height:
                    max_jump_height = ankle_height
                    max_jump_frame = frame
        
        return max_jump_frame
    
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
    
    def _calculate_upper_body_angle(self, pose: Dict) -> Optional[float]:
        """Calculate upper body angle (shoulder to hip line relative to vertical)."""
        left_shoulder = pose.get('left_shoulder', {})
        right_shoulder = pose.get('right_shoulder', {})
        left_hip = pose.get('left_hip', {})
        right_hip = pose.get('right_hip', {})
        
        if not self._has_valid_coordinates(left_shoulder, right_shoulder, left_hip, right_hip):
            return None
        
        # Calculate shoulder and hip centers
        shoulder_center_x = (left_shoulder.get('x', 0) + right_shoulder.get('x', 0)) / 2
        shoulder_center_y = (left_shoulder.get('y', 0) + right_shoulder.get('y', 0)) / 2
        hip_center_x = (left_hip.get('x', 0) + right_hip.get('x', 0)) / 2
        hip_center_y = (left_hip.get('y', 0) + right_hip.get('y', 0)) / 2
        
        # Calculate angle from vertical
        dx = shoulder_center_x - hip_center_x
        dy = shoulder_center_y - hip_center_y
        angle = np.degrees(np.arctan2(dx, abs(dy)))
        
        return angle
    
    def _calculate_waist_angle(self, pose: Dict) -> Optional[float]:
        """Calculate waist angle (hip to shoulder line relative to vertical)."""
        # This is similar to upper body angle but from hip perspective
        return self._calculate_upper_body_angle(pose)
    
    def _calculate_thigh_angle(self, pose: Dict) -> Optional[float]:
        """Calculate average thigh angle (hip to knee line relative to vertical)."""
        left_hip = pose.get('left_hip', {})
        right_hip = pose.get('right_hip', {})
        left_knee = pose.get('left_knee', {})
        right_knee = pose.get('right_knee', {})
        
        if not self._has_valid_coordinates(left_hip, left_knee, right_hip, right_knee):
            return None
        
        # Calculate left and right thigh angles
        left_thigh_angle = self._calculate_angle_to_vertical(left_hip, left_knee)
        right_thigh_angle = self._calculate_angle_to_vertical(right_hip, right_knee)
        
        # Return average
        return (left_thigh_angle + right_thigh_angle) / 2
    
    def _calculate_shoulder_elbow_wrist_angle(self, pose: Dict) -> Optional[float]:
        """Calculate angle between shoulder, elbow, and wrist (right arm)."""
        selected_shoulder = pose.get(f'{self.selected_hand}_shoulder', {})
        selected_elbow = pose.get(f'{self.selected_hand}_elbow', {})
        selected_wrist = pose.get(f'{self.selected_hand}_wrist', {})
        
        if not self._has_valid_coordinates(selected_shoulder, selected_elbow, selected_wrist):
            return None
        
        # Calculate angle between three points
        angle = self._calculate_angle(
            selected_shoulder.get('x', 0), selected_shoulder.get('y', 0),
            selected_elbow.get('x', 0), selected_elbow.get('y', 0),
            selected_wrist.get('x', 0), selected_wrist.get('y', 0)
        )
        
        return angle
    
    def _calculate_wrist_shoulder_hip_angle(self, pose: Dict) -> Optional[float]:
        """Calculate angle between wrist, shoulder, and hip (right side)."""
        selected_wrist = pose.get(f'{self.selected_hand}_wrist', {})
        selected_shoulder = pose.get(f'{self.selected_hand}_shoulder', {})
        selected_hip = pose.get(f'{self.selected_hand}_hip', {})
        
        if not self._has_valid_coordinates(selected_wrist, selected_shoulder, selected_hip):
            return None
        
        # Calculate angle between three points
        angle = self._calculate_angle(
            selected_wrist.get('x', 0), selected_wrist.get('y', 0),
            selected_shoulder.get('x', 0), selected_shoulder.get('y', 0),
            selected_hip.get('x', 0), selected_hip.get('y', 0)
        )
        
        return angle 