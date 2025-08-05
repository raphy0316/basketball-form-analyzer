"""
Set-up Phase Analyzer

This module analyzes the set-up phase of basketball shooting form.
It extracts key information from the 3 frames before the Loading phase.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json


class SetupAnalyzer:
    """
    Analyzer for set-up phase information.
    
    Extracts key measurements from the 3 frames before Loading phase:
    - Hip-knee-ankle angles (both sides)
    - Foot positions
    - Shoulder tilt relative to hip vertical line
    - Ball position relative to hip (vertical and horizontal distance)
    """
    
    def __init__(self):
        self.setup_data = {}
    
    def analyze_setup_phase(self, video_data: Dict) -> Dict:
        """
        Analyze set-up phase information from video data.
        
        Args:
            video_data: Video analysis data containing frames
            
        Returns:
            Dictionary containing set-up phase analysis results
        """
        frames = video_data.get('frames', [])
        if not frames:
            return {"error": "No frames available for analysis"}
        
        # Get FPS from metadata (default to 30fps)
        fps = video_data.get('metadata', {}).get('fps', 30.0)
        
        # Calculate frame count based on FPS (30fps 기준 3프레임)
        # 30fps에서 3프레임 = 0.1초, 다른 FPS에서도 같은 시간을 사용
        target_duration = 0.1  # 3 frames at 30fps = 0.1 seconds
        frame_count = max(1, int(fps * target_duration))
        
        # Find the last Set-up frame before transitioning to Loading or Rising
        setup_end_frame = None
        
        # First, find all Set-up phases and their transitions
        setup_phases = []
        for i, frame in enumerate(frames):
            phase = frame.get('phase', '')
            if phase == 'Set-up':
                # Find the end of this Set-up phase
                setup_start = i
                setup_end = i
                for j in range(i, len(frames)):
                    if frames[j].get('phase') == 'Set-up':
                        setup_end = j
                    else:
                        break
                setup_phases.append((setup_start, setup_end))
        
        # Find the last Set-up phase that has a transition to Loading/Rising/Loading-Rising
        for setup_start, setup_end in reversed(setup_phases):
            # Check if there's a transition after this Set-up phase
            if setup_end + 1 < len(frames):
                next_phase = frames[setup_end + 1].get('phase', '')
                if next_phase in ['Loading', 'Rising', 'Loading-Rising']:
                    setup_end_frame = setup_end
                    break
        
        # If no transition found, use the last Set-up frame in the video
        if setup_end_frame is None and setup_phases:
            setup_end_frame = setup_phases[-1][1]
        
        if setup_end_frame is not None:
            # Use the last frame_count frames of Set-up phase
            setup_frames = []
            for i in range(max(0, setup_end_frame - frame_count + 1), setup_end_frame + 1):
                if i < len(frames):
                    setup_frames.append(frames[i])
            
            if setup_frames:
                setup_analysis = {
                    'frame_range': f"{setup_frames[0].get('frame_index', 0)}-{setup_frames[-1].get('frame_index', 0)}",
                    'frame_count': len(setup_frames),
                    'fps': fps,
                    'target_duration': target_duration,
                    'note': f'Using last {len(setup_frames)} Set-up frames before transition at {fps}fps ({target_duration}s)',
                    'hip_knee_ankle_angles': self._analyze_hip_knee_ankle_angles(setup_frames),
                    'foot_positions': self._analyze_foot_positions(setup_frames),
                    'shoulder_tilt': self._analyze_shoulder_tilt(setup_frames),
                    'ball_hip_distances': self._analyze_ball_hip_distances(setup_frames)
                }
                return setup_analysis
            else:
                return {"error": "No Set-up frames available"}
        else:
            return {"error": "No Set-up phase found in video"}
    
    def _analyze_hip_knee_ankle_angles(self, setup_frames: List[Dict]) -> Dict:
        """
        Analyze hip-knee-ankle angles for both sides.
        
        Args:
            setup_frames: List of set-up frame data
            
        Returns:
            Dictionary containing angle measurements
        """
        left_angles = []
        right_angles = []
        
        for frame in setup_frames:
            pose = frame.get('normalized_pose', {})
            
            # Left side
            left_hip = pose.get('left_hip', {})
            left_knee = pose.get('left_knee', {})
            left_ankle = pose.get('left_ankle', {})
            
            if self._has_valid_coordinates(left_hip, left_knee, left_ankle):
                left_angle = self._calculate_angle(
                    left_hip.get('x', 0), left_hip.get('y', 0),
                    left_knee.get('x', 0), left_knee.get('y', 0),
                    left_ankle.get('x', 0), left_ankle.get('y', 0)
                )
                left_angles.append(left_angle)
            
            # Right side
            right_hip = pose.get('right_hip', {})
            right_knee = pose.get('right_knee', {})
            right_ankle = pose.get('right_ankle', {})
            
            if self._has_valid_coordinates(right_hip, right_knee, right_ankle):
                right_angle = self._calculate_angle(
                    right_hip.get('x', 0), right_hip.get('y', 0),
                    right_knee.get('x', 0), right_knee.get('y', 0),
                    right_ankle.get('x', 0), right_ankle.get('y', 0)
                )
                right_angles.append(right_angle)
        
        return {
            'left': {
                'angles': left_angles,
                'average': np.mean(left_angles) if left_angles else 'Undefined',
                'std': np.std(left_angles) if len(left_angles) > 1 else 'Undefined'
            },
            'right': {
                'angles': right_angles,
                'average': np.mean(right_angles) if right_angles else 'Undefined',
                'std': np.std(right_angles) if len(right_angles) > 1 else 'Undefined'
            }
        }
    
    def _analyze_foot_positions(self, setup_frames: List[Dict]) -> Dict:
        """
        Analyze foot positions.
        
        Args:
            setup_frames: List of set-up frame data
            
        Returns:
            Dictionary containing foot position measurements
        """
        left_foot_positions = []
        right_foot_positions = []
        
        for frame in setup_frames:
            pose = frame.get('normalized_pose', {})
            
            # Left foot
            left_ankle = pose.get('left_ankle', {})
            if self._has_valid_coordinates(left_ankle):
                left_foot_positions.append({
                    'x': left_ankle.get('x', 0),
                    'y': left_ankle.get('y', 0)
                })
            
            # Right foot
            right_ankle = pose.get('right_ankle', {})
            if self._has_valid_coordinates(right_ankle):
                right_foot_positions.append({
                    'x': right_ankle.get('x', 0),
                    'y': right_ankle.get('y', 0)
                })
        
        return {
            'left_foot': {
                'positions': left_foot_positions,
                'average_x': np.mean([pos['x'] for pos in left_foot_positions]) if left_foot_positions else 'Undefined',
                'average_y': np.mean([pos['y'] for pos in left_foot_positions]) if left_foot_positions else 'Undefined',
                'std_x': np.std([pos['x'] for pos in left_foot_positions]) if len(left_foot_positions) > 1 else 'Undefined',
                'std_y': np.std([pos['y'] for pos in left_foot_positions]) if len(left_foot_positions) > 1 else 'Undefined'
            },
            'right_foot': {
                'positions': right_foot_positions,
                'average_x': np.mean([pos['x'] for pos in right_foot_positions]) if right_foot_positions else 'Undefined',
                'average_y': np.mean([pos['y'] for pos in right_foot_positions]) if right_foot_positions else 'Undefined',
                'std_x': np.std([pos['x'] for pos in right_foot_positions]) if len(right_foot_positions) > 1 else 'Undefined',
                'std_y': np.std([pos['y'] for pos in right_foot_positions]) if len(right_foot_positions) > 1 else 'Undefined'
            }
        }
    
    def _analyze_shoulder_tilt(self, setup_frames: List[Dict]) -> Dict:
        """
        Analyze shoulder tilt relative to hip vertical line.
        
        Args:
            setup_frames: List of set-up frame data
            
        Returns:
            Dictionary containing shoulder tilt measurements
        """
        shoulder_tilts = []
        
        for frame in setup_frames:
            pose = frame.get('normalized_pose', {})
            
            # Get hip and shoulder positions
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            left_shoulder = pose.get('left_shoulder', {})
            right_shoulder = pose.get('right_shoulder', {})
            
            if (self._has_valid_coordinates(left_hip, right_hip) and 
                self._has_valid_coordinates(left_shoulder, right_shoulder)):
                
                # Calculate hip center
                hip_center_x = (left_hip.get('x', 0) + right_hip.get('x', 0)) / 2
                hip_center_y = (left_hip.get('y', 0) + right_hip.get('y', 0)) / 2
                
                # Calculate shoulder center
                shoulder_center_x = (left_shoulder.get('x', 0) + right_shoulder.get('x', 0)) / 2
                shoulder_center_y = (left_shoulder.get('y', 0) + right_shoulder.get('y', 0)) / 2
                
                # Calculate tilt angle (angle between vertical line and hip-shoulder line)
                dx = shoulder_center_x - hip_center_x
                dy = shoulder_center_y - hip_center_y
                tilt_angle = np.degrees(np.arctan2(dx, abs(dy)))
                
                shoulder_tilts.append(tilt_angle)
        
        return {
            'angles': shoulder_tilts,
            'average': np.mean(shoulder_tilts) if shoulder_tilts else 'Undefined',
            'std': np.std(shoulder_tilts) if len(shoulder_tilts) > 1 else 'Undefined'
        }
    
    def _analyze_ball_hip_distances(self, setup_frames: List[Dict]) -> Dict:
        """
        Analyze ball position relative to hip (vertical and horizontal distance).
        
        Args:
            setup_frames: List of set-up frame data
            
        Returns:
            Dictionary containing ball-hip distance measurements
        """
        vertical_distances = []
        horizontal_distances = []
        
        for frame in setup_frames:
            pose = frame.get('normalized_pose', {})
            ball = frame.get('normalized_ball', {})
            
            # Get hip center
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            
            # Check if we have valid hip coordinates (at least one hip)
            has_valid_hip = (self._has_valid_coordinates(left_hip) or 
                           self._has_valid_coordinates(right_hip))
            
            # Check if ball has valid coordinates (not empty object and has center coordinates)
            has_valid_ball = (ball != {} and 
                            'center_x' in ball and 'center_y' in ball)
            
            if has_valid_hip and has_valid_ball:
                # Calculate hip center (use available hip)
                if self._has_valid_coordinates(left_hip) and self._has_valid_coordinates(right_hip):
                    hip_center_x = (left_hip.get('x', 0) + right_hip.get('x', 0)) / 2
                    hip_center_y = (left_hip.get('y', 0) + right_hip.get('y', 0)) / 2
                elif self._has_valid_coordinates(left_hip):
                    hip_center_x = left_hip.get('x', 0)
                    hip_center_y = left_hip.get('y', 0)
                else:  # right_hip is valid
                    hip_center_x = right_hip.get('x', 0)
                    hip_center_y = right_hip.get('y', 0)
                
                # Calculate distances using ball center coordinates
                ball_x = ball.get('center_x', 0)
                ball_y = ball.get('center_y', 0)
                
                vertical_distance = abs(ball_y - hip_center_y)
                horizontal_distance = abs(ball_x - hip_center_x)
                
                vertical_distances.append(vertical_distance)
                horizontal_distances.append(horizontal_distance)
        
        return {
            'vertical_distances': vertical_distances,
            'horizontal_distances': horizontal_distances,
            'average_vertical': np.mean(vertical_distances) if vertical_distances else 'Undefined',
            'average_horizontal': np.mean(horizontal_distances) if horizontal_distances else 'Undefined',
            'std_vertical': np.std(vertical_distances) if len(vertical_distances) > 1 else 'Undefined',
            'std_horizontal': np.std(horizontal_distances) if len(horizontal_distances) > 1 else 'Undefined'
        }
    
    def _has_valid_coordinates(self, *points) -> bool:
        """Check if all points have valid x, y coordinates."""
        for point in points:
            if not point or 'x' not in point or 'y' not in point:
                return False
        return True
    
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