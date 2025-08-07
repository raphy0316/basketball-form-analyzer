"""
Loading Phase Analyzer

This module analyzes the loading phase of basketball shooting form.
It extracts key information from Loading and Loading-Rising phases combined.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json


class LoadingAnalyzer:
    """
    Analyzer for loading phase information.
    
    Extracts key measurements from Loading and Loading-Rising phases:
    - Maximum leg angles (hip-knee-ankle)
    - Maximum upper body tilt relative to hip vertical line
    - Loading-Rising transition time (seconds)
    - Total loading time (seconds)
    """
    
    def __init__(self):
        self.loading_data = {}
    
    def analyze_loading_phase(self, video_data: Dict) -> Dict:
        """
        Analyze loading phase information from video data.
        
        Args:
            video_data: Video analysis data containing frames
            
        Returns:
            Dictionary containing loading phase analysis results
        """
        frames = video_data.get('frames', [])
        if not frames:
            return {"error": "No frames available for analysis"}
        
        # Get FPS from metadata (default to 30fps)
        fps = video_data.get('metadata', {}).get('fps', 30.0)
        
        # Find all Loading and Loading-Rising frames
        loading_frames = []
        loading_rising_frames = []
        
        for frame in frames:
            phase = frame.get('phase', '')
            if phase == 'Loading':
                loading_frames.append(frame)
            elif phase == 'Loading-Rising':
                loading_rising_frames.append(frame)
        
        # Combine all loading-related frames
        all_loading_frames = loading_frames + loading_rising_frames
        
        if not all_loading_frames:
            return {"error": "No Loading or Loading-Rising frames found"}
        
        # Analyze loading phase
        loading_analysis = {
            'fps': fps,
            'total_loading_frames': len(all_loading_frames),
            'loading_frames': len(loading_frames),
            'loading_rising_frames': len(loading_rising_frames),
            'total_loading_time': len(all_loading_frames) / fps,
            'loading_rising_time': len(loading_rising_frames) / fps if loading_rising_frames else 0,
            'max_leg_angles': self._analyze_max_leg_angles(all_loading_frames),
            'max_upper_body_tilt': self._analyze_max_upper_body_tilt(all_loading_frames),
            'max_angle_to_transition': self._analyze_max_angle_to_transition(all_loading_frames, frames, fps)
        }
        
        return loading_analysis
    
    def _analyze_max_leg_angles(self, loading_frames: List[Dict]) -> Dict:
        """
        Analyze maximum leg angles (hip-knee-ankle) during loading phase.
        
        Args:
            loading_frames: List of loading frame data
            
        Returns:
            Dictionary containing maximum angle measurements
        """
        left_angles = []
        right_angles = []
        
        for frame in loading_frames:
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
                'max_angle': max(left_angles) if left_angles else None,
                'min_angle': min(left_angles) if left_angles else None,
                'average': np.mean(left_angles) if left_angles else None
            },
            'right': {
                'angles': right_angles,
                'max_angle': max(right_angles) if right_angles else None,
                'min_angle': min(right_angles) if right_angles else None,
                'average': np.mean(right_angles) if right_angles else None
            }
        }
    
    def _analyze_max_upper_body_tilt(self, loading_frames: List[Dict]) -> Dict:
        """
        Analyze maximum upper body tilt relative to hip vertical line.
        
        Args:
            loading_frames: List of loading frame data
            
        Returns:
            Dictionary containing maximum tilt measurements
        """
        shoulder_tilts = []
        
        for frame in loading_frames:
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
            'max_tilt': max(shoulder_tilts) if shoulder_tilts else None,
            'min_tilt': min(shoulder_tilts) if shoulder_tilts else None,
            'average': np.mean(shoulder_tilts) if shoulder_tilts else None
        }
    
    def _analyze_max_angle_to_transition(self, loading_frames: List[Dict], all_frames: List[Dict], fps: float) -> Dict:
        """
        Analyze the time from when both legs reach maximum angles to next phase transition.
        
        Args:
            loading_frames: List of loading frame data
            all_frames: All frames from the video
            fps: Frames per second
            
        Returns:
            Dictionary containing timing analysis from max angle to transition
        """
        # Find maximum leg angles and their frame indices
        left_angles_with_frames = []
        right_angles_with_frames = []
        
        for i, frame in enumerate(loading_frames):
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
                left_angles_with_frames.append((left_angle, i))
            
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
                right_angles_with_frames.append((right_angle, i))
        
        # Find maximum angles and their frame indices
        max_left_angle = None
        max_left_idx = None
        max_right_angle = None
        max_right_idx = None
        
        if left_angles_with_frames:
            max_left_angle, max_left_idx = max(left_angles_with_frames, key=lambda x: x[0])
        
        if right_angles_with_frames:
            max_right_angle, max_right_idx = max(right_angles_with_frames, key=lambda x: x[0])
        
        # Find when both legs reach their maximum angles (the later of the two)
        both_max_frame_idx = None
        if max_left_idx is not None and max_right_idx is not None:
            both_max_frame_idx = max(max_left_idx, max_right_idx)
        elif max_left_idx is not None:
            both_max_frame_idx = max_left_idx
        elif max_right_idx is not None:
            both_max_frame_idx = max_right_idx
        
        # Find next phase transition after loading frames
        next_transition_frame = self._find_next_transition_after_loading(all_frames, loading_frames)
        
        # Calculate timing from when both legs reach max to transition
        time_to_transition = 0.0
        if both_max_frame_idx is not None and next_transition_frame is not None:
            both_max_frame = loading_frames[both_max_frame_idx]
            time_to_transition = self._calculate_timing_to_transition(both_max_frame, next_transition_frame, all_frames, fps)
        
        return {
            'left_max_angle': max_left_angle if max_left_angle is not None else None,
            'right_max_angle': max_right_angle if max_right_angle is not None else None,
            'left_max_frame': max_left_idx if max_left_idx is not None else None,
            'right_max_frame': max_right_idx if max_right_idx is not None else None,
            'both_max_frame': both_max_frame_idx if both_max_frame_idx is not None else None,
            'time_to_transition': time_to_transition,
            'next_transition_frame': next_transition_frame.get('frame_index', 0) if next_transition_frame else None,
            'next_transition_phase': next_transition_frame.get('phase', 'Unknown') if next_transition_frame else None
        }
    
    def _find_next_transition_after_loading(self, all_frames: List[Dict], loading_frames: List[Dict]) -> Optional[Dict]:
        """Find the next phase transition after the loading frames."""
        if not loading_frames:
            return None
        
        # Get the last loading frame
        last_loading_frame = loading_frames[-1]
        last_loading_idx = all_frames.index(last_loading_frame)
        
        # Look for the next frame with a different phase
        for i in range(last_loading_idx + 1, len(all_frames)):
            frame = all_frames[i]
            if frame.get('phase', '') != last_loading_frame.get('phase', ''):
                return frame
        
        return None
    
    def _calculate_timing_to_transition(self, max_angle_frame: Optional[Dict], 
                                      transition_frame: Optional[Dict], 
                                      all_frames: List[Dict], fps: float) -> float:
        """Calculate time from max angle frame to transition frame."""
        if not max_angle_frame or not transition_frame:
            return 0.0
        
        max_angle_idx = all_frames.index(max_angle_frame)
        transition_idx = all_frames.index(transition_frame)
        
        frame_diff = transition_idx - max_angle_idx
        time_diff = frame_diff / fps
        
        return time_diff
    
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