"""
Landing Phase Analyzer

This module analyzes the landing phase after follow-through in basketball shooting form.
It extracts key information about foot landing position and torso angle at landing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json


class LandingAnalyzer:
    """
    Analyzer for landing phase information after follow-through.
    
    Extracts key measurements from landing phase:
    - Foot landing position comparison with setup phase
    - Landing detection based on foot position stability
    - Torso angle relative to vertical at landing point
    - Landing timing relative to follow-through
    """
    
    def __init__(self):
        self.landing_data = {}
        self.landing_threshold = 0.02  # Threshold for detecting landing (foot position change)
        self.stability_threshold = 0.01  # Threshold for stable foot position
    
    def analyze_landing_phase(self, video_data: Dict) -> Dict:
        """
        Analyze landing phase information from video data.
        
        Args:
            video_data: Video analysis data containing frames
            
        Returns:
            Dictionary containing landing phase analysis results
        """
        frames = video_data.get('frames', [])
        if not frames:
            return {"error": "No frames available for analysis"}
        
        # Get FPS from metadata (default to 30fps)
        fps = video_data.get('metadata', {}).get('fps', 30.0)
        
        # Find setup phase foot positions (reference for landing)
        setup_foot_positions = self._get_setup_foot_positions(frames)
        
        # Find follow-through and post follow-through frames
        follow_through_frames, post_follow_through_frames = self._get_follow_through_and_post_frames(frames)
        
        if not follow_through_frames:
            return {"error": "No Follow-through frames found"}
        
        # Analyze landing phase
        landing_analysis = {
            'fps': fps,
            'setup_foot_positions': setup_foot_positions,
            'follow_through_frames_count': len(follow_through_frames),
            'post_follow_through_frames_count': len(post_follow_through_frames),
            'landing_detection': self._detect_landing(post_follow_through_frames, setup_foot_positions),
            'landing_position_analysis': self._analyze_landing_position(post_follow_through_frames, setup_foot_positions),
            'landing_torso_analysis': self._analyze_landing_torso_angle(post_follow_through_frames),
            'landing_timing': self._analyze_landing_timing(follow_through_frames, post_follow_through_frames, fps)
        }
        
        return landing_analysis
    
    def _get_setup_foot_positions(self, frames: List[Dict]) -> Dict:
        """
        Get foot positions from setup phase as reference.
        
        Args:
            frames: List of frame data
            
        Returns:
            Dictionary containing setup foot positions
        """
        setup_frames = []
        
        # Find setup frames (last 3 frames before Loading/Rising transition)
        for i, frame in enumerate(frames):
            phase = frame.get('phase', '')
            if phase == 'Set-up':
                setup_frames.append(frame)
        
        if not setup_frames:
            return {"error": "No Set-up frames found"}
        
        # Use last 3 setup frames
        setup_frames = setup_frames[-3:] if len(setup_frames) >= 3 else setup_frames
        
        # Calculate average foot positions
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
                'average_y': np.mean([pos['y'] for pos in left_foot_positions]) if left_foot_positions else 'Undefined'
            },
            'right_foot': {
                'positions': right_foot_positions,
                'average_x': np.mean([pos['x'] for pos in right_foot_positions]) if right_foot_positions else 'Undefined',
                'average_y': np.mean([pos['y'] for pos in right_foot_positions]) if right_foot_positions else 'Undefined'
            }
        }
    
    def _get_follow_through_and_post_frames(self, frames: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Get follow-through frames and frames after follow-through.
        
        Args:
            frames: List of frame data
            
        Returns:
            Tuple of (follow_through_frames, post_follow_through_frames)
        """
        follow_through_frames = []
        post_follow_through_frames = []
        in_follow_through = False
        follow_through_ended = False
        
        for frame in frames:
            phase = frame.get('phase', '')
            
            if phase == 'Follow-through':
                follow_through_frames.append(frame)
                in_follow_through = True
                follow_through_ended = False
            elif in_follow_through and phase != 'Follow-through':
                # First frame after follow-through
                post_follow_through_frames.append(frame)
                in_follow_through = False
                follow_through_ended = True
            elif follow_through_ended:
                # Continue collecting post follow-through frames
                post_follow_through_frames.append(frame)
        
        return follow_through_frames, post_follow_through_frames
    
    def _detect_landing(self, post_follow_through_frames: List[Dict], setup_foot_positions: Dict) -> Dict:
        """
        Detect if landing occurred by comparing foot positions with setup positions.
        
        Args:
            post_follow_through_frames: Frames after follow-through
            setup_foot_positions: Reference foot positions from setup
            
        Returns:
            Dictionary containing landing detection results
        """
        if not post_follow_through_frames:
            return {"error": "No post follow-through frames available"}
        
        if 'error' in setup_foot_positions:
            return {"error": f"Setup foot positions error: {setup_foot_positions['error']}"}
        
        # Get setup reference positions
        setup_left_x = setup_foot_positions.get('left_foot', {}).get('average_x', 0)
        setup_left_y = setup_foot_positions.get('left_foot', {}).get('average_y', 0)
        setup_right_x = setup_foot_positions.get('right_foot', {}).get('average_x', 0)
        setup_right_y = setup_foot_positions.get('right_foot', {}).get('average_y', 0)
        
        # Check if setup positions are valid
        if (setup_left_x == 'Undefined' or setup_left_y == 'Undefined' or 
            setup_right_x == 'Undefined' or setup_right_y == 'Undefined'):
            return {"error": "Invalid setup foot positions"}
        
        # Analyze foot positions in post follow-through frames
        landing_detected = False
        landing_frame_idx = -1
        landing_foot_positions = []
        
        for i, frame in enumerate(post_follow_through_frames):
            pose = frame.get('normalized_pose', {})
            
            # Get current foot positions
            left_ankle = pose.get('left_ankle', {})
            right_ankle = pose.get('right_ankle', {})
            
            if self._has_valid_coordinates(left_ankle, right_ankle):
                left_x = left_ankle.get('x', 0)
                left_y = left_ankle.get('y', 0)
                right_x = right_ankle.get('x', 0)
                right_y = right_ankle.get('y', 0)
                
                # Calculate distance from setup positions
                left_distance = np.sqrt((left_x - setup_left_x)**2 + (left_y - setup_left_y)**2)
                right_distance = np.sqrt((right_x - setup_right_x)**2 + (right_y - setup_right_y)**2)
                
                # Check if feet are close to setup positions (landing detected)
                if left_distance <= self.landing_threshold and right_distance <= self.landing_threshold:
                    if not landing_detected:
                        landing_detected = True
                        landing_frame_idx = i
                    
                    landing_foot_positions.append({
                        'frame_idx': i,
                        'left_x': left_x,
                        'left_y': left_y,
                        'right_x': right_x,
                        'right_y': right_y,
                        'left_distance': left_distance,
                        'right_distance': right_distance
                    })
        
        # Check for stable landing (multiple consecutive frames)
        stable_landing = False
        if len(landing_foot_positions) >= 3:  # At least 3 consecutive frames
            # Check if positions are stable
            left_x_positions = [pos['left_x'] for pos in landing_foot_positions]
            left_y_positions = [pos['left_y'] for pos in landing_foot_positions]
            right_x_positions = [pos['right_x'] for pos in landing_foot_positions]
            right_y_positions = [pos['right_y'] for pos in landing_foot_positions]
            
            left_x_std = np.std(left_x_positions)
            left_y_std = np.std(left_y_positions)
            right_x_std = np.std(right_x_positions)
            right_y_std = np.std(right_y_positions)
            
            # Check if positions are stable (low standard deviation)
            if (left_x_std <= self.stability_threshold and left_y_std <= self.stability_threshold and
                right_x_std <= self.stability_threshold and right_y_std <= self.stability_threshold):
                stable_landing = True
        
        return {
            'landing_detected': landing_detected,
            'stable_landing': stable_landing,
            'landing_frame_idx': landing_frame_idx,
            'landing_foot_positions': landing_foot_positions,
            'landing_threshold': self.landing_threshold,
            'stability_threshold': self.stability_threshold
        }
    
    def _analyze_landing_position(self, post_follow_through_frames: List[Dict], setup_foot_positions: Dict) -> Dict:
        """
        Analyze landing position compared to setup position.
        
        Args:
            post_follow_through_frames: Frames after follow-through
            setup_foot_positions: Reference foot positions from setup
            
        Returns:
            Dictionary containing landing position analysis
        """
        if not post_follow_through_frames:
            return {"error": "No post follow-through frames available"}
        
        if 'error' in setup_foot_positions:
            return {"error": f"Setup foot positions error: {setup_foot_positions['error']}"}
        
        # Get setup reference positions
        setup_left_x = setup_foot_positions.get('left_foot', {}).get('average_x', 0)
        setup_left_y = setup_foot_positions.get('left_foot', {}).get('average_y', 0)
        setup_right_x = setup_foot_positions.get('right_foot', {}).get('average_x', 0)
        setup_right_y = setup_foot_positions.get('right_foot', {}).get('average_y', 0)
        
        # Check if setup positions are valid
        if (setup_left_x == 'Undefined' or setup_left_y == 'Undefined' or 
            setup_right_x == 'Undefined' or setup_right_y == 'Undefined'):
            return {"error": "Invalid setup foot positions"}
        
        # Analyze all post follow-through frames
        left_foot_positions = []
        right_foot_positions = []
        
        for frame in post_follow_through_frames:
            pose = frame.get('normalized_pose', {})
            
            # Get current foot positions
            left_ankle = pose.get('left_ankle', {})
            right_ankle = pose.get('right_ankle', {})
            
            if self._has_valid_coordinates(left_ankle, right_ankle):
                left_x = left_ankle.get('x', 0)
                left_y = left_ankle.get('y', 0)
                right_x = right_ankle.get('x', 0)
                right_y = right_ankle.get('y', 0)
                
                left_foot_positions.append({
                    'x': left_x,
                    'y': left_y,
                    'distance_from_setup': np.sqrt((left_x - setup_left_x)**2 + (left_y - setup_left_y)**2)
                })
                
                right_foot_positions.append({
                    'x': right_x,
                    'y': right_y,
                    'distance_from_setup': np.sqrt((right_x - setup_right_x)**2 + (right_y - setup_right_y)**2)
                })
        
        if not left_foot_positions or not right_foot_positions:
            return {"error": "No valid foot positions found"}
        
        # Calculate statistics
        left_distances = [pos['distance_from_setup'] for pos in left_foot_positions]
        right_distances = [pos['distance_from_setup'] for pos in right_foot_positions]
        
        return {
            'left_foot': {
                'positions': left_foot_positions,
                'average_distance_from_setup': np.mean(left_distances),
                'min_distance_from_setup': np.min(left_distances),
                'max_distance_from_setup': np.max(left_distances),
                'std_distance_from_setup': np.std(left_distances)
            },
            'right_foot': {
                'positions': right_foot_positions,
                'average_distance_from_setup': np.mean(right_distances),
                'min_distance_from_setup': np.min(right_distances),
                'max_distance_from_setup': np.max(right_distances),
                'std_distance_from_setup': np.std(right_distances)
            },
            'setup_reference': {
                'left_x': setup_left_x,
                'left_y': setup_left_y,
                'right_x': setup_right_x,
                'right_y': setup_right_y
            }
        }
    
    def _analyze_landing_torso_angle(self, post_follow_through_frames: List[Dict]) -> Dict:
        """
        Analyze torso angle relative to vertical at landing point.
        
        Args:
            post_follow_through_frames: Frames after follow-through
            
        Returns:
            Dictionary containing torso angle analysis
        """
        if not post_follow_through_frames:
            return {"error": "No post follow-through frames available"}
        
        torso_angles = []
        
        for frame in post_follow_through_frames:
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
                
                # Calculate torso angle relative to vertical
                dx = shoulder_center_x - hip_center_x
                dy = shoulder_center_y - hip_center_y
                torso_angle = np.degrees(np.arctan2(dx, abs(dy)))
                
                torso_angles.append(torso_angle)
        
        if not torso_angles:
            return {"error": "No valid torso angles found"}
        
        return {
            'torso_angles': torso_angles,
            'average_torso_angle': np.mean(torso_angles),
            'std_torso_angle': np.std(torso_angles),
            'min_torso_angle': np.min(torso_angles),
            'max_torso_angle': np.max(torso_angles)
        }
    
    def _analyze_landing_timing(self, follow_through_frames: List[Dict], 
                               post_follow_through_frames: List[Dict], fps: float) -> Dict:
        """
        Analyze landing timing relative to follow-through.
        
        Args:
            follow_through_frames: Follow-through frames
            post_follow_through_frames: Frames after follow-through
            fps: Frames per second
            
        Returns:
            Dictionary containing landing timing analysis
        """
        if not follow_through_frames:
            return {"error": "No follow-through frames found"}
        
        follow_through_duration = len(follow_through_frames) / fps
        post_follow_through_duration = len(post_follow_through_frames) / fps if post_follow_through_frames else 0
        
        return {
            'follow_through_duration': follow_through_duration,
            'post_follow_through_duration': post_follow_through_duration,
            'total_analysis_duration': follow_through_duration + post_follow_through_duration
        }
    
    def _has_valid_coordinates(self, *points) -> bool:
        """Check if all points have valid coordinates."""
        for point in points:
            if not point or 'x' not in point or 'y' not in point:
                return False
        return True 