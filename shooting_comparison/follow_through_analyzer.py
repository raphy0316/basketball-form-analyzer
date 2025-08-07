"""
Follow-through Phase Analyzer

This module analyzes the follow-through phase of basketball shooting form.
It extracts key information from Follow-through phase including angle stability analysis
and form maintenance duration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json


class FollowThroughAnalyzer:
    """
    Analyzer for follow-through phase information.
    
    Extracts key measurements from Follow-through phase:
    - Standard deviation of all body angles at maximum elbow angle point
    - Stable form maintenance time based on angle stability
    - Arm angle stability (shoulder, elbow, wrist)
    - Overall body stability analysis
    """
    
    def __init__(self):
        self.follow_through_data = {}
        self.stability_threshold = 5.0  # degrees - threshold for stable form
        self.selected_hand = 'right'  # default to right hand shooting
    def analyze_follow_through_phase(self, video_data: Dict, selected_hand) -> Dict:
        """
        Analyze follow-through phase information from video data.
        
        Args:
            video_data: Video analysis data containing frames
            
        Returns:
            Dictionary containing follow-through phase analysis results
        """
        self.selected_hand = selected_hand
        frames = video_data.get('frames', [])
        if not frames:
            return {"error": "No frames available for analysis"}
        
        # Get FPS from metadata (default to 30fps)
        fps = video_data.get('metadata', {}).get('fps', 30.0)
        
        # Find all Follow-through frames
        follow_through_frames = []
        
        for frame in frames:
            phase = frame.get('phase', '')
            if phase == 'Follow-through':
                follow_through_frames.append(frame)
        
        if not follow_through_frames:
            return {"error": "No Follow-through frames found"}
        
        # Analyze follow-through phase
        follow_through_analysis = {
            'fps': fps,
            'total_follow_through_frames': len(follow_through_frames),
            'total_follow_through_time': len(follow_through_frames) / fps,
            'max_elbow_angle_analysis': self._analyze_max_elbow_angle_point(follow_through_frames),
            'stability_analysis': self._analyze_form_stability(follow_through_frames, fps),
            'arm_stability': self._analyze_arm_stability(follow_through_frames),
            'overall_stability': self._analyze_overall_stability(follow_through_frames)
        }
        
        return follow_through_analysis
    
    def _analyze_max_elbow_angle_point(self, follow_through_frames: List[Dict]) -> Dict:
        """
        Analyze the point where elbow angle is maximum.
        
        Args:
            follow_through_frames: List of follow-through frame data
            
        Returns:
            Dictionary containing analysis at maximum elbow angle point
        """
        if not follow_through_frames:
            return {"error": "No follow-through frames available"}
        
        # Find frame with maximum elbow angle
        max_elbow_angle = -1
        max_elbow_frame = None
        max_elbow_frame_idx = -1
        
        for i, frame in enumerate(follow_through_frames):
            pose = frame.get('normalized_pose', {})
            
            # Get elbow angle
            elbow_angle = self._calculate_elbow_angle(pose)
            if elbow_angle is not None and elbow_angle > max_elbow_angle:
                max_elbow_angle = elbow_angle
                max_elbow_frame = frame
                max_elbow_frame_idx = i
        
        if max_elbow_frame is None:
            return {"error": "Could not find maximum elbow angle frame"}
        
        # Calculate standard deviation of all angles at max elbow angle point
        pose = max_elbow_frame.get('normalized_pose', {})
        
        angle_std_analysis = {
            'max_elbow_angle': max_elbow_angle,
            'max_elbow_frame_idx': max_elbow_frame_idx,
            'arm_angles_std': self._calculate_arm_angles_std(pose),
            'body_angles_std': self._calculate_body_angles_std(pose),
            'leg_angles_std': self._calculate_leg_angles_std(pose),
            'overall_angles_std': self._calculate_overall_angles_std(pose)
        }
        
        return angle_std_analysis
    
    def _analyze_form_stability(self, follow_through_frames: List[Dict], fps: float) -> Dict:
        """
        Analyze form stability duration based on angle standard deviation threshold.
        
        Args:
            follow_through_frames: List of follow-through frame data
            fps: Frames per second
            
        Returns:
            Dictionary containing stability analysis
        """
        if not follow_through_frames:
            return {"error": "No follow-through frames available"}
        
        # Calculate stability for each frame
        stability_data = []
        
        for i, frame in enumerate(follow_through_frames):
            pose = frame.get('normalized_pose', {})
            
            # Calculate overall angle standard deviation for this frame
            overall_std = self._calculate_overall_angles_std(pose)
            
            # Determine if frame is stable (std below threshold)
            is_stable = overall_std <= self.stability_threshold if overall_std is not None else False
            
            stability_data.append({
                'frame_idx': i,
                'overall_std': overall_std,
                'is_stable': is_stable
            })
        
        # Calculate stable duration
        stable_frames = [data for data in stability_data if data['is_stable']]
        stable_duration = len(stable_frames) / fps if stable_frames else 0
        
        # Calculate arm-specific stability
        arm_stable_frames = self._calculate_arm_stable_frames(follow_through_frames)
        arm_stable_duration = len(arm_stable_frames) / fps if arm_stable_frames else 0
        
        # Calculate other body parts stability
        other_stable_frames = self._calculate_other_body_stable_frames(follow_through_frames)
        other_stable_duration = len(other_stable_frames) / fps if other_stable_frames else 0
        
        stability_analysis = {
            'stability_threshold': self.stability_threshold,
            'total_follow_through_duration': len(follow_through_frames) / fps,
            'overall_stable_duration': stable_duration,
            'arm_stable_duration': arm_stable_duration,
            'other_body_stable_duration': other_stable_duration,
            'stability_percentage': (stable_duration / (len(follow_through_frames) / fps)) * 100 if follow_through_frames else 0,
            'arm_stability_percentage': (arm_stable_duration / (len(follow_through_frames) / fps)) * 100 if follow_through_frames else 0,
            'other_stability_percentage': (other_stable_duration / (len(follow_through_frames) / fps)) * 100 if follow_through_frames else 0
        }
        
        return stability_analysis
    
    def _analyze_arm_stability(self, follow_through_frames: List[Dict]) -> Dict:
        """
        Analyze arm angle stability (shoulder, elbow, wrist).
        
        Args:
            follow_through_frames: List of follow-through frame data
            
        Returns:
            Dictionary containing arm stability analysis
        """
        arm_angles = []
        
        for frame in follow_through_frames:
            pose = frame.get('normalized_pose', {})
            
            # Calculate arm angles
            shoulder_angle = self._calculate_shoulder_angle(pose)
            elbow_angle = self._calculate_elbow_angle(pose)
            # wrist_angle = self._calculate_wrist_angle(pose)
            
            if all(angle is not None for angle in [shoulder_angle, elbow_angle]):
                arm_angles.append({
                    'shoulder_angle': shoulder_angle,
                    'elbow_angle': elbow_angle,
                    # 'wrist_angle': wrist_angle
                })
        
        if not arm_angles:
            return {"error": "No valid arm angles found"}
        
        # Calculate statistics
        shoulder_angles = [angle['shoulder_angle'] for angle in arm_angles]
        elbow_angles = [angle['elbow_angle'] for angle in arm_angles]
        # wrist_angles = [angle['wrist_angle'] for angle in arm_angles]
        
        arm_stability = {
            'shoulder_angle': {
                'average': np.mean(shoulder_angles),
                'std': np.std(shoulder_angles),
                'min': np.min(shoulder_angles),
                'max': np.max(shoulder_angles)
            },
            'elbow_angle': {
                'average': np.mean(elbow_angles),
                'std': np.std(elbow_angles),
                'min': np.min(elbow_angles),
                'max': np.max(elbow_angles)
            },
            # 'wrist_angle': {
            #     'average': np.mean(wrist_angles),
            #     'std': np.std(wrist_angles),
            #     'min': np.min(wrist_angles),
            #     'max': np.max(wrist_angles)
            # }
        }

        return arm_stability
    
    def _analyze_overall_stability(self, follow_through_frames: List[Dict]) -> Dict:
        """
        Analyze overall body stability excluding arms.
        
        Args:
            follow_through_frames: List of follow-through frame data
            
        Returns:
            Dictionary containing overall stability analysis
        """
        body_angles = []
        
        for frame in follow_through_frames:
            pose = frame.get('normalized_pose', {})
            
            # Calculate body angles (excluding arms)
            hip_angle = self._calculate_hip_angle(pose)
            knee_angle = self._calculate_knee_angle(pose)
            # ankle_angle = self._calculate_ankle_angle(pose)
            torso_angle = self._calculate_torso_angle(pose)
            
            if all(angle is not None for angle in [hip_angle, knee_angle, torso_angle]):
                body_angles.append({
                    'hip_angle': hip_angle,
                    'knee_angle': knee_angle,
                    # 'ankle_angle': ankle_angle,
                    'torso_angle': torso_angle
                })
        
        if not body_angles:
            return {"error": "No valid body angles found"}
        
        # Calculate statistics
        hip_angles = [angle['hip_angle'] for angle in body_angles]
        knee_angles = [angle['knee_angle'] for angle in body_angles]
        # ankle_angles = [angle['ankle_angle'] for angle in body_angles]
        torso_angles = [angle['torso_angle'] for angle in body_angles]
        
        overall_stability = {
            'hip_angle': {
                'average': np.mean(hip_angles),
                'std': np.std(hip_angles),
                'min': np.min(hip_angles),
                'max': np.max(hip_angles)
            },
            'knee_angle': {
                'average': np.mean(knee_angles),
                'std': np.std(knee_angles),
                'min': np.min(knee_angles),
                'max': np.max(knee_angles)
            },
            # 'ankle_angle': {
            #     'average': np.mean(ankle_angles),
            #     'std': np.std(ankle_angles),
            #     'min': np.min(ankle_angles),
            #     'max': np.max(ankle_angles)
            # },
            'torso_angle': {
                'average': np.mean(torso_angles),
                'std': np.std(torso_angles),
                'min': np.min(torso_angles),
                'max': np.max(torso_angles)
            }
        }
        
        return overall_stability
    
    def _calculate_elbow_angle(self, pose: Dict) -> Optional[float]:
        """Calculate elbow angle (shoulder-elbow-wrist)."""
        shoulder = pose.get(f'{self.selected_hand}_shoulder', {})
        elbow = pose.get(f'{self.selected_hand}_elbow', {})
        wrist = pose.get(f'{self.selected_hand}_wrist', {})
        
        if self._has_valid_coordinates(shoulder, elbow, wrist):
            return self._calculate_angle(
                shoulder.get('x', 0), shoulder.get('y', 0),
                elbow.get('x', 0), elbow.get('y', 0),
                wrist.get('x', 0), wrist.get('y', 0)
            )
        return None
    
    def _calculate_shoulder_angle(self, pose: Dict) -> Optional[float]:
        """Calculate shoulder angle (hip-shoulder-elbow)."""
        hip = pose.get(f'{self.selected_hand}_hip', {})
        shoulder = pose.get(f'{self.selected_hand}_shoulder', {})
        elbow = pose.get(f'{self.selected_hand}_elbow', {})
        
        if self._has_valid_coordinates(hip, shoulder, elbow):
            return self._calculate_angle(
                hip.get('x', 0), hip.get('y', 0),
                shoulder.get('x', 0), shoulder.get('y', 0),
                elbow.get('x', 0), elbow.get('y', 0)
            )
        return None
    
    # def _calculate_wrist_angle(self, pose: Dict) -> Optional[float]:
    #     """Calculate wrist angle (elbow-wrist-finger)."""
    #     elbow = pose.get(f'{self.selected_hand}_elbow', {})
    #     wrist = pose.get(f'{self.selected_hand}_wrist', {})
    #     # Use a point above wrist as finger reference
    #     finger_x = wrist.get('x', 0)
    #     finger_y = wrist.get('y', 0) - 20  # 20 pixels above wrist
        
    #     if self._has_valid_coordinates(elbow, wrist):
    #         return self._calculate_angle(
    #             elbow.get('x', 0), elbow.get('y', 0),
    #             wrist.get('x', 0), wrist.get('y', 0),
    #             finger_x, finger_y
    #         )
    #     return None
    
    def _calculate_hip_angle(self, pose: Dict) -> Optional[float]:
        """Calculate hip angle (shoulder-hip-knee)."""
        shoulder = pose.get(f'{self.selected_hand}_shoulder', {})
        hip = pose.get(f'{self.selected_hand}_hip', {})
        knee = pose.get(f'{self.selected_hand}_knee', {})
        
        if self._has_valid_coordinates(shoulder, hip, knee):
            return self._calculate_angle(
                shoulder.get('x', 0), shoulder.get('y', 0),
                hip.get('x', 0), hip.get('y', 0),
                knee.get('x', 0), knee.get('y', 0)
            )
        return None
    
    def _calculate_knee_angle(self, pose: Dict) -> Optional[float]:
        """Calculate knee angle (hip-knee-ankle)."""
        hip = pose.get(f'{self.selected_hand}_hip', {})
        knee = pose.get(f'{self.selected_hand}_knee', {})
        ankle = pose.get(f'{self.selected_hand}_ankle', {})
        
        if self._has_valid_coordinates(hip, knee, ankle):
            return self._calculate_angle(
                hip.get('x', 0), hip.get('y', 0),
                knee.get('x', 0), knee.get('y', 0),
                ankle.get('x', 0), ankle.get('y', 0)
            )
        return None
    
    # def _calculate_ankle_angle(self, pose: Dict) -> Optional[float]:
    #     """Calculate ankle angle (knee-ankle-toe)."""
    #     knee = pose.get(f'{self.selected_hand}_knee', {})
    #     ankle = pose.get(f'{self.selected_hand}_ankle', {})
    #     # Use a point below ankle as toe reference
    #     toe_x = ankle.get('x', 0)
    #     toe_y = ankle.get('y', 0) + 20  # 20 pixels below ankle
        
    #     if self._has_valid_coordinates(knee, ankle):
    #         return self._calculate_angle(
    #             knee.get('x', 0), knee.get('y', 0),
    #             ankle.get('x', 0), ankle.get('y', 0),
    #             toe_x, toe_y
    #         )
    #     return None
    
    def _calculate_torso_angle(self, pose: Dict) -> Optional[float]:
        """Calculate torso angle relative to vertical."""
        shoulder = pose.get(f'{self.selected_hand}_shoulder', {})
        hip = pose.get(f'{self.selected_hand}_hip', {})
        
        if self._has_valid_coordinates(shoulder, hip):
            # Calculate angle relative to vertical
            dx = shoulder.get('x', 0) - hip.get('x', 0)
            dy = shoulder.get('y', 0) - hip.get('y', 0)
            angle = np.degrees(np.arctan2(dx, dy))
            return abs(angle)
        return None
    
    def _calculate_arm_angles_std(self, pose: Dict) -> Optional[float]:
        """Calculate standard deviation of arm angles."""
        angles = []
        
        shoulder_angle = self._calculate_shoulder_angle(pose)
        elbow_angle = self._calculate_elbow_angle(pose)
        # wrist_angle = self._calculate_wrist_angle(pose)
        
        if shoulder_angle is not None:
            angles.append(shoulder_angle)
        if elbow_angle is not None:
            angles.append(elbow_angle)
        # if wrist_angle is not None:
        #     angles.append(wrist_angle)
        
        return np.std(angles) if angles else None
    
    def _calculate_body_angles_std(self, pose: Dict) -> Optional[float]:
        """Calculate standard deviation of body angles."""
        angles = []
        
        hip_angle = self._calculate_hip_angle(pose)
        knee_angle = self._calculate_knee_angle(pose)
        # ankle_angle = self._calculate_ankle_angle(pose)
        torso_angle = self._calculate_torso_angle(pose)
        
        if hip_angle is not None:
            angles.append(hip_angle)
        if knee_angle is not None:
            angles.append(knee_angle)
        # if ankle_angle is not None:
        #     angles.append(ankle_angle)
        if torso_angle is not None:
            angles.append(torso_angle)
        
        return np.std(angles) if angles else None
    
    def _calculate_leg_angles_std(self, pose: Dict) -> Optional[float]:
        """Calculate standard deviation of leg angles."""
        angles = []
        
        hip_angle = self._calculate_hip_angle(pose)
        knee_angle = self._calculate_knee_angle(pose)
        # ankle_angle = self._calculate_ankle_angle(pose)
        
        if hip_angle is not None:
            angles.append(hip_angle)
        if knee_angle is not None:
            angles.append(knee_angle)
        # if ankle_angle is not None:
        #     angles.append(ankle_angle)
        
        return np.std(angles) if angles else None
    
    def _calculate_overall_angles_std(self, pose: Dict) -> Optional[float]:
        """Calculate standard deviation of all angles."""
        angles = []
        
        # Arm angles
        shoulder_angle = self._calculate_shoulder_angle(pose)
        elbow_angle = self._calculate_elbow_angle(pose)
        # wrist_angle = self._calculate_wrist_angle(pose)
        
        # Body angles
        hip_angle = self._calculate_hip_angle(pose)
        knee_angle = self._calculate_knee_angle(pose)
        # ankle_angle = self._calculate_ankle_angle(pose)
        torso_angle = self._calculate_torso_angle(pose)
        
        # Add all valid angles
        for angle in [shoulder_angle, elbow_angle, 
                     hip_angle, knee_angle, torso_angle]:
            if angle is not None:
                angles.append(angle)
        
        return np.std(angles) if angles else None
    
    def _calculate_arm_stable_frames(self, follow_through_frames: List[Dict]) -> List[int]:
        """Calculate frames where arm angles are stable."""
        stable_frames = []
        
        for i, frame in enumerate(follow_through_frames):
            pose = frame.get('normalized_pose', {})
            arm_std = self._calculate_arm_angles_std(pose)
            
            if arm_std is not None and arm_std <= self.stability_threshold:
                stable_frames.append(i)
        
        return stable_frames
    
    def _calculate_other_body_stable_frames(self, follow_through_frames: List[Dict]) -> List[int]:
        """Calculate frames where other body angles are stable."""
        stable_frames = []
        
        for i, frame in enumerate(follow_through_frames):
            pose = frame.get('normalized_pose', {})
            body_std = self._calculate_body_angles_std(pose)
            
            if body_std is not None and body_std <= self.stability_threshold:
                stable_frames.append(i)
        
        return stable_frames
    
    def _calculate_angle(self, ax: float, ay: float, bx: float, by: float, 
                        cx: float, cy: float) -> float:
        """Calculate angle between three points."""
        # Vector AB
        ab_x = bx - ax
        ab_y = by - ay
        
        # Vector CB
        cb_x = bx - cx
        cb_y = by - cy
        
        # Dot product
        dot_product = ab_x * cb_x + ab_y * cb_y
        
        # Magnitudes
        ab_magnitude = np.sqrt(ab_x**2 + ab_y**2)
        cb_magnitude = np.sqrt(cb_x**2 + cb_y**2)
        
        # Avoid division by zero
        if ab_magnitude == 0 or cb_magnitude == 0:
            return 0.0
        
        # Calculate angle
        cos_angle = dot_product / (ab_magnitude * cb_magnitude)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to valid range
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def _has_valid_coordinates(self, *points) -> bool:
        """Check if all points have valid coordinates."""
        for point in points:
            if not point or 'x' not in point or 'y' not in point:
                return False
        return True 