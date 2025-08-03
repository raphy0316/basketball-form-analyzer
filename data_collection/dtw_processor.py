"""
DTW (Dynamic Time Warping) Processor Module

This module provides DTW analysis functionality for basketball shooting phases.
It includes both coordinate-based and feature-based DTW analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import json
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pandas as pd


class DTWProcessor:
    """
    DTW Processor for basketball shooting analysis.
    
    Provides DTW analysis for:
    1. Coordinate-based analysis (raw joint coordinates)
    2. Feature-based analysis (angles, distances, etc.)
    3. Phase-specific analysis with different features for each phase
    """
    
    def __init__(self):
        self.data = []
        
        # Overall phase features (for complete shooting motion)
        self.overall_feature_names = [
            'hip_knee_ankle_angle',      # 점프 힘 전달 및 하체 안정성
            'shoulder_hip_knee_angle',   # 상체 기울기 변화, 준비–발사 흐름
            'shoulder_elbow_wrist_angle', # 릴리즈 타이밍과 방향
            'elbow_shoulder_hip_angle',  # 릴리즈 높이 위치
            'ball_to_eye_vertical_distance', # 릴리즈 준비–출발 위치
            'elbow_to_eye_height'        # 릴리즈 시 손 높이 판단
        ]
        
        # Phase-specific features
        # Loading phase specific features (딥, 스쿼트, 준비 자세)
        self.loading_features = [
            'foot_rim_difference',       # 발의 림 방향 차이
            'ball_hip_vertical_distance', # 공-엉덩이 수직거리 (딥 측정)
            'hip_knee_ankle_angle',      # 엉덩이-무릎-발목 각도 (스쿼트 자세)
            'shoulder_hip_knee_angle',   # 어깨-엉덩이-무릎 각도 (상체 준비)
            'hip_shoulder_incline_angle', # 엉덩이-어깨 기울기 각도
            'ankle_width_vs_shoulder_width', # 발목 간격 vs 어깨 간격 (스탠스)
        ]
        
        # Rising phase specific features (윈드업, 상승, 셋포인트)
        self.rising_features = [
            'windup_trajectory_length',  # 윈드업 궤도 길이
            'dip_to_setpoint_time',      # 딥에서 셋포인트까지 시간
            'shoulder_elbow_wrist_angle', # 어깨-팔꿈치-손목 각도 (셋포인트 준비)
            'elbow_shoulder_hip_angle',   # 팔꿈치-어깨-엉덩이 각도 (상승 자세)
            'ball_to_eye_vertical_distance', # 공-눈 수직거리 (셋포인트 높이)
            'upward_movement_after_setpoint' # 셋포인트 후 상승 여부
        ]
        
        # Legacy: Combined features (backward compatibility)
        self.loading_rising_features = self.loading_features + self.rising_features
        
        self.release_features = [
            'shoulder_elbow_wrist_angle', # 어깨-팔꿈치-손목 각도
            'elbow_shoulder_hip_angle',   # 팔꿈치-어깨-엉덩이 각도
            'hip_knee_ankle_angle',       # 엉덩이-무릎-발목 각도
            'hip_shoulder_incline_angle', # 엉덩이-어깨 기울기 각도
            'ball_to_eye_distance',       # 공-눈 거리
            'elbow_to_eye_height'         # 팔꿈치-눈 높이
        ]
        
        self.follow_through_features = [
            'max_wrist_elbow_shoulder_angle', # 최대 손목-팔꿈치-어깨 각도
            'elbow_to_eye_height_at_max_angle', # 최대 각도 시 팔꿈치-눈 높이
            'time_to_max_angle_from_release',   # 릴리즈에서 최대 각도까지 시간
            'hip_knee_ankle_angle_at_max',      # 최대 각도 시 엉덩이-무릎-발목 각도
            'leg_kick_angle',                    # 레그 킥 각도
            'one_motion_vs_two_motion'          # 원모션 vs 투모션 판단
        ]
    
    def load_data(self, json_file: str) -> None:
        """
        Load shooting data from JSON file.
        
        Args:
            json_file: Path to the JSON file containing shooting data
        """
        with open(json_file, "r") as f:
            self.data = json.load(f)
    
    # ==================== COORDINATE-BASED DTW ====================
    
    def extract_coordinates_from_frame(self, frame: Dict, selected_hand: str = "right") -> List[float]:
        """
        Extract raw coordinates from a single frame.
        
        Args:
            frame: Frame data containing pose and ball information
            selected_hand: Which hand is being used for shooting
            
        Returns:
            List of coordinates [x1, y1, x2, y2, ...]
        """
        pose = frame.get("normalized_pose", {})
        ball = frame.get("normalized_ball", {})
        
        coordinates = []
        
        # Extract joint coordinates
        joints = [
            f"{selected_hand}_shoulder", f"{selected_hand}_elbow", f"{selected_hand}_wrist",
            f"{selected_hand}_hip", f"{selected_hand}_knee", f"{selected_hand}_ankle",
            f"{selected_hand}_eye"
        ]
        
        for joint in joints:
            joint_data = pose.get(joint, {})
            coordinates.extend([joint_data.get('x', 0.0), joint_data.get('y', 0.0)])
        
        # Add ball coordinates
        coordinates.extend([ball.get('x', 0.0), ball.get('y', 0.0)])
        
        return coordinates
    
    def extract_phase_coordinates(self, phase_frames: List[Dict], selected_hand: str = "right") -> np.ndarray:
        """
        Extract coordinates from a list of phase frames.
        
        Args:
            phase_frames: List of frame data for a phase
            selected_hand: Which hand is being used for shooting
            
        Returns:
            Numpy array of coordinates (frames x coordinates)
        """
        coordinates_list = []
        for frame in phase_frames:
            coords = self.extract_coordinates_from_frame(frame, selected_hand)
            coordinates_list.append(coords)
        
        return np.array(coordinates_list)
    
    def perform_coordinate_dtw(self, sequence1: np.ndarray, sequence2: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Perform coordinate-based DTW analysis.
        
        Args:
            sequence1: First sequence (frames x coordinates)
            sequence2: Second sequence (frames x coordinates)
            
        Returns:
            Tuple of (DTW distance, warping path)
        """
        distance, path = fastdtw(sequence1, sequence2, dist=euclidean)
        return distance, path
    
    # ==================== FEATURE-BASED DTW ====================
    
    def extract_overall_features_from_frame(self, frame: Dict, selected_hand: str = "right") -> Dict[str, float]:
        """
        Extract overall phase features from a single frame.
        
        Args:
            frame: Frame data containing pose and ball information
            selected_hand: Which hand is being used for shooting
            
        Returns:
            Dictionary of extracted features
        """
        pose = frame.get("normalized_pose", {})
        ball = frame.get("normalized_ball", {})
        
        features = {}
        
        # Extract joint positions
        shoulder = pose.get(f"{selected_hand}_shoulder", {})
        elbow = pose.get(f"{selected_hand}_elbow", {})
        wrist = pose.get(f"{selected_hand}_wrist", {})
        hip = pose.get(f"{selected_hand}_hip", {})
        knee = pose.get(f"{selected_hand}_knee", {})
        ankle = pose.get(f"{selected_hand}_ankle", {})
        eyes = pose.get(f"{selected_hand}_eye", {})
        
        # Calculate overall features
        if self._has_valid_coordinates(hip, knee, ankle):
            features['hip_knee_ankle_angle'] = self._calculate_angle(
                hip['x'], hip['y'], knee['x'], knee['y'], ankle['x'], ankle['y']
            )
        else:
            features['hip_knee_ankle_angle'] = 0.0
            
        if self._has_valid_coordinates(shoulder, hip, knee):
            features['shoulder_hip_knee_angle'] = self._calculate_angle(
                shoulder['x'], shoulder['y'], hip['x'], hip['y'], knee['x'], knee['y']
            )
        else:
            features['shoulder_hip_knee_angle'] = 0.0
            
        if self._has_valid_coordinates(shoulder, elbow, wrist):
            features['shoulder_elbow_wrist_angle'] = self._calculate_angle(
                shoulder['x'], shoulder['y'], elbow['x'], elbow['y'], wrist['x'], wrist['y']
            )
        else:
            features['shoulder_elbow_wrist_angle'] = 0.0
            
        if self._has_valid_coordinates(elbow, shoulder, hip):
            features['elbow_shoulder_hip_angle'] = self._calculate_angle(
                elbow['x'], elbow['y'], shoulder['x'], shoulder['y'], hip['x'], hip['y']
            )
        else:
            features['elbow_shoulder_hip_angle'] = 0.0
        
        # Calculate distances
        if self._has_valid_coordinates(ball, eyes):
            features['ball_to_eye_vertical_distance'] = self._calculate_vertical_distance(ball, eyes)
        else:
            features['ball_to_eye_vertical_distance'] = 0.0
            
        if self._has_valid_coordinates(elbow, eyes):
            features['elbow_to_eye_height'] = self._calculate_vertical_distance(elbow, eyes)
        else:
            features['elbow_to_eye_height'] = 0.0
        
        return features
    
    def extract_loading_features_from_frame(self, frame: Dict, selected_hand: str = "right") -> Dict[str, float]:
        """
        Extract loading phase specific features (딥, 스쿼트, 준비 자세).
        """
        pose = frame.get("normalized_pose", {})
        ball = frame.get("normalized_ball", {})
        
        features = {}
        
        # Extract joint positions
        shoulder = pose.get(f"{selected_hand}_shoulder", {})
        hip = pose.get(f"{selected_hand}_hip", {})
        knee = pose.get(f"{selected_hand}_knee", {})
        ankle = pose.get(f"{selected_hand}_ankle", {})
        left_ankle = pose.get("left_ankle", {})
        right_ankle = pose.get("right_ankle", {})
        left_shoulder = pose.get("left_shoulder", {})
        right_shoulder = pose.get("right_shoulder", {})
        
        # Loading specific features (딥, 스쿼트, 준비 자세)
        if hip and ball:
            features['ball_hip_vertical_distance'] = self._calculate_vertical_distance(ball, hip)
        else:
            features['ball_hip_vertical_distance'] = 0.0
            
        if hip and knee and ankle:
            features['hip_knee_ankle_angle'] = self._calculate_angle(
                hip['x'], hip['y'], knee['x'], knee['y'], ankle['x'], ankle['y']
            )
        else:
            features['hip_knee_ankle_angle'] = 0.0
            
        if shoulder and hip and knee:
            features['shoulder_hip_knee_angle'] = self._calculate_angle(
                shoulder['x'], shoulder['y'], hip['x'], hip['y'], knee['x'], knee['y']
            )
        else:
            features['shoulder_hip_knee_angle'] = 0.0
            
        if hip and shoulder:
            features['hip_shoulder_incline_angle'] = self._calculate_incline_angle(hip, shoulder)
        else:
            features['hip_shoulder_incline_angle'] = 0.0
        
        # Foot rim difference (simplified)
        features['foot_rim_difference'] = 0.0  # Would need rim position data
        
        # Ankle width vs shoulder width
        if left_ankle and right_ankle and left_shoulder and right_shoulder:
            ankle_width = abs(left_ankle['x'] - right_ankle['x'])
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
            features['ankle_width_vs_shoulder_width'] = ankle_width - shoulder_width
        else:
            features['ankle_width_vs_shoulder_width'] = 0.0
        
        return features

    def extract_rising_features_from_frame(self, frame: Dict, selected_hand: str = "right") -> Dict[str, float]:
        """
        Extract rising phase specific features (윈드업, 상승, 셋포인트).
        """
        pose = frame.get("normalized_pose", {})
        ball = frame.get("normalized_ball", {})
        
        features = {}
        
        # Extract joint positions
        shoulder = pose.get(f"{selected_hand}_shoulder", {})
        elbow = pose.get(f"{selected_hand}_elbow", {})
        wrist = pose.get(f"{selected_hand}_wrist", {})
        hip = pose.get(f"{selected_hand}_hip", {})
        eyes = pose.get(f"{selected_hand}_eye", {})
        
        # Rising specific features (윈드업, 상승, 셋포인트)
        if shoulder and elbow and wrist:
            features['shoulder_elbow_wrist_angle'] = self._calculate_angle(
                shoulder['x'], shoulder['y'], elbow['x'], elbow['y'], wrist['x'], wrist['y']
            )
        else:
            features['shoulder_elbow_wrist_angle'] = 0.0
            
        if elbow and shoulder and hip:
            features['elbow_shoulder_hip_angle'] = self._calculate_angle(
                elbow['x'], elbow['y'], shoulder['x'], shoulder['y'], hip['x'], hip['y']
            )
        else:
            features['elbow_shoulder_hip_angle'] = 0.0
        
        if ball and eyes:
            features['ball_to_eye_vertical_distance'] = self._calculate_vertical_distance(ball, eyes)
        else:
            features['ball_to_eye_vertical_distance'] = 0.0
        
        # Windup trajectory (simplified)
        features['windup_trajectory_length'] = 0.0  # Would need trajectory calculation
        
        # Time features (simplified)
        features['dip_to_setpoint_time'] = 0.0  # Would need timing data
        
        # Upward movement (simplified)
        features['upward_movement_after_setpoint'] = 0.0  # Would need movement analysis
        
        return features

    def extract_loading_rising_features_from_frame(self, frame: Dict, selected_hand: str = "right") -> Dict[str, float]:
        """
        Extract loading & rising phase specific features (legacy method for backward compatibility).
        """
        # Combine loading and rising features for backward compatibility
        loading_features = self.extract_loading_features_from_frame(frame, selected_hand)
        rising_features = self.extract_rising_features_from_frame(frame, selected_hand)
        
        # Merge both feature sets
        combined_features = {}
        combined_features.update(loading_features)
        combined_features.update(rising_features)
        
        return combined_features
    
    def extract_release_features_from_frame(self, frame: Dict, selected_hand: str = "right") -> Dict[str, float]:
        """
        Extract release phase specific features.
        """
        pose = frame.get("normalized_pose", {})
        ball = frame.get("normalized_ball", {})
        
        features = {}
        
        # Extract joint positions
        shoulder = pose.get(f"{selected_hand}_shoulder", {})
        elbow = pose.get(f"{selected_hand}_elbow", {})
        wrist = pose.get(f"{selected_hand}_wrist", {})
        hip = pose.get(f"{selected_hand}_hip", {})
        knee = pose.get(f"{selected_hand}_knee", {})
        ankle = pose.get(f"{selected_hand}_ankle", {})
        eyes = pose.get(f"{selected_hand}_eye", {})
        
        # Release specific features
        if shoulder and elbow and wrist:
            features['shoulder_elbow_wrist_angle'] = self._calculate_angle(
                shoulder['x'], shoulder['y'], elbow['x'], elbow['y'], wrist['x'], wrist['y']
            )
        else:
            features['shoulder_elbow_wrist_angle'] = 0.0
            
        if elbow and shoulder and hip:
            features['elbow_shoulder_hip_angle'] = self._calculate_angle(
                elbow['x'], elbow['y'], shoulder['x'], shoulder['y'], hip['x'], hip['y']
            )
        else:
            features['elbow_shoulder_hip_angle'] = 0.0
            
        if hip and knee and ankle:
            features['hip_knee_ankle_angle'] = self._calculate_angle(
                hip['x'], hip['y'], knee['x'], knee['y'], ankle['x'], ankle['y']
            )
        else:
            features['hip_knee_ankle_angle'] = 0.0
            
        if hip and shoulder:
            features['hip_shoulder_incline_angle'] = self._calculate_incline_angle(hip, shoulder)
        else:
            features['hip_shoulder_incline_angle'] = 0.0
        
        if ball and eyes:
            features['ball_to_eye_distance'] = self._calculate_ball_distance(ball, eyes)
        else:
            features['ball_to_eye_distance'] = 0.0
            
        if elbow and eyes:
            features['elbow_to_eye_height'] = self._calculate_vertical_distance(elbow, eyes)
        else:
            features['elbow_to_eye_height'] = 0.0
        
        return features
    
    def extract_follow_through_features_from_frame(self, frame: Dict, selected_hand: str = "right") -> Dict[str, float]:
        """
        Extract follow-through phase specific features.
        """
        pose = frame.get("normalized_pose", {})
        
        features = {}
        
        # Extract joint positions
        shoulder = pose.get(f"{selected_hand}_shoulder", {})
        elbow = pose.get(f"{selected_hand}_elbow", {})
        wrist = pose.get(f"{selected_hand}_wrist", {})
        hip = pose.get(f"{selected_hand}_hip", {})
        knee = pose.get(f"{selected_hand}_knee", {})
        ankle = pose.get(f"{selected_hand}_ankle", {})
        eyes = pose.get(f"{selected_hand}_eye", {})
        left_knee = pose.get("left_knee", {})
        right_knee = pose.get("right_knee", {})
        
        # Follow-through specific features
        if wrist and elbow and shoulder:
            features['max_wrist_elbow_shoulder_angle'] = self._calculate_angle(
                wrist['x'], wrist['y'], elbow['x'], elbow['y'], shoulder['x'], shoulder['y']
            )
        else:
            features['max_wrist_elbow_shoulder_angle'] = 0.0
            
        if elbow and eyes:
            features['elbow_to_eye_height_at_max_angle'] = self._calculate_vertical_distance(elbow, eyes)
        else:
            features['elbow_to_eye_height_at_max_angle'] = 0.0
        
        # Time features (simplified)
        features['time_to_max_angle_from_release'] = 0.0  # Would need timing data
        
        if hip and knee and ankle:
            features['hip_knee_ankle_angle_at_max'] = self._calculate_angle(
                hip['x'], hip['y'], knee['x'], knee['y'], ankle['x'], ankle['y']
            )
        else:
            features['hip_knee_ankle_angle_at_max'] = 0.0
        
        # Leg kick angle
        if right_knee and hip and left_knee:
            features['leg_kick_angle'] = self._calculate_angle(
                right_knee['x'], right_knee['y'], hip['x'], hip['y'], left_knee['x'], left_knee['y']
            )
        else:
            features['leg_kick_angle'] = 0.0
        
        # One motion vs two motion (simplified)
        features['one_motion_vs_two_motion'] = 0.0  # Would need movement analysis
        
        return features
    
    def extract_phase_features(self, phase_frames: List[Dict], feature_type: str, selected_hand: str = "right") -> np.ndarray:
        """
        Extract features from a list of phase frames based on feature type.
        
        Args:
            phase_frames: List of frame data for a phase
            feature_type: Type of features to extract ("overall", "loading", "rising", "loading_rising", "release", "follow_through")
            selected_hand: Which hand is being used for shooting
            
        Returns:
            Numpy array of features (frames x features)
        """
        features_list = []
        
        for frame in phase_frames:
            try:
                if feature_type == "overall":
                    features = self.extract_overall_features_from_frame(frame, selected_hand)
                    feature_names = self.overall_feature_names
                elif feature_type == "loading":
                    features = self.extract_loading_features_from_frame(frame, selected_hand)
                    feature_names = self.loading_features
                elif feature_type == "rising":
                    features = self.extract_rising_features_from_frame(frame, selected_hand)
                    feature_names = self.rising_features
                elif feature_type == "loading_rising":
                    features = self.extract_loading_rising_features_from_frame(frame, selected_hand)
                    feature_names = self.loading_rising_features
                elif feature_type == "release":
                    features = self.extract_release_features_from_frame(frame, selected_hand)
                    feature_names = self.release_features
                elif feature_type == "follow_through":
                    features = self.extract_follow_through_features_from_frame(frame, selected_hand)
                    feature_names = self.follow_through_features
                else:
                    raise ValueError(f"Unknown feature type: {feature_type}")
                
                features_list.append([features[name] for name in feature_names])
            except Exception as e:
                print(f"⚠️  Error extracting features from frame: {e}")
                # Create zero features for failed frame
                if feature_type == "overall":
                    feature_names = self.overall_feature_names
                elif feature_type == "loading":
                    feature_names = self.loading_features
                elif feature_type == "rising":
                    feature_names = self.rising_features
                elif feature_type == "loading_rising":
                    feature_names = self.loading_rising_features
                elif feature_type == "release":
                    feature_names = self.release_features
                elif feature_type == "follow_through":
                    feature_names = self.follow_through_features
                else:
                    feature_names = []
                
                features_list.append([0.0] * len(feature_names))
        
        return np.array(features_list)
    
    def perform_feature_dtw(self, sequence1: np.ndarray, sequence2: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Perform feature-based DTW analysis.
        
        Args:
            sequence1: First sequence (frames x features)
            sequence2: Second sequence (frames x features)
            
        Returns:
            Tuple of (DTW distance, warping path)
        """
        distance, path = fastdtw(sequence1, sequence2, dist=euclidean)
        return distance, path
    
    # ==================== PHASE FRAME EXTRACTION ====================
    
    def get_phase_frames(self, phase_name: str) -> List[Dict]:
        """
        Get frames for a specific phase.
        
        Args:
            phase_name: Name of the phase ("Loading", "Rising", "Release", "Follow-through", "Loading-Rising")
            
        Returns:
            List of frames for the specified phase
        """
        if not self.data or "frames" not in self.data:
            return []
        
        phase_frames = []
        for frame in self.data["frames"]:
            if frame.get("phase") == phase_name:
                phase_frames.append(frame)
        
        return phase_frames
    
    def get_combined_phase_frames(self, phase_names: List[str]) -> List[Dict]:
        """
        Get frames for multiple phases combined.
        
        Args:
            phase_names: List of phase names to combine
            
        Returns:
            List of frames for the combined phases
        """
        if not self.data or "frames" not in self.data:
            return []
        
        combined_frames = []
        for frame in self.data["frames"]:
            if frame.get("phase") in phase_names:
                combined_frames.append(frame)
        
        return combined_frames
    
    # ==================== ANALYSIS METHODS ====================
    
    def analyze_overall_phases_coordinate(self, json_file1: str, json_file2: str, 
                                        selected_hand: str = "right") -> Dict:
        """
        Perform coordinate-based DTW analysis on overall shooting phases.
        """
        # Load data
        self.load_data(json_file1)
        frames1 = self.data.get("frames", [])
        
        self.load_data(json_file2)
        frames2 = self.data.get("frames", [])
        
        # Extract coordinates for all frames
        coords1 = self.extract_phase_coordinates(frames1, selected_hand)
        coords2 = self.extract_phase_coordinates(frames2, selected_hand)
        
        # Perform DTW
        distance, path = self.perform_coordinate_dtw(coords1, coords2)
        
        return {
            'method': 'coordinate',
            'dtw_distance': distance,
            'warping_path': path,
            'sequence1_length': len(coords1),
            'sequence2_length': len(coords2)
        }
    
    def analyze_overall_phases_feature(self, json_file1: str, json_file2: str, 
                                     selected_hand: str = "right") -> Dict:
        """
        Perform feature-based DTW analysis on overall shooting phases.
        """
        # Load data
        self.load_data(json_file1)
        frames1 = self.data.get("frames", [])
        
        self.load_data(json_file2)
        frames2 = self.data.get("frames", [])
        
        # Extract features for all frames
        features1 = self.extract_phase_features(frames1, "overall", selected_hand)
        features2 = self.extract_phase_features(frames2, "overall", selected_hand)
        
        # Perform DTW
        distance, path = self.perform_feature_dtw(features1, features2)
        
        return {
            'method': 'feature',
            'dtw_distance': distance,
            'warping_path': path,
            'sequence1_length': len(features1),
            'sequence2_length': len(features2),
            'feature_names': self.overall_feature_names
        }
    
    def analyze_loading_phases(self, json_file1: str, json_file2: str,
                              selected_hand: str = "right") -> Dict:
        """
        Analyze Loading phases using feature-based DTW.
        Includes Loading-Rising frames as part of Loading phase.
        
        Args:
            json_file1: Path to first JSON file
            json_file2: Path to second JSON file
            selected_hand: Which hand is being used for shooting
            
        Returns:
            Dictionary containing DTW analysis results
        """
        # Load data
        self.load_data(json_file1)
        phase_frames1 = self.get_combined_phase_frames(["Loading", "Loading-Rising"])
        
        self.load_data(json_file2)
        phase_frames2 = self.get_combined_phase_frames(["Loading", "Loading-Rising"])
        
        if not phase_frames1 or not phase_frames2:
            return {
                "error": "No Loading/Loading-Rising phases found in one or both files",
                "distance": float('inf'),
                "path": []
            }
        
        # Extract loading-specific features (Loading + Loading-Rising frames but with loading features only)
        features1 = self.extract_phase_features(phase_frames1, "loading", selected_hand)
        features2 = self.extract_phase_features(phase_frames2, "loading", selected_hand)
        
        # Perform DTW
        distance, path = self.perform_feature_dtw(features1, features2)
        
        return {
            "phase": "loading",
            "method": "feature",
            "dtw_distance": distance,
            "warping_path": path,
            "sequence1_length": len(features1),
            "sequence2_length": len(features2),
            "feature_names": self.loading_features,
            "frames1": len(phase_frames1),
            "frames2": len(phase_frames2)
        }
    
    def analyze_rising_phases(self, json_file1: str, json_file2: str,
                             selected_hand: str = "right") -> Dict:
        """
        Analyze Rising phases using feature-based DTW.
        Includes Loading-Rising frames as part of Rising phase.
        
        Args:
            json_file1: Path to first JSON file
            json_file2: Path to second JSON file
            selected_hand: Which hand is being used for shooting
            
        Returns:
            Dictionary containing DTW analysis results
        """
        # Load data
        self.load_data(json_file1)
        phase_frames1 = self.get_combined_phase_frames(["Rising", "Loading-Rising"])
        
        self.load_data(json_file2)
        phase_frames2 = self.get_combined_phase_frames(["Rising", "Loading-Rising"])
        
        if not phase_frames1 or not phase_frames2:
            return {
                "error": "No Rising/Loading-Rising phases found in one or both files",
                "distance": float('inf'),
                "path": []
            }
        
        # Extract rising-specific features (Rising + Loading-Rising frames but with rising features only)
        features1 = self.extract_phase_features(phase_frames1, "rising", selected_hand)
        features2 = self.extract_phase_features(phase_frames2, "rising", selected_hand)
        
        # Perform DTW
        distance, path = self.perform_feature_dtw(features1, features2)
        
        return {
            "phase": "rising", 
            "method": "feature",
            "dtw_distance": distance,
            "warping_path": path,
            "sequence1_length": len(features1),
            "sequence2_length": len(features2),
            "feature_names": self.rising_features,
            "frames1": len(phase_frames1),
            "frames2": len(phase_frames2)
        }
    
    def analyze_release_phases(self, json_file1: str, json_file2: str,
                              selected_hand: str = "right") -> Dict:
        """
        Perform DTW analysis on release phases.
        """
        # Load data and get release frames
        self.load_data(json_file1)
        frames1 = self.get_phase_frames("Release")
        
        self.load_data(json_file2)
        frames2 = self.get_phase_frames("Release")
        
        if not frames1 or not frames2:
            return {
                "error": "No Release phases found in one or both files",
                "phase": "release",
                "method": "feature",
                "dtw_distance": float('inf'),
                "warping_path": [],
                "sequence1_length": 0,
                "sequence2_length": 0,
                "frames1": len(frames1) if frames1 else 0,
                "frames2": len(frames2) if frames2 else 0,
                "feature_names": self.release_features
            }
        
        # Extract features
        features1 = self.extract_phase_features(frames1, "release", selected_hand)
        features2 = self.extract_phase_features(frames2, "release", selected_hand)
        
        # Perform DTW
        distance, path = self.perform_feature_dtw(features1, features2)
        
        return {
            'phase': 'release',
            'method': 'feature',
            'dtw_distance': distance,
            'warping_path': path,
            'sequence1_length': len(features1),
            'sequence2_length': len(features2),
            'frames1': len(frames1),
            'frames2': len(frames2),
            'feature_names': self.release_features
        }
    
    def analyze_follow_through_phases(self, json_file1: str, json_file2: str,
                                     selected_hand: str = "right") -> Dict:
        """
        Perform DTW analysis on follow-through phases.
        """
        # Load data and get follow-through frames
        self.load_data(json_file1)
        frames1 = self.get_phase_frames("Follow-through")
        
        self.load_data(json_file2)
        frames2 = self.get_phase_frames("Follow-through")
        
        if not frames1 or not frames2:
            return {
                "error": "No Follow-through phases found in one or both files",
                "phase": "follow_through",
                "method": "feature",
                "dtw_distance": float('inf'),
                "warping_path": [],
                "sequence1_length": 0,
                "sequence2_length": 0,
                "frames1": len(frames1) if frames1 else 0,
                "frames2": len(frames2) if frames2 else 0,
                "feature_names": self.follow_through_features
            }
        
        # Extract features
        features1 = self.extract_phase_features(frames1, "follow_through", selected_hand)
        features2 = self.extract_phase_features(frames2, "follow_through", selected_hand)
        
        # Perform DTW
        distance, path = self.perform_feature_dtw(features1, features2)
        
        return {
            'phase': 'follow_through',
            'method': 'feature',
            'dtw_distance': distance,
            'warping_path': path,
            'sequence1_length': len(features1),
            'sequence2_length': len(features2),
            'frames1': len(frames1),
            'frames2': len(frames2),
            'feature_names': self.follow_through_features
        }
    
    def compare_multiple_shots(self, json_files: List[str], 
                              selected_hand: str = "right") -> Dict:
        """
        Compare multiple shots using both coordinate and feature-based DTW analysis.
        """
        results = {
            'coordinate_overall': [],
            'feature_overall': [],
            'loading': [],
            'rising': [],
            'release': [],
            'follow_through': []
        }
        
        for i in range(len(json_files)):
            for j in range(i + 1, len(json_files)):
                # Coordinate-based overall analysis
                coord_result = self.analyze_overall_phases_coordinate(
                    json_files[i], json_files[j], selected_hand
                )
                coord_result['file1'] = json_files[i]
                coord_result['file2'] = json_files[j]
                results['coordinate_overall'].append(coord_result)
                
                # Feature-based overall analysis
                feature_result = self.analyze_overall_phases_feature(
                    json_files[i], json_files[j], selected_hand
                )
                feature_result['file1'] = json_files[i]
                feature_result['file2'] = json_files[j]
                results['feature_overall'].append(feature_result)
                
                # Phase-specific analysis
                loading_result = self.analyze_loading_phases(
                    json_files[i], json_files[j], selected_hand
                )
                loading_result['file1'] = json_files[i]
                loading_result['file2'] = json_files[j]
                results['loading'].append(loading_result)
                
                rising_result = self.analyze_rising_phases(
                    json_files[i], json_files[j], selected_hand
                )
                rising_result['file1'] = json_files[i]
                rising_result['file2'] = json_files[j]
                results['rising'].append(rising_result)
                
                release_result = self.analyze_release_phases(
                    json_files[i], json_files[j], selected_hand
                )
                release_result['file1'] = json_files[i]
                release_result['file2'] = json_files[j]
                results['release'].append(release_result)
                
                ft_result = self.analyze_follow_through_phases(
                    json_files[i], json_files[j], selected_hand
                )
                ft_result['file1'] = json_files[i]
                ft_result['file2'] = json_files[j]
                results['follow_through'].append(ft_result)
        
        return results
    
    # Helper methods for calculations
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
    
    def _calculate_incline_angle(self, hip: Dict, shoulder: Dict) -> float:
        """Calculate incline angle between hip and shoulder."""
        if not hip or not shoulder:
            return 0.0
        
        dx = shoulder['x'] - hip['x']
        dy = shoulder['y'] - hip['y']
        angle = np.degrees(np.arctan2(dy, dx))
        return angle
    
    def _calculate_ball_distance(self, ball: Dict, eyes: Dict) -> float:
        """Calculate distance between ball and eyes."""
        if not ball or not eyes:
            return 0.0
        
        dx = ball['x'] - eyes['x']
        dy = ball['y'] - eyes['y']
        return np.sqrt(dx**2 + dy**2)
    
    def _calculate_vertical_distance(self, point1: Dict, point2: Dict) -> float:
        """Calculate vertical distance between two points."""
        if not point1 or not point2:
            return 0.0
        
        return abs(point1['y'] - point2['y'])
    
    def _calculate_horizontal_distance(self, point1: Dict, point2: Dict) -> float:
        """Calculate horizontal distance between two points."""
        if not point1 or not point2:
            return 0.0
        
        return abs(point1['x'] - point2['x'])