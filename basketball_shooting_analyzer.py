# -*- coding: utf-8 -*-
"""
Basketball shooting motion analysis pipeline
Integrate pose data and ball data to analyze shooting movements and visualize
"""

import cv2
import numpy as np
import json
import os
import glob
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import math

class BasketballShootingAnalyzer:
    def __init__(self):
        """Initialize basketball shooting analyzer"""
        self.references_dir = "data"
        self.video_dir = os.path.join(self.references_dir, "video")
        self.extracted_data_dir = os.path.join(self.references_dir, "extracted_data")
        self.results_dir = os.path.join(self.references_dir, "results")
        self.visualized_video_dir = os.path.join(self.references_dir, "visualized_video")
        
        # Create directories
        for dir_path in [self.video_dir, self.extracted_data_dir, self.results_dir, self.visualized_video_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Data storage
        self.pose_data = []
        self.ball_data = []
        self.normalized_data = []
        self.phases = []
        self.phase_statistics = {}
        self.selected_video = None
        self.available_videos = []

    def list_available_videos(self) -> List[str]:
        """Return a list of available video files"""
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        videos = []
        
        for ext in video_extensions:
            pattern = os.path.join(self.video_dir, ext)
            videos.extend(glob.glob(pattern))
        
        return sorted(videos)
    
    def prompt_video_selection(self) -> Optional[str]:
        """Prompt user to select a video"""
        self.available_videos = self.list_available_videos()
        
        if not self.available_videos:
            print("❌ No video files found in data/video folder.")
            return None
        
        print("\n🎬 STEP 0: Select video")
        print("=" * 50)
        print("Select the video to analyze:")
        
        for i, video in enumerate(self.available_videos, 1):
            print(f"[{i}] {video}")
        
        while True:
            try:
                choice = input("\nEnter the number or directly enter the file name: ").strip()
                
                # Select by number
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(self.available_videos):
                        return self.available_videos[idx]
                    else:
                        print("❌ Invalid number.")
                        continue
                
                # Select by file name
                if choice in self.available_videos:
                    return choice
                
                # If only the file name is entered (without extension)
                full_path = os.path.join(self.video_dir, choice)
                if full_path in self.available_videos:
                    return full_path
                
                print("❌ Invalid selection. Please try again.")
                
            except KeyboardInterrupt:
                print("\n❌ Analysis canceled.")
                return None
    
    def load_associated_data(self, video_path: str, overwrite_mode: bool = False) -> bool:
        """Load original pose/ball data associated with the video"""
        print(f"\n📂 STEP 1: Load original data")
        print("=" * 50)
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Original data file paths
        pose_original_json = os.path.join(self.extracted_data_dir, f"{base_name}_pose_original.json")
        ball_original_json = os.path.join(self.extracted_data_dir, f"{base_name}_ball_original.json")
        
        # If existing files exist and overwrite mode is not selected, check
        if not overwrite_mode and (os.path.exists(pose_original_json) or os.path.exists(ball_original_json)):
            print(f"\n⚠️ Existing original extraction data found:")
            if os.path.exists(pose_original_json):
                print(f"  - Pose data: {os.path.basename(pose_original_json)}")
            if os.path.exists(ball_original_json):
                print(f"  - Ball data: {os.path.basename(ball_original_json)}")
            choice = input("Overwrite and extract new data? (y/n): ").strip().lower()
            if choice != 'y':
                print("Using existing original extraction data.")
            else:
                print("Overwrite existing data and extract new data.")
                overwrite_mode = True
        
        # Load original pose data
        pose_files = glob.glob(os.path.join(self.extracted_data_dir, f"{base_name}_pose_original*.json"))
        if pose_files:
            pose_file = pose_files[0]  # Use the first file
            try:
                with open(pose_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "pose_data" in data:
                        self.pose_data = data["pose_data"]
                    else:
                        self.pose_data = data
                print(f"✅ Original pose data loaded: {os.path.basename(pose_file)}")
            except Exception as e:
                print(f"❌ Failed to load original pose data: {e}")
                return False
        else:
            print(f"❌ Original pose data file not found: {base_name}_pose_original*.json")
            return False
        
        # Load original ball data
        ball_files = glob.glob(os.path.join(self.extracted_data_dir, f"{base_name}_ball_original*.json"))
        if ball_files:
            ball_file = ball_files[0]  # Use the first file
            try:
                with open(ball_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "ball_trajectory" in data:
                        self.ball_data = data["ball_trajectory"]
                    else:
                        self.ball_data = data
                print(f"✅ Original ball data loaded: {os.path.basename(ball_file)}")
            except Exception as e:
                print(f"❌ Failed to load original ball data: {e}")
                return False
        else:
            print(f"❌ Original ball data file not found: {base_name}_ball_original*.json")
            return False
        
        return True
    
    def normalize_pose_data(self, video_path: Optional[str] = None):
        """Normalize pose data (hip_center based, use average ball size) and save separately"""
        print(f"\n🔄 STEP 2: Normalize data and save")
        print("=" * 50)
        
        if not self.pose_data:
            print("❌ Pose data not found.")
            return
        
        # Use selected_video if video_path is None
        if video_path is None:
            video_path = self.selected_video
        
        # Calculate average ball radius from detected frames
        ball_radii = []
        for frame_data in self.ball_data:
            # Check if frame_data is a dictionary
            if isinstance(frame_data, dict):
                for ball in frame_data.get('ball_detections', []):
                    if isinstance(ball, dict):
                        radius = (ball.get('width', 0) + ball.get('height', 0)) / 4
                        ball_radii.append(radius)
            else:
                print(f"⚠️ Unexpected ball_data structure: {type(frame_data)}")
        
        mean_ball_radius = np.mean(ball_radii) if ball_radii else 1.0
        print(f"Average ball radius: {mean_ball_radius:.3f} (Detected frames: {len(ball_radii)} frames)")
        
        self.normalized_data = []
        previous_hip_center = None
        consecutive_missing_hip = 0
        max_consecutive_missing = 5  # Warn if more than 5 consecutive frames are missing
        
        for i, frame_data in enumerate(self.pose_data):
            pose = frame_data.get('pose', {})
            
            # Calculate hip_center (use original coordinates) - added exception handling
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            
            # Validate hip_center
            hip_center_valid = False
            hip_center_x = 0
            hip_center_y = 0
            
            if (isinstance(left_hip, dict) and 'x' in left_hip and 'y' in left_hip and
                isinstance(right_hip, dict) and 'x' in right_hip and 'y' in right_hip):
                # Both hips exist
                hip_center_x = (left_hip['x'] + right_hip['x']) / 2
                hip_center_y = (left_hip['y'] + right_hip['y']) / 2
                hip_center_valid = True
                consecutive_missing_hip = 0
            elif (isinstance(left_hip, dict) and 'x' in left_hip and 'y' in left_hip):
                # Left hip exists
                hip_center_x = left_hip['x']
                hip_center_y = left_hip['y']
                hip_center_valid = True
                consecutive_missing_hip = 0
            elif (isinstance(right_hip, dict) and 'x' in right_hip and 'y' in right_hip):
                # Right hip exists
                hip_center_x = right_hip['x']
                hip_center_y = right_hip['y']
                hip_center_valid = True
                consecutive_missing_hip = 0
            else:
                # hip_center does not exist
                consecutive_missing_hip += 1
                if previous_hip_center is not None:
                    # Use previous frame value
                    hip_center_x, hip_center_y = previous_hip_center
                    hip_center_valid = True
                    print(f"⚠️ Frame {i}: hip_center missing, using previous frame value")
                else:
                    # First frame and hip_center does not exist
                    hip_center_x = 0
                    hip_center_y = 0
                    hip_center_valid = False
                    print(f"⚠️ Frame {i}: hip_center missing, using default value")
            
            # Warn for consecutive missing frames
            if consecutive_missing_hip >= max_consecutive_missing:
                print(f"⚠️ Warning: {consecutive_missing_hip} consecutive frames missing hip_center (Frames {i-max_consecutive_missing+1}~{i})")
            
            # Save current hip_center as previous value
            if hip_center_valid:
                previous_hip_center = (hip_center_x, hip_center_y)
            
            # Use average ball radius for scaling
            current_ball_radius = mean_ball_radius
            
            # Calculate normalized pose (do not add missing keypoints)
            normalized_pose = {}
            for key, kp in pose.items():
                if isinstance(kp, dict) and 'x' in kp and 'y' in kp:
                    # Center around hip_center
                    norm_x = (kp['x'] - hip_center_x) / current_ball_radius
                    norm_y = (kp['y'] - hip_center_y) / current_ball_radius
                    
                    normalized_pose[key] = {
                        'x': norm_x,
                        'y': norm_y,
                        'confidence': kp.get('confidence', 0)
                    }
                # Missing keypoints are not added to normalized_pose (automatically excluded)
            
            # Normalize ball position (safely)
            normalized_ball = {}
            ball_detected = False
            
            if i < len(self.ball_data):
                ball_frame_data = self.ball_data[i]
                if isinstance(ball_frame_data, dict) and ball_frame_data.get('ball_detections'):
                    ball_detections = ball_frame_data['ball_detections']
                    if ball_detections and isinstance(ball_detections[0], dict):
                        ball = ball_detections[0]
                        ball_x = ball.get('center_x', 0)
                        ball_y = ball.get('center_y', 0)
                        
                        normalized_ball = {
                            'center_x': (ball_x - hip_center_x) / current_ball_radius,
                            'center_y': (ball_y - hip_center_y) / current_ball_radius,
                            'width': ball.get('width', 0.1) / current_ball_radius,
                            'height': ball.get('height', 0.1) / current_ball_radius
                        }
                        ball_detected = True
            
            normalized_frame = {
                'frame_index': i,
                'normalized_pose': normalized_pose,
                'normalized_ball': normalized_ball,
                'original_hip_center': [hip_center_x, hip_center_y],
                'scaling_factor': current_ball_radius,
                'ball_detected': ball_detected,
                'hip_center_valid': hip_center_valid,
                'consecutive_missing_hip': consecutive_missing_hip
            }
            
            self.normalized_data.append(normalized_frame)
        
        # Print statistics
        detected_frames = sum(1 for frame in self.normalized_data if frame['ball_detected'])
        total_frames = len(self.normalized_data)
        valid_hip_frames = sum(1 for frame in self.normalized_data if frame['hip_center_valid'])
        
        print(f"✅ Normalization completed: {len(self.normalized_data)} frames")
        print(f"Detected ball frames: {detected_frames}/{total_frames} ({detected_frames/total_frames*100:.1f}%)")
        print(f"Valid hip_center frames: {valid_hip_frames}/{total_frames} ({valid_hip_frames/total_frames*100:.1f}%)")
        
        if consecutive_missing_hip > 0:
            print(f"⚠️ Last {consecutive_missing_hip} frames missing hip_center")
        
        # Save normalized data as separate JSON file
        self._save_normalized_data(video_path)

    def convert_numpy_types(self, obj):
        """Convert numpy types to Python basic types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj

    def _save_normalized_data(self, video_path: Optional[str]):
        """Save normalized data as separate JSON file"""
        if video_path is None:
            print("❌ video_path not provided, cannot save normalized data")
            return
            
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Save normalized pose data
        pose_normalized_file = os.path.join(self.extracted_data_dir, f"{base_name}_pose_normalized.json")
        pose_data_to_save = {
            "metadata": {
                "total_frames": len(self.normalized_data),
                "normalization_time": datetime.now().isoformat(),
                "coordinate_system": "normalized_hip_center_based",
                "scaling_factor": self.normalized_data[0]['scaling_factor'] if self.normalized_data else 1.0
            },
            "normalized_pose_data": self.normalized_data
        }
        
        # Convert numpy types to Python basic types
        pose_data_to_save = self.convert_numpy_types(pose_data_to_save)
        
        try:
            with open(pose_normalized_file, 'w', encoding='utf-8') as f:
                json.dump(pose_data_to_save, f, indent=2, ensure_ascii=False)
            print(f"✅ Normalized pose data saved: {os.path.basename(pose_normalized_file)}")
        except Exception as e:
            print(f"❌ Failed to save normalized pose data: {e}")
        
        # Save normalized ball data
        ball_normalized_file = os.path.join(self.extracted_data_dir, f"{base_name}_ball_normalized.json")
        ball_data_to_save = {
            "metadata": {
                "total_frames": len(self.normalized_data),
                "normalization_time": datetime.now().isoformat(),
                "coordinate_system": "normalized_hip_center_based",
                "scaling_factor": self.normalized_data[0]['scaling_factor'] if self.normalized_data else 1.0
            },
            "normalized_ball_data": [frame['normalized_ball'] for frame in self.normalized_data]
        }
        
        # Convert numpy types to Python basic types
        ball_data_to_save = self.convert_numpy_types(ball_data_to_save)
        
        try:
            with open(ball_normalized_file, 'w', encoding='utf-8') as f:
                json.dump(ball_data_to_save, f, indent=2, ensure_ascii=False)
            print(f"✅ Normalized ball data saved: {os.path.basename(ball_normalized_file)}")
        except Exception as e:
            print(f"❌ Failed to save normalized ball data: {e}")
    
    def segment_shooting_phases(self):
        """Segment shooting movement into 6 steps (quick transition + short noise filtering)"""
        print(f"\n📐 STEP 3: Segment shooting phases")
        print("=" * 50)
        
        if not self.normalized_data:
            print("❌ Normalized data not found.")
            return
        
        self.phases = []
        current_phase = "Set-up"
        phase_start_frame = 0
        
        # Setup for noise filtering
        min_phase_duration = 2  # Must last at least 2 frames
        noise_threshold = 4  # Changes of 4 frames or less are considered noise
        
        for i, frame_data in enumerate(self.normalized_data):
            pose = frame_data['normalized_pose']
            
            # Extract necessary keypoints
            left_knee = pose.get('left_knee', {'y': 0})
            right_knee = pose.get('right_knee', {'y': 0})
            left_wrist = pose.get('left_wrist', {'y': 0})
            right_wrist = pose.get('right_wrist', {'y': 0})
            left_hip = pose.get('left_hip', {'y': 0})
            right_hip = pose.get('right_hip', {'y': 0})
            left_ankle = pose.get('left_ankle', {'y': 0})
            right_ankle = pose.get('right_ankle', {'y': 0})
            
            # Calculate average values
            knee_y = (left_knee['y'] + right_knee['y']) / 2
            wrist_y = (left_wrist['y'] + right_wrist['y']) / 2
            hip_y = (left_hip['y'] + right_hip['y']) / 2
            ankle_y = (left_ankle['y'] + right_ankle['y']) / 2
            
            # Calculate change amounts compared to previous frames
            if i > 0:
                prev_frame = self.normalized_data[i-1]
                prev_pose = prev_frame['normalized_pose']
                
                prev_knee_y = (prev_pose.get('left_knee', {'y': 0})['y'] + 
                              prev_pose.get('right_knee', {'y': 0})['y']) / 2
                prev_wrist_y = (prev_pose.get('left_wrist', {'y': 0})['y'] + 
                               prev_pose.get('right_wrist', {'y': 0})['y']) / 2
                prev_hip_y = (prev_pose.get('left_hip', {'y': 0})['y'] + 
                             prev_pose.get('right_hip', {'y': 0})['y']) / 2
                
                d_knee_y = knee_y - prev_knee_y
                d_wrist_y = wrist_y - prev_wrist_y
                d_hip_y = hip_y - prev_hip_y
            else:
                d_knee_y = d_wrist_y = d_hip_y = 0
            
            # Check if current phase transitions to next phase
            next_phase = self._check_phase_transition(current_phase, i, knee_y, wrist_y, hip_y, ankle_y, 
                                                    d_knee_y, d_wrist_y, d_hip_y)
            
            # Check minimum phase duration (allow quick transition)
            if next_phase != current_phase and (i - phase_start_frame) >= min_phase_duration:
                # Noise filtering based on trend
                if self._is_trend_based_transition(i, current_phase, next_phase, noise_threshold):
                    current_phase = next_phase
                    phase_start_frame = i
                    print(f"Frame {i}: {current_phase} phase started")
                else:
                    # Consider as noise and keep current phase
                    if i % 10 == 0:
                        print(f"Frame {i}: Noise detected, {current_phase} maintained")
            
            self.phases.append(current_phase)
        
        # Final trend-based organization
        self._finalize_phases_by_trend(noise_threshold)
        
        # Print phase-by-frame statistics
        phase_counts = {}
        for phase in self.phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        print("\nPhase-by-frame count:")
        for phase, count in phase_counts.items():
            print(f"  {phase}: {count} frames")
    
    def _is_trend_based_transition(self, frame_idx: int, current_phase: str, next_phase: str, noise_threshold: int) -> bool:
        """Trend-based transition determination (always returns True)"""
        return True
    
    def _finalize_phases_by_trend(self, noise_threshold: int):
        """Final trend-based organization (does nothing)"""
        return
    
    def _check_phase_transition(self, current_phase: str, frame_idx: int, knee_y: float, 
                               wrist_y: float, hip_y: float, ankle_y: float,
                               d_knee_y: float, d_wrist_y: float, d_hip_y: float) -> str:
        """Check phase transition conditions (priority-based, excluding unrecognized parts)"""
        
        # Get ball data
        ball_info = None
        if frame_idx < len(self.normalized_data):
            ball_info = self.normalized_data[frame_idx].get('normalized_ball', {})
        
        # Previous frame ball data
        prev_ball_info = None
        if frame_idx > 0 and frame_idx < len(self.normalized_data):
            prev_ball_info = self.normalized_data[frame_idx-1].get('normalized_ball', {})
        
        # Extract ball-related information
        ball_x = ball_info.get('center_x', 0) if ball_info else 0
        ball_y = ball_info.get('center_y', 0) if ball_info else 0
        ball_detected = ball_info.get('detected', False) if ball_info else False
        
        # Calculate ball change amount compared to previous frame
        d_ball_y = 0
        if prev_ball_info:
            prev_ball_y = prev_ball_info.get('center_y', 0)
            d_ball_y = ball_y - prev_ball_y
        
        # Check if arm is extended (Y-axis difference between wrist and shoulder)
        pose = self.normalized_data[frame_idx]['normalized_pose'] if frame_idx < len(self.normalized_data) else {}
        left_shoulder = pose.get('left_shoulder', {'y': 0})
        right_shoulder = pose.get('right_shoulder', {'y': 0})
        left_wrist = pose.get('left_wrist', {'y': 0})
        right_wrist = pose.get('right_wrist', {'y': 0})
        
        shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        wrist_y_actual = (left_wrist['y'] + right_wrist['y']) / 2
        
        # Check if arm is extended (wrist is above shoulder and at appropriate distance)
        arm_extended = (wrist_y_actual < shoulder_y - 0.2 and 
                       abs(wrist_y_actual - shoulder_y) < 0.8)
        
        # Calculate distance between ball and wrist
        ball_wrist_distance = abs(ball_y - wrist_y_actual) if ball_detected else float('inf')
        ball_far_from_hands = ball_wrist_distance > 0.3  # Ball is moving away from hands
        
        # 1. Set-up → Loading: Knee, wrist, and ball move down simultaneously
        if current_phase == "Set-up":
            conditions = []
            
            # Knee moves down
            if d_knee_y < -0.01:
                conditions.append("knee_down")
            
            # Wrist moves down
            if d_wrist_y < -0.01:
                conditions.append("wrist_down")
            
            # Ball moves down
            if ball_detected and d_ball_y > 0.01:
                conditions.append("ball_down")
            
            # Minimum 2 conditions must be met to transition to Loading
            if len(conditions) >= 2:
                if frame_idx % 10 == 0:
                    print(f"Frame {frame_idx}: Set-up→Loading conditions: {conditions}")
                return "Loading"
        
        # 2. Loading → Rising: Wrist, elbow, and ball move up simultaneously
        if current_phase == "Loading":
            conditions = []
            # Calculate elbow y value
            left_elbow = pose.get('left_elbow', {'y': 0})
            right_elbow = pose.get('right_elbow', {'y': 0})
            elbow_y = (left_elbow['y'] + right_elbow['y']) / 2
            if frame_idx > 0:
                prev_pose = self.normalized_data[frame_idx-1]['normalized_pose']
                prev_left_elbow = prev_pose.get('left_elbow', {'y': 0})
                prev_right_elbow = prev_pose.get('right_elbow', {'y': 0})
                prev_elbow_y = (prev_left_elbow['y'] + prev_right_elbow['y']) / 2
                d_elbow_y = elbow_y - prev_elbow_y
                # Both wrist, elbow, and ball move up (Rising)
                if d_wrist_y < 0 and d_elbow_y < 0 and d_ball_y < 0:
                    conditions.append("wrist_up")
                    conditions.append("elbow_up")
                    conditions.append("ball_up")
                # All three conditions must be met to transition to Rising
                if len(conditions) == 3:
                    if frame_idx % 10 == 0:
                        print(f"Frame {frame_idx}: Loading→Rising conditions: {conditions}")
                    return "Rising"
        
        # 3. Rising → Release: When arm is extended (angle 120 degrees or more)
        if current_phase == "Rising":
            conditions = []
            
            # Calculate shoulder-elbow-wrist angle
            left_elbow = pose.get('left_elbow', {'x': 0, 'y': 0})
            right_elbow = pose.get('right_elbow', {'x': 0, 'y': 0})
            left_angle = self._calculate_angle(
                left_shoulder.get('x', 0), left_shoulder.get('y', 0),
                left_elbow.get('x', 0), left_elbow.get('y', 0),
                left_wrist.get('x', 0), left_wrist.get('y', 0)
            )
            right_angle = self._calculate_angle(
                right_shoulder.get('x', 0), right_shoulder.get('y', 0),
                right_elbow.get('x', 0), right_elbow.get('y', 0),
                right_wrist.get('x', 0), right_wrist.get('y', 0)
            )
            if left_angle >= 120 or right_angle >= 120:
                conditions.append("arm_extended")
            
            # Transition to Release if conditions are met
            if len(conditions) >= 1:
                if frame_idx % 10 == 0:
                    print(f"Frame {frame_idx}: Rising→Release conditions: {conditions}")
                    print(f"  left_angle: {left_angle:.1f}, right_angle: {right_angle:.1f}")
                return "Release"
        
        # 4. Release → Follow-through: Ball is completely away from hands + body starts moving down
        if current_phase == "Release":
            conditions = []
            
            # Ball is completely away
            if ball_detected and ball_wrist_distance > 0.5:
                conditions.append("ball_very_far")
            
            # Body starts moving down (knee or butt)
            if d_knee_y > 0.005 or d_hip_y > 0.005:
                conditions.append("body_down")
            
            # Minimum one condition must be met to transition to Follow-through
            if len(conditions) >= 1:
                if frame_idx % 10 == 0:
                    print(f"Frame {frame_idx}: Release→Follow-through conditions: {conditions}")
                return "Follow-through"
        
        # 5. Follow-through → Recovery: Knee goes down and comes up, arm folds below 80 degrees (both conditions met)
        if current_phase == "Follow-through":
            # Check recent 5 frames' knee change amounts
            recent_knee_changes = []
            for i in range(max(0, frame_idx-4), frame_idx+1):
                if i < len(self.normalized_data):
                    pose = self.normalized_data[i]['normalized_pose']
                    knee_y_i = (pose.get('left_knee', {'y': 0})['y'] + 
                               pose.get('right_knee', {'y': 0})['y']) / 2
                    recent_knee_changes.append(knee_y_i)
            # Calculate arm angle
            left_elbow = pose.get('left_elbow', {'x': 0, 'y': 0})
            right_elbow = pose.get('right_elbow', {'x': 0, 'y': 0})
            left_shoulder = pose.get('left_shoulder', {'x': 0, 'y': 0})
            right_shoulder = pose.get('right_shoulder', {'x': 0, 'y': 0})
            left_wrist = pose.get('left_wrist', {'x': 0, 'y': 0})
            right_wrist = pose.get('right_wrist', {'x': 0, 'y': 0})
            left_angle = self._calculate_angle(
                left_shoulder.get('x', 0), left_shoulder.get('y', 0),
                left_elbow.get('x', 0), left_elbow.get('y', 0),
                left_wrist.get('x', 0), left_wrist.get('y', 0)
            )
            right_angle = self._calculate_angle(
                right_shoulder.get('x', 0), right_shoulder.get('y', 0),
                right_elbow.get('x', 0), right_elbow.get('y', 0),
                right_wrist.get('x', 0), right_wrist.get('y', 0)
            )
            arm_folded = (left_angle < 80 or right_angle < 80)
            if len(recent_knee_changes) >= 3:
                # Check if knee starts going down and coming up
                knee_going_down = all(recent_knee_changes[i] >= recent_knee_changes[i-1] 
                                    for i in range(1, len(recent_knee_changes)-1))
                knee_starting_up = recent_knee_changes[-1] < recent_knee_changes[-2]
                if (knee_going_down and knee_starting_up) and arm_folded:
                    if frame_idx % 10 == 0:
                        print(f"Frame {frame_idx}: Follow-through→Recovery conditions: Knee+arm folded (AND)")
                        print(f"  left_angle: {left_angle:.1f}, right_angle: {right_angle:.1f}")
                    return "Recovery"
        
        # 6. Recovery → Set-up: New cycle starts (all movements stabilize)
        if current_phase == "Recovery":
            # Recent 5 frames' change amounts are all small
            recent_changes = []
            for i in range(max(0, frame_idx-4), frame_idx+1):
                if i < len(self.normalized_data):
                    pose = self.normalized_data[i]['normalized_pose']
                    knee_y_i = (pose.get('left_knee', {'y': 0})['y'] + 
                               pose.get('right_knee', {'y': 0})['y']) / 2
                    wrist_y_i = (pose.get('left_wrist', {'y': 0})['y'] + 
                                pose.get('right_wrist', {'y': 0})['y']) / 2
                    hip_y_i = (pose.get('left_hip', {'y': 0})['y'] + 
                              pose.get('right_hip', {'y': 0})['y']) / 2
                    recent_changes.append((knee_y_i, wrist_y_i, hip_y_i))
            
            if len(recent_changes) >= 5:
                # All change amounts are very small
                all_stable = True
                for i in range(1, len(recent_changes)):
                    knee_diff = abs(recent_changes[i][0] - recent_changes[i-1][0])
                    wrist_diff = abs(recent_changes[i][1] - recent_changes[i-1][1])
                    hip_diff = abs(recent_changes[i][2] - recent_changes[i-1][2])
                    
                    if knee_diff > 0.002 or wrist_diff > 0.002 or hip_diff > 0.002:
                        all_stable = False
                        break
                
                if all_stable:
                    if frame_idx % 10 == 0:
                        print(f"Frame {frame_idx}: Recovery→Set-up conditions: All movements stabilize")
                    return "Set-up"
        
        # If no conditions are met, keep current phase
        return current_phase
    
    def save_results(self, video_path: str, overwrite_mode: bool = False):
        """Save results as structured format"""
        print(f"\n💾 STEP 4: Save results")
        print("=" * 50)
        if not self.normalized_data or not self.phases:
            print("❌ No data to save.")
            return
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(self.results_dir, f"{base_name}_normalized_output.json")
        
        # If existing file exists and overwrite mode is not selected, check
        if os.path.exists(output_file) and not overwrite_mode:
            print(f"⚠️ Existing file exists: {output_file}")
            choice = input("Overwrite? (y/n): ").strip().lower()
            if choice != 'y':
                print("Skipping result saving.")
                return
        
        # Configure result data
        results = {
            "metadata": {
                "video_path": video_path,
                "analysis_date": datetime.now().isoformat(),
                "total_frames": len(self.normalized_data),
                "phases_detected": list(set(self.phases)),
                "normalization_method": "ball_radius_based",
                "phase_detection_method": "sequential_transition"
            },
            "frames": []
        }
        
        for i, frame_data in enumerate(self.normalized_data):
            frame_result = {
                "frame_index": i,
                "phase": self.phases[i] if i < len(self.phases) else "Unknown",
                "normalized_pose": frame_data["normalized_pose"],
                "normalized_ball": frame_data["normalized_ball"],
                "original_hip_center": frame_data["original_hip_center"],
                "scaling_factor": frame_data["scaling_factor"],
                "ball_detected": frame_data["ball_detected"]
            }
            results["frames"].append(frame_result)
        
        # Save as JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved: {output_file}")
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    def generate_visualization(self, video_path: str, overwrite_mode: bool = False) -> bool:
        """Generate visualization video (left: original absolute coordinates, right: normalized data)"""
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # If existing file exists and overwrite mode is not selected, check
        output_video = os.path.join(self.visualized_video_dir, f"{base_name}_analyzed.mp4")
        if os.path.exists(output_video) and not overwrite_mode:
            print(f"\n⚠️ Existing visualization video found:")
            print(f"  - {os.path.basename(output_video)}")
            choice = input("Overwrite? (y/n): ").strip().lower()
            if choice != 'y':
                print("Keeping existing visualization video.")
                return True
        
        try:
            # Load original data and normalized data
            original_pose_data = self.pose_data  # Already loaded original data
            normalized_pose_data = self.normalized_data  # Already loaded normalized data
            
            # Load original ball data and normalized ball data
            original_ball_data = self.ball_data  # Already loaded original data
            normalized_ball_data = [frame['normalized_ball'] for frame in self.normalized_data]  # Normalized ball data
            
            self.create_dual_analysis_video(
                video_path=video_path,
                output_path=output_video,
                original_pose_data=original_pose_data,
                normalized_pose_data=normalized_pose_data,
                original_ball_data=original_ball_data,
                normalized_ball_data=normalized_ball_data,
                shooting_phases=self.phases
            )
            print(f"✅ Visualization video generated: {os.path.basename(output_video)}")
            return True
        except Exception as e:
            print(f"❌ Failed to generate visualization video: {e}")
            return False
    
    def create_dual_analysis_video(self, video_path: str, output_path: str, 
                                  original_pose_data: List[Dict], normalized_pose_data: List[Dict],
                                  original_ball_data: List[Dict], normalized_ball_data: List[Dict],
                                  shooting_phases: List[str]) -> bool:
        """Generate dual visualization video (left: original absolute coordinates, right: normalized data)"""
        try:
            # Video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Unable to open video file: {video_path}")
                return False
                
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📹 Video information: {width}x{height}, {fps}fps")
            
            # New size for screen splitting (horizontal 2 times)
            new_width = width * 2
            new_height = height
            
            print(f"🎬 Output size: {new_width}x{new_height}")
            
            # Initialize video writer with H264 codec
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
            
            if not out.isOpened():
                print("❌ Failed to initialize H264 video writer")
                return False
            
            print("✅ H264 video writer initialized successfully")
            
            frame_count = 0
            total_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                
                # Copy original frame
                original_frame = frame.copy()
                normalized_frame = frame.copy()
                
                # Left: Original absolute coordinates data
                if frame_count < len(original_pose_data):
                    original_frame = self._draw_pose_skeleton_original(original_frame, frame_count, original_pose_data)
                    original_frame = self._draw_ball_original(original_frame, frame_count, original_ball_data)
                
                if shooting_phases and frame_count < len(shooting_phases):
                    original_frame = self._draw_phase_label(original_frame, frame_count, "Original", shooting_phases)
                
                # Right: Normalized data
                if frame_count < len(normalized_pose_data):
                    normalized_frame = self._draw_pose_skeleton_normalized(normalized_frame, frame_count, normalized_pose_data)
                    normalized_frame = self._draw_ball_normalized(normalized_frame, frame_count, normalized_ball_data)
                
                if shooting_phases and frame_count < len(shooting_phases):
                    normalized_frame = self._draw_phase_label(normalized_frame, frame_count, "Normalized", shooting_phases)
                
                # Stack two frames side by side
                combined_frame = np.hstack([original_frame, normalized_frame])
                
                out.write(combined_frame)
                frame_count += 1
                
                # Print progress (every 10 frames)
                if frame_count % 10 == 0:
                    print(f"�� Processing frames: {frame_count}/{total_frames}")
            
            cap.release()
            out.release()
            
            print(f"✅ Dual visualization video generated: {output_path}")
            print(f"📊 Total processed frames: {frame_count}")
            print("Left: Original absolute coordinates, Right: Normalized data")
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate dual visualization: {e}")
            return False

    def _draw_pose_skeleton_original(self, frame: np.ndarray, frame_idx: int, pose_data: List[Dict]) -> np.ndarray:
        """Draw original absolute coordinates pose skeleton"""
        if frame_idx >= len(pose_data):
            return frame
        
        # Check data structure and extract pose data
        frame_data = pose_data[frame_idx]
        if isinstance(frame_data, dict):
            pose = frame_data.get('pose', {})
        else:
            pose = frame_data
        
        h, w = frame.shape[:2]
        
        # Define keypoint connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('right_shoulder', 'right_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
            ('left_knee', 'left_ankle'),
            ('right_knee', 'right_ankle')
        ]
        
        # Draw keypoints (use original absolute coordinates)
        for key, kp in pose.items():
            if isinstance(kp, dict) and 'x' in kp and 'y' in kp:
                # Use original absolute coordinates directly
                x = int(kp['x'])
                y = int(kp['y'])
                
                # Coordinate range restriction
                x = max(0, min(w-1, x))
                y = max(0, min(h-1, y))
                
                # Change color based on confidence
                confidence = kp.get('confidence', 0)
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green (high confidence)
                elif confidence > 0.4:
                    color = (0, 255, 255)  # Yellow (medium confidence)
                else:
                    color = (0, 0, 255)  # Red (low confidence)
                cv2.circle(frame, (x, y), 4, color, -1)
        
        # Draw connections (use original absolute coordinates)
        for start_key, end_key in connections:
            if start_key in pose and end_key in pose:
                start_kp = pose[start_key]
                end_kp = pose[end_key]
                
                start_x = int(start_kp['x'])
                start_y = int(start_kp['y'])
                end_x = int(end_kp['x'])
                end_y = int(end_kp['y'])
                
                # Coordinate range restriction
                start_x = max(0, min(w-1, start_x))
                start_y = max(0, min(h-1, start_y))
                end_x = max(0, min(w-1, end_x))
                end_y = max(0, min(h-1, end_y))
                
                # Draw only high-confidence connections
                if (start_kp.get('confidence', 0) > 0.3 and 
                    end_kp.get('confidence', 0) > 0.3):
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        
        return frame

    def _draw_ball_original(self, frame: np.ndarray, frame_idx: int, ball_data: List[Dict]) -> np.ndarray:
        """Draw original absolute coordinates ball"""
        if frame_idx >= len(ball_data):
            return frame
        
        # Check data structure and extract ball data
        frame_data = ball_data[frame_idx]
        if isinstance(frame_data, dict):
            ball_detections = frame_data.get('ball_detections', [])
        else:
            ball_detections = []
        
        for ball in ball_detections:
            if isinstance(ball, dict):
                # Use original absolute coordinates
                center_x = int(ball.get('center_x', 0))
                center_y = int(ball.get('center_y', 0))
                width = int(ball.get('width', 10))
                height = int(ball.get('height', 10))
                confidence = ball.get('confidence', 0)
                
                # Change color based on confidence
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green
                elif confidence > 0.4:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw ball
                cv2.circle(frame, (center_x, center_y), 8, color, -1)
                cv2.circle(frame, (center_x, center_y), 10, (255, 255, 255), 2)
                
                # Confidence text
                cv2.putText(frame, f"{confidence:.2f}", 
                           (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
        
        return frame

    def _draw_pose_skeleton_normalized(self, frame: np.ndarray, frame_idx: int, pose_data: List[Dict]) -> np.ndarray:
        """Draw normalized pose skeleton (centered on screen)"""
        if frame_idx >= len(pose_data):
            return frame
        
        pose = pose_data[frame_idx]['normalized_pose']
        h, w = frame.shape[:2]
        
        # Set center point as reference
        center_x = w // 2
        center_y = h // 2
        
        # Define keypoint connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('right_shoulder', 'right_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
            ('left_knee', 'left_ankle'),
            ('right_knee', 'right_ankle')
        ]
        
        # Draw keypoints (use normalized coordinates to display on screen)
        for key, kp in pose.items():
            if isinstance(kp, dict) and 'x' in kp and 'y' in kp:
                # Convert normalized coordinates to screen coordinates
                # Scale factor is adjusted to fit screen
                scale_factor = min(w, h) / 12
                x = int(center_x + kp['x'] * scale_factor)
                y = int(center_y + kp['y'] * scale_factor)
                
                # Coordinate range restriction
                x = max(0, min(w-1, x))
                y = max(0, min(h-1, y))
                
                # Change color based on confidence
                confidence = kp.get('confidence', 0)
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green (high confidence)
                elif confidence > 0.4:
                    color = (0, 255, 255)  # Yellow (medium confidence)
                else:
                    color = (0, 0, 255)  # Red (low confidence)
                cv2.circle(frame, (x, y), 4, color, -1)
        
        # Draw connections (use normalized coordinates to display on screen)
        for start_key, end_key in connections:
            if start_key in pose and end_key in pose:
                start_kp = pose[start_key]
                end_kp = pose[end_key]
                
                # Scale factor is adjusted to fit screen
                scale_factor = min(w, h) / 12
                start_x = int(center_x + start_kp['x'] * scale_factor)
                start_y = int(center_y + start_kp['y'] * scale_factor)
                end_x = int(center_x + end_kp['x'] * scale_factor)
                end_y = int(center_y + end_kp['y'] * scale_factor)
                
                # Coordinate range restriction
                start_x = max(0, min(w-1, start_x))
                start_y = max(0, min(h-1, start_y))
                end_x = max(0, min(w-1, end_x))
                end_y = max(0, min(h-1, end_y))
                
                # Draw only high-confidence connections
                if (start_kp.get('confidence', 0) > 0.3 and 
                    end_kp.get('confidence', 0) > 0.3):
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        
        return frame
    
    def _draw_ball_normalized(self, frame: np.ndarray, frame_idx: int, ball_data: List[Dict]) -> np.ndarray:
        """Draw normalized ball (centered on screen)"""
        if frame_idx >= len(ball_data):
            return frame
        
        ball_info = ball_data[frame_idx]
        if not ball_info:
            return frame
        
        h, w = frame.shape[:2]
        
        # Set center point as reference
        center_x = w // 2
        center_y = h // 2
        
        # Convert normalized coordinates to screen coordinates
        # Scale factor is adjusted to fit screen
        scale_factor = min(w, h) / 12
        ball_x = int(center_x + ball_info.get('center_x', 0) * scale_factor)
        ball_y = int(center_y + ball_info.get('center_y', 0) * scale_factor)
        
        # Ball size (convert normalized size to appropriate pixel size)
        norm_width = ball_info.get('width', 0.1)
        norm_height = ball_info.get('height', 0.1)
        radius = int(max(norm_width, norm_height) * scale_factor)
        
        # Minimum/maximum radius restriction
        radius = max(3, min(20, radius))
        
        # Coordinate range restriction
        ball_x = max(radius, min(w-radius, ball_x))
        ball_y = max(radius, min(h-radius, ball_y))
        
        # Draw basketball (orange color)
        cv2.circle(frame, (ball_x, ball_y), radius, (0, 165, 255), -1)
        cv2.circle(frame, (ball_x, ball_y), radius, (255, 255, 255), 2)
        
        # Ball center point display
        cv2.circle(frame, (ball_x, ball_y), 3, (255, 255, 255), -1)
        
        return frame
    
    def _draw_phase_label(self, frame: np.ndarray, frame_idx: int, data_type: str = "", shooting_phases: List[str] = None) -> np.ndarray:
        """Draw phase label"""
        if frame_idx < len(shooting_phases):
            phase = shooting_phases[frame_idx]
            
            # Debug: Print phase information (every 10 frames)
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx} ({data_type}): Phase = {phase}")
            
            # Label background
            label_text = f"{data_type}: {phase}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Background rectangle
            cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + 10), (0, 0, 0), -1)
            
            # Text
            cv2.putText(frame, label_text, (15, 15 + text_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Debug: Index out of range
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx} ({data_type}): Phase index out of range (max: {len(shooting_phases) if shooting_phases else 0})")
        
        return frame
    
    def run_analysis(self):
        """Run entire analysis pipeline"""
        print("🏀 Basketball shooting motion analysis pipeline")
        print("=" * 60)
        
        # STEP 0: Select video
        video_path = self.prompt_video_selection()
        if not video_path:
            return
        
        self.selected_video = video_path
        
        # STEP 0.5: Overwrite existing file option
        print(f"\n📁 Overwrite file option")
        print("=" * 50)
        print("If existing extraction data or analysis result files exist:")
        print("1. Overwrite (delete existing files and create new)")
        print("2. Skip (skip if existing files exist)")
        print("3. Cancel")
        
        overwrite_choice = input("Select (1/2/3): ").strip()
        if overwrite_choice == "3":
            print("Analysis canceled.")
            return
        elif overwrite_choice not in ["1", "2"]:
            print("Invalid selection. Proceeding with default (skip) option.")
            overwrite_choice = "2"
        
        overwrite_mode = overwrite_choice == "1"
        
        # STEP 1: Load data
        if not self.load_associated_data(video_path, overwrite_mode):
            return
        
        # STEP 2: Normalize
        self.normalize_pose_data()
        
        # STEP 3: Segment phases
        self.segment_shooting_phases()
        
        # STEP 4: Save results
        self.save_results(video_path, overwrite_mode)
        
        # STEP 5: Visualize (optional)
        self.generate_visualization(video_path, overwrite_mode)
        
        print("\n✅ Analysis completed!")
        print("=" * 60)

    def _calculate_angle(self, ax, ay, bx, by, cx, cy):
        """Return angle between three points (ax,ay)-(bx,by)-(cx,cy) in degrees"""
        import numpy as np
        ba = np.array([ax - bx, ay - by])
        bc = np.array([cx - bx, cy - by])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

def main():
    """Main execution function"""
    analyzer = BasketballShootingAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 