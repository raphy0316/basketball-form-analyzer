# -*- coding: utf-8 -*-
"""
Basketball shooting motion analysis pipeline
Integrate pose data and ball data to analyze shooting movements and visualize
"""

from re import T
import cv2
import numpy as np
import json
import os
import glob
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import math
import matplotlib.pyplot as plt
from pathlib import Path

# Import phase detection modules
from phase_detection.ball_based_phase_detector import BallBasedPhaseDetector
from phase_detection.torso_based_phase_detector import TorsoBasedPhaseDetector
from phase_detection.resolution_based_phase_detector import ResolutionBasedPhaseDetector
from phase_detection.hybrid_fps_phase_detector import HybridFPSPhaseDetector

class BasketballShootingAnalyzer:
    def __init__(self):
        """Initialize the analyzer"""
        self.video_dir = "data/video"
        self.standard_video_dir = os.path.join(self.video_dir, "Standard")
        self.edgecase_video_dir = os.path.join(self.video_dir, "EdgeCase")
        self.extracted_data_dir = "data/extracted_data"
        self.results_dir = "data/results"
        self.visualized_video_dir = "data/visualized_video"
        
        # Data storage
        self.pose_data = []
        self.ball_data = []
        self.rim_data = []
        self.normalized_data = []
        self.phases = []
        self.video_fps = 30.0

        # Detectors
        self.ball_detector = BallBasedPhaseDetector()
        self.torso_detector = TorsoBasedPhaseDetector()
        self.hybrid_fps_detector = HybridFPSPhaseDetector()
        self.resolution_detector = ResolutionBasedPhaseDetector()
        self.current_detector = self.ball_detector
        
        # Hand selection settings
        self.hand_selection_threshold = 0.3  # More relaxed threshold for hand selection (vs 0.2 for phase detection)
        self.hand_selection_min_frames = 10   # Minimum frames to consider a hand as "stable"
        self.hand_selection_stability_bonus = 20  # Bonus points for stable detection
        self.selected_hand = None  # "left" or "right"
        self.selected_hand_confidence = 0.0   # Confidence score for selected hand

    def list_available_videos(self) -> List[str]:
        """Return a list of available video files from Standard, EdgeCase, Bakke, and test folders"""
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        videos = []
        
        # Check Standard folder
        for ext in video_extensions:
            pattern = os.path.join(self.standard_video_dir, ext)
            videos.extend(glob.glob(pattern))
        
        # Check EdgeCase folder
        for ext in video_extensions:
            pattern = os.path.join(self.edgecase_video_dir, ext)
            videos.extend(glob.glob(pattern))
        
        # Check Bakke folder
        bakke_video_dir = os.path.join(self.video_dir, "Bakke")
        if os.path.exists(bakke_video_dir):
            for ext in video_extensions:
                pattern = os.path.join(bakke_video_dir, ext)
                videos.extend(glob.glob(pattern))
        
        # Check test folder
        test_video_dir = os.path.join(self.video_dir, "Test")
        if os.path.exists(test_video_dir):
            # Check combined_output.mov
            combined_video = os.path.join(test_video_dir, "combined_output.mov")
            if os.path.exists(combined_video):
                videos.append(combined_video)
            
            # Check clips folder
            clips_dir = os.path.join(test_video_dir, "Clips")
            if os.path.exists(clips_dir):
                for ext in video_extensions:
                    pattern = os.path.join(clips_dir, ext)
                    videos.extend(glob.glob(pattern))
        
        return sorted(videos)
    
    def prompt_video_selection(self) -> Optional[str]:
        """Prompt user to select processing mode"""
        self.available_videos = self.list_available_videos()
        
        if not self.available_videos:
            print("❌ No video files found in data/video/Standard or data/video/EdgeCase folders.")
            return None
        
        # Categorize videos by folder
        standard_videos = []
        edgecase_videos = []
        
        for video in self.available_videos:
            if self.standard_video_dir in video:
                standard_videos.append(video)
            elif self.edgecase_video_dir in video:
                edgecase_videos.append(video)
        
        print("\n🎬 STEP 0: Select processing mode")
        print("=" * 50)
        print("Available processing options:")
        print(f"[1] Single video selection ({len(self.available_videos)} total videos)")
        print(f"[2] Process all Standard videos ({len(standard_videos)} videos)")
        print(f"[3] Process all EdgeCase videos ({len(edgecase_videos)} videos)")
        print(f"[4] Process all videos ({len(self.available_videos)} videos)")
        print("[5] Cancel")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == "1":
                    # Single video selection
                    return self._prompt_single_video_selection(standard_videos, edgecase_videos)
                
                elif choice == "2":
                    if standard_videos:
                        print(f"✅ Selected: Process all Standard videos ({len(standard_videos)} videos)")
                        return "standard_all"
                    else:
                        print("❌ No videos found in Standard folder.")
                        continue
                
                elif choice == "3":
                    if edgecase_videos:
                        print(f"✅ Selected: Process all EdgeCase videos ({len(edgecase_videos)} videos)")
                        return "edgecase_all"
                    else:
                        print("❌ No videos found in EdgeCase folder.")
                        continue
                
                elif choice == "4":
                    if self.available_videos:
                        print(f"✅ Selected: Process all videos ({len(self.available_videos)} videos)")
                        return "all_videos"
                    else:
                        print("❌ No videos found.")
                        continue
                
                elif choice == "5":
                    print("❌ Analysis canceled.")
                    return None
                
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
                    continue
                    
            except KeyboardInterrupt:
                print("\n❌ Analysis canceled.")
                return None
    
    def _prompt_single_video_selection(self, standard_videos: List[str], edgecase_videos: List[str]) -> Optional[str]:
        """Prompt user to select a single video"""
        print("\nAvailable videos:")
        video_list = []
        video_categories = []
        
        if standard_videos:
            print(f"\n📁 Standard folder:")
            for video in standard_videos:
                display_name = os.path.basename(video)
                print(f"  [{len(video_list) + 1}] {display_name}")
                video_list.append(video)
                video_categories.append("Standard")
        
        if edgecase_videos:
            print(f"\n📁 EdgeCase folder:")
            for video in edgecase_videos:
                display_name = os.path.basename(video)
                print(f"  [{len(video_list) + 1}] {display_name}")
                video_list.append(video)
                video_categories.append("EdgeCase")
        
        print(f"\nTotal videos: {len(video_list)}")
        
        while True:
            try:
                video_choice = input("\nEnter the number or directly enter the file name: ").strip()
                
                # Select by number
                if video_choice.isdigit():
                    idx = int(video_choice) - 1
                    if 0 <= idx < len(video_list):
                        selected_video = video_list[idx]
                        category = video_categories[idx]
                        print(f"✅ Selected: {os.path.basename(selected_video)} ({category})")
                        return selected_video
                    else:
                        print("❌ Invalid number.")
                        continue
                
                # Select by file name
                for i, video in enumerate(video_list):
                    if os.path.basename(video) == video_choice:
                        category = video_categories[i]
                        print(f"✅ Selected: {video_choice} ({category})")
                        return video
                
                print("❌ Invalid selection. Please try again.")
                
            except KeyboardInterrupt:
                print("\n❌ Video selection canceled.")
                return None
    
    def load_associated_data(self, video_path: str, overwrite_mode: bool = False) -> bool:
        """Load original pose/ball data associated with the video"""
        print(f"\n📂 STEP 1: Load original data")
        print("=" * 50)
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Original data file paths
        pose_original_json = os.path.join(self.extracted_data_dir, f"{base_name}_pose_original.json")
        ball_original_json = os.path.join(self.extracted_data_dir, f"{base_name}_ball_original.json")
        rim_original_json = os.path.join(self.extracted_data_dir, f"{base_name}_rim_original.json")
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
        
        # Load rim data
        rim_files = glob.glob(os.path.join(self.extracted_data_dir, f"{base_name}_rim_original*.json"))
        if rim_files:
            rim_file = rim_files[0]  # Use the first file
            try:
                with open(rim_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "rim_info" in data:
                        self.rim_data = data["rim_info"]
                    else:
                        self.rim_data = data
                print(f"✅ Original rim data loaded: {os.path.basename(rim_file)}")
            except Exception as e:
                print(f"❌ Failed to load original rim data: {e}")
                return False
        else:
            print(f"❌ Original ball data file not found: {base_name}_rim_original*.json")
            return False
        
        # Get video FPS
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"✅ Video FPS: {self.video_fps:.2f}")
            else:
                self.video_fps = 30.0  # Default FPS
                print(f"⚠️ Could not read video FPS, using default: {self.video_fps}")
            cap.release()
        except Exception as e:
            self.video_fps = 30.0  # Default FPS
            print(f"⚠️ Error reading video FPS: {e}, using default: {self.video_fps}")
        
        return True
    
    def normalize_pose_data(self, video_path: Optional[str] = None):
        """Normalize pose data (torso-based scaling) and save separately"""
        print(f"\n🔄 STEP 2: Normalize data and save")
        print("=" * 50)
        
        if not self.pose_data:
            print("❌ Pose data not found.")
            return
        
        # Use selected_video if video_path is None
        if video_path is None:
            video_path = self.selected_video
        
        # First pass: collect torso distances from all frames
        print("📊 Collecting torso distances from all frames...")
        torso_distances = []
        
        for i, frame_data in enumerate(self.pose_data):
            pose = frame_data.get('pose', {})
            
            # Get shoulder and hip keypoints
            left_shoulder = pose.get('left_shoulder', {})
            right_shoulder = pose.get('right_shoulder', {})
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            
            # Calculate torso lengths (shoulder to hip distances)
            left_torso_length = 0
            right_torso_length = 0
            
            if (isinstance(left_shoulder, dict) and 'x' in left_shoulder and 'y' in left_shoulder and
                isinstance(left_hip, dict) and 'x' in left_hip and 'y' in left_hip):
                left_torso_length = ((left_hip['x'] - left_shoulder['x'])**2 + 
                                   (left_hip['y'] - left_shoulder['y'])**2)**0.5
            
            if (isinstance(right_shoulder, dict) and 'x' in right_shoulder and 'y' in right_shoulder and
                isinstance(right_hip, dict) and 'x' in right_hip and 'y' in right_hip):
                right_torso_length = ((right_hip['x'] - right_shoulder['x'])**2 + 
                                    (right_hip['y'] - right_shoulder['y'])**2)**0.5
            
            # Use the longer distance as torso size
            if left_torso_length > 0 and right_torso_length > 0:
                torso_distance = max(left_torso_length, right_torso_length)
                torso_distances.append(torso_distance)
                print(f"Frame {i}: Left torso length = {left_torso_length:.4f}, Right torso length = {right_torso_length:.4f}, Using = {torso_distance:.4f}")
            elif left_torso_length > 0:
                torso_distances.append(left_torso_length)
                print(f"Frame {i}: Using left torso length = {left_torso_length:.4f}")
            elif right_torso_length > 0:
                torso_distances.append(right_torso_length)
                print(f"Frame {i}: Using right torso length = {right_torso_length:.4f}")
        
        # Calculate average torso distance
        if torso_distances:
            average_torso_distance = np.mean(torso_distances)
            print(f"✅ Average torso distance: {average_torso_distance:.4f} (from {len(torso_distances)} frames)")
        else:
            average_torso_distance = 0.1  # Default value
            print(f"⚠️ No valid torso distances found, using default: {average_torso_distance}")
        
        # Second pass: normalize all frames using average torso distance
        print("🔄 Normalizing all frames using average torso distance...")
        self.normalized_data = []
        previous_hip_center = None
        consecutive_missing_hip = 0
        max_consecutive_missing = 5  # Warn if more than 5 consecutive frames are missing
        
        for i, frame_data in enumerate(self.pose_data):
            pose = frame_data.get('pose', {})
            
            # Calculate hip_center for reference
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
            
            # Use average torso distance for consistent normalization
            current_scaling_factor = average_torso_distance
            
            # Calculate normalized pose (torso-based scaling)
            normalized_pose = {}
            for key, kp in pose.items():
                if isinstance(kp, dict) and 'x' in kp and 'y' in kp:
                    # Scale normalization using average torso size
                    norm_x = kp['x'] / current_scaling_factor
                    norm_y = kp['y'] / current_scaling_factor
                    
                    normalized_pose[key] = {
                        'x': norm_x,
                        'y': norm_y,
                        'confidence': kp.get('confidence', 0)
                    }
                # Missing keypoints are not added to normalized_pose (automatically excluded)
            
            # Normalize ball position (torso-based scaling)
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
                            'center_x': ball_x / current_scaling_factor,
                            'center_y': ball_y / current_scaling_factor,
                            'width': ball.get('width', 0.01) / current_scaling_factor,
                            'height': ball.get('height', 0.01) / current_scaling_factor
                        }
                        ball_detected = True
            
            normalized_frame = {
                'frame_index': i,
                'normalized_pose': normalized_pose,
                'normalized_ball': normalized_ball,
                'original_hip_center': [hip_center_x, hip_center_y],
                'scaling_factor': current_scaling_factor,
                'average_torso_distance': average_torso_distance,
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
    
    def segment_shooting_phases(self, detector_type: str = "ball"):
        """
        Segment shooting movement into 6 steps using specified detector
        
        Args:
            detector_type: Type of detector to use ("ball", "torso", "hybrid", "resolution")
        """
        print(f"\n📐 STEP 3: Segment shooting phases (using {detector_type} detector)")
        print("=" * 50)
        
        if not self.pose_data or not self.ball_data:
            print("❌ Original pose or ball data not found.")
            return
        
        # Select primary hand before phase detection
        self.selected_hand, self.selected_hand_confidence = self.select_primary_hand(self.normalized_data)
        
        # Set detector based on type
        if detector_type == "ball":
            self.current_detector = self.ball_detector
        elif detector_type == "torso":
            self.current_detector = self.torso_detector
        elif detector_type == "hybrid_fps":
            self.current_detector = self.hybrid_fps_detector
        elif detector_type == "resolution":
            self.current_detector = self.resolution_detector
        else:
            print(f"❌ Unknown detector type: {detector_type}. Using ball detector.")
            self.current_detector = self.ball_detector
        
        self.phases = []
        current_phase = "General" # Start with a general phase
        phase_start_frame = 0
        
        # Track phase history for cancellation
        phase_history = []  # List of (phase, start_frame, end_frame)
        current_phase_start = 0
        
        # Setup for noise filtering
        min_phase_duration = 3  # Must last at least 3 frames
        noise_threshold = 4  # Changes of 4 frames or less are considered noise
        
        for i, frame_data in enumerate(self.normalized_data):
            pose = frame_data.get('normalized_pose', {})
            
            # Get selected hand keypoints
            selected_shoulder, selected_elbow, selected_wrist = self.get_selected_hand_keypoints(pose)
            
            # Extract necessary keypoints from normalized data (using selected hand)
            wrist_x = selected_wrist.get('x', 0)
            wrist_y = selected_wrist.get('y', 0)
            elbow_x = selected_elbow.get('x', 0)
            elbow_y = selected_elbow.get('y', 0)
            shoulder_x = selected_shoulder.get('x', 0)
            shoulder_y = selected_shoulder.get('y', 0)
            
            # Get hip data (still need both for torso calculations)
            left_hip = pose.get('left_hip', {'y': 0})
            right_hip = pose.get('right_hip', {'y': 0})
            left_hip_y = left_hip.get('y', None)
            right_hip_y = right_hip.get('y', None)
            
            if left_hip_y is not None and right_hip_y is not None:
                hip_y = max(left_hip_y, right_hip_y)
            elif left_hip_y is not None:
                hip_y = left_hip_y
            elif right_hip_y is not None:
                hip_y = right_hip_y
            else:
                hip_y = 0
            
            # Get ball position for calculations
            ball_info = None
            if i < len(self.normalized_data):
                normalized_frame = self.normalized_data[i]
                if normalized_frame.get('ball_detected', False):
                    ball_info = normalized_frame.get('normalized_ball', {})
            
            ball_x = ball_info.get('center_x', 0) if ball_info else 0
            ball_y = ball_info.get('center_y', 0) if ball_info else 0
            
            # Calculate movement deltas using selected hand
            if i > 0 and i-1 < len(self.normalized_data):
                prev_frame_data = self.normalized_data[i-1]
                prev_pose = prev_frame_data.get('normalized_pose', {})
                
                # Get previous selected hand position
                prev_selected_shoulder, prev_selected_elbow, prev_selected_wrist = self.get_selected_hand_keypoints(prev_pose)
                prev_wrist_x = prev_selected_wrist.get('x', 0)
                prev_wrist_y = prev_selected_wrist.get('y', 0)
                prev_elbow_x = prev_selected_elbow.get('x', 0)
                prev_elbow_y = prev_selected_elbow.get('y', 0)
                
                # Get previous hip position
                prev_left_hip = prev_pose.get('left_hip', {'y': None})
                prev_right_hip = prev_pose.get('right_hip', {'y': None})
                prev_left_hip_y = prev_left_hip.get('y', None)
                prev_right_hip_y = prev_right_hip.get('y', None)
                
                if prev_left_hip_y is not None and prev_right_hip_y is not None:
                    prev_hip_y = max(prev_left_hip_y, prev_right_hip_y)
                elif prev_left_hip_y is not None:
                    prev_hip_y = prev_left_hip_y
                elif prev_right_hip_y is not None:
                    prev_hip_y = prev_right_hip_y
                else:
                    prev_hip_y = hip_y
                
                d_wrist_y = wrist_y - prev_wrist_y
                d_elbow_y = elbow_y - prev_elbow_y
                d_hip_y = hip_y - prev_hip_y
            else:
                d_wrist_y = d_elbow_y = d_hip_y = 0
            
            # Check if current phase transitions to next phase using current detector
            # Get FPS from video if available
            fps = 30.0  # Default FPS
            if hasattr(self, 'video_fps'):
                fps = self.video_fps
                
            next_phase = self.current_detector.check_phase_transition(
                current_phase, i, self.pose_data, self.ball_data, 
                fps=fps, selected_hand=self.selected_hand
            )
            
            # Minimum phase duration disabled - transition immediately when conditions are met
            if next_phase != current_phase:
                            # Record current phase in history before changing
                            if current_phase != "General":
                                phase_history.append((current_phase, current_phase_start, i))
                            
                            current_phase = next_phase
                            phase_start_frame = i
                            current_phase_start = i
                            print(f"Frame {i}: {current_phase} phase started")
            
            self.phases.append(current_phase)
        
        # Print phase-by-frame statistics
        phase_counts = {}
        for phase in self.phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        print("\nPhase-by-frame count:")
        for phase, count in phase_counts.items():
            print(f"  {phase}: {count} frames")
        
        # Process cancellations (Phase Filling enabled)
        self._process_cancellations()
    
    def _is_trend_based_transition(self, frame_idx: int, current_phase: str, next_phase: str, noise_threshold: int) -> bool:
        """Trend-based transition determination (always returns True)"""
        return True
    
    def _finalize_phases_by_trend(self, noise_threshold: int):
        """Final trend-based organization"""
        if not self.phases:
            return
        
        print("\n🔄 Finalizing phases by trend...")
        
        # Minimum frame duration disabled - no phase post-processing
        print("  Phase duration enforcement disabled.")
    
    def _process_cancellations(self):
        """Process cancellations by replacing abnormal transitions with Set-up"""
        if not self.phases:
            return
        
        print("\n🔄 Processing cancellations...")
        
        # Define normal phase transitions (same phase transitions are also normal)
        cancel_transitions = {
            "Loading": ["Set-up", "General"],
            "Loading-Rising": ["Set-up", "General"],  # Added Loading-Rising
            "Rising": ["Set-up", "General"],
        }
        
        # Find abnormal transitions
        abnormal_points = []
        for i in range(1, len(self.phases)):
            current_phase = self.phases[i-1]
            next_phase = self.phases[i]
            
            # Check if this transition is normal
            is_normal = True
            if current_phase in cancel_transitions:
                if next_phase in cancel_transitions[current_phase]:
                    is_normal = False
            
            # If transition is abnormal, mark it
            if not is_normal:
                abnormal_points.append(i)
                print(f"    Abnormal transition at frame {i}: {current_phase} → {next_phase}")
        
        if not abnormal_points:
            print("  No abnormal transitions found.")
            return
        
        print(f"  Found {len(abnormal_points)} abnormal transition points.")
        
        # Process each abnormal transition point
        for abnormal_point in abnormal_points:
            # Find the start of the sequence to replace (look backwards for consecutive phases)
            start_point = abnormal_point - 1
            while start_point >= 0 and self.phases[start_point] not in ["General", "Set-up"]:
                start_point -= 1
            
            # Replace the entire sequence with Set-up
            for i in range(start_point + 1, abnormal_point):
                if self.phases[i] in ["Loading", "Loading-Rising", "Rising", "Release", "Follow-through"]:  # Added Loading-Rising
                    self.phases[i] = "Set-up"
                    print(f"    Frame {i}: {self.phases[i]} → Set-up (abnormal transition)")
        
        print("  Cancellation processing completed.")
    
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
                "phase_detection_method": "sequential_transition",
                "hand": self.selected_hand,
                "fps" : self.video_fps,
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
        """Generate visualization video (original data only)"""
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
            # Load original data only
            original_pose_data = self.pose_data  # Already loaded original data
            original_ball_data = self.ball_data  # Already loaded original data
            original_rim_data = self.rim_data
            
            self.create_original_analysis_video(
                video_path=video_path,
                output_path=output_video,
                original_pose_data=original_pose_data,
                original_ball_data=original_ball_data,
                original_rim_data=original_rim_data,
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
                                  original_rim_data: List[Dict], shooting_phases: List[str]) -> bool:
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
            
            # Initialize video writer with mp4v codec (fallback from H264)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                    original_frame = self._draw_rim_original(original_frame, frame_count, original_rim_data)
    
                if shooting_phases and frame_count < len(shooting_phases):
                    original_frame = self._draw_phase_label(original_frame, frame_count, "Original", shooting_phases)
                
                # Add selected hand label to original frame
                original_frame = self._draw_selected_hand_label(original_frame, self.selected_hand, self.selected_hand_confidence)
                
                # Right: Normalized data
                if frame_count < len(normalized_pose_data):
                    normalized_frame = self._draw_pose_skeleton_normalized(normalized_frame, frame_count, normalized_pose_data)
                    normalized_frame = self._draw_ball_normalized(normalized_frame, frame_count, normalized_ball_data)
                
                if shooting_phases and frame_count < len(shooting_phases):
                    normalized_frame = self._draw_phase_label(normalized_frame, frame_count, "Normalized", shooting_phases)
                
                # Add selected hand label to normalized frame
                normalized_frame = self._draw_selected_hand_label(normalized_frame, self.selected_hand, self.selected_hand_confidence)
                
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

    def create_original_analysis_video(self, video_path: str, output_path: str, 
                                      original_pose_data: List[Dict], original_ball_data: List[Dict],
                                      original_rim_data: List[Dict], shooting_phases: List[str]) -> bool:
        """Generate original data visualization video"""
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
            print(f"🎬 Output size: {width}x{height}")
            
            # Initialize video writer with mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print("❌ Failed to initialize video writer")
                return False
            
            print("✅ Video writer initialized successfully")
            
            frame_count = 0
            total_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                
                # Draw original data on frame
                if frame_count < len(original_pose_data):
                    frame = self._draw_pose_skeleton_original(frame, frame_count, original_pose_data)
                    frame = self._draw_ball_original(frame, frame_count, original_ball_data)
                    frame = self._draw_rim_original(frame, frame_count, original_rim_data)
    
                if shooting_phases and frame_count < len(shooting_phases):
                    frame = self._draw_phase_label(frame, frame_count, "Original", shooting_phases)
                
                # Add selected hand label
                frame = self._draw_selected_hand_label(frame, self.selected_hand, self.selected_hand_confidence)
                
                out.write(frame)
                frame_count += 1
                
                # Print progress (every 10 frames)
                if frame_count % 10 == 0:
                    print(f"🎬 Processing frames: {frame_count}/{total_frames}")
            
            cap.release()
            out.release()
            
            print(f"✅ Original data visualization video generated: {output_path}")
            print(f"📊 Total processed frames: {frame_count}")
            print("Original data with aspect ratio corrected coordinates")
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate original visualization: {e}")
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
        aspect_ratio = w / h
        
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
        
        # Draw keypoints (convert aspect ratio corrected relative coordinates to pixel coordinates)
        for key, kp in pose.items():
            if isinstance(kp, dict) and 'x' in kp and 'y' in kp:
                # x: (rel_x / aspect_ratio) * w
                x = int(kp['x'] / aspect_ratio * w)
                y = int(kp['y'] * h)
                x = max(0, min(w-1, x))
                y = max(0, min(h-1, y))
                confidence = kp.get('confidence', 0)
                if confidence > 0.7:
                    color = (0, 255, 0)
                elif confidence > 0.4:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                cv2.circle(frame, (x, y), 4, color, -1)
        
        # Draw connections (convert aspect ratio corrected relative coordinates to pixel coordinates)
        for start_key, end_key in connections:
            if start_key in pose and end_key in pose:
                start_kp = pose[start_key]
                end_kp = pose[end_key]
                start_x = int(start_kp['x'] / aspect_ratio * w)
                start_y = int(start_kp['y'] * h)
                end_x = int(end_kp['x'] / aspect_ratio * w)
                end_y = int(end_kp['y'] * h)
                start_x = max(0, min(w-1, start_x))
                start_y = max(0, min(h-1, start_y))
                end_x = max(0, min(w-1, end_x))
                end_y = max(0, min(h-1, end_y))
                if (start_kp.get('confidence', 0) > 0.3 and end_kp.get('confidence', 0) > 0.3):
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        
        # Draw arm angles
        self._draw_arm_angles_original(frame, pose, w, h)
        
        return frame

    def _draw_arm_angles_original(self, frame: np.ndarray, pose: Dict, w: int, h: int):
        font_scale = 1.0
        thickness = 2
        aspect_ratio = w / h
        if all(key in pose for key in ['left_shoulder', 'left_elbow', 'left_wrist']):
            left_shoulder = pose['left_shoulder']
            left_elbow = pose['left_elbow']
            left_wrist = pose['left_wrist']
            if (left_shoulder.get('confidence', 0) > 0.3 and left_elbow.get('confidence', 0) > 0.3 and left_wrist.get('confidence', 0) > 0.3):
                left_angle = self._calculate_angle(
                    left_shoulder['x'], left_shoulder['y'],
                    left_elbow['x'], left_elbow['y'],
                    left_wrist['x'], left_wrist['y']
                )
                elbow_x = int(left_elbow['x'] / aspect_ratio * w)
                elbow_y = int(left_elbow['y'] * h)
                wrist_x = int(left_wrist['x'] / aspect_ratio * w)
                wrist_y = int(left_wrist['y'] * h)
                elbow_x = max(0, min(w-1, elbow_x))
                elbow_y = max(0, min(h-1, elbow_y))
                wrist_x = max(0, min(w-1, wrist_x))
                wrist_y = max(0, min(h-1, wrist_y))
                if 110 <= left_angle <= 180:
                    color = (0, 255, 0)
                elif 90 <= left_angle < 110:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                text = f"L:{left_angle:.0f}"
                vec_x = wrist_x - elbow_x
                vec_y = wrist_y - elbow_y
                norm = (vec_x**2 + vec_y**2)**0.5
                if norm > 0:
                    offset_x = int(vec_x / norm * 40)
                    offset_y = int(vec_y / norm * 40)
                else:
                    offset_x = -40
                    offset_y = 20
                text_pos = (elbow_x + offset_x, elbow_y + offset_y)
                cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        if all(key in pose for key in ['right_shoulder', 'right_elbow', 'right_wrist']):
            right_shoulder = pose['right_shoulder']
            right_elbow = pose['right_elbow']
            right_wrist = pose['right_wrist']
            if (right_shoulder.get('confidence', 0) > 0.3 and right_elbow.get('confidence', 0) > 0.3 and right_wrist.get('confidence', 0) > 0.3):
                right_angle = self._calculate_angle(
                    right_shoulder['x'], right_shoulder['y'],
                    right_elbow['x'], right_elbow['y'],
                    right_wrist['x'], right_wrist['y']
                )
                elbow_x = int(right_elbow['x'] / aspect_ratio * w)
                elbow_y = int(right_elbow['y'] * h)
                wrist_x = int(right_wrist['x'] / aspect_ratio * w)
                wrist_y = int(right_wrist['y'] * h)
                elbow_x = max(0, min(w-1, elbow_x))
                elbow_y = max(0, min(h-1, elbow_y))
                wrist_x = max(0, min(w-1, wrist_x))
                wrist_y = max(0, min(h-1, wrist_y))
                if 110 <= right_angle <= 180:
                    color = (0, 255, 0)
                elif 90 <= right_angle < 110:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                text = f"R:{right_angle:.0f}"
                vec_x = wrist_x - elbow_x
                vec_y = wrist_y - elbow_y
                norm = (vec_x**2 + vec_y**2)**0.5
                if norm > 0:
                    offset_x = int(vec_x / norm * 60)
                    offset_y = int(vec_y / norm * 60)
                else:
                    offset_x = -60
                    offset_y = 30
                text_pos = (elbow_x + offset_x, elbow_y + offset_y)
                cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def _draw_ball_original(self, frame: np.ndarray, frame_idx: int, ball_data: List[Dict]) -> np.ndarray:
        if frame_idx >= len(ball_data):
            return frame
        frame_data = ball_data[frame_idx]
        if isinstance(frame_data, dict):
            ball_detections = frame_data.get('ball_detections', [])
        else:
            ball_detections = []
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        for ball in ball_detections:
            if isinstance(ball, dict):
                center_x = int(ball.get('center_x', 0) / aspect_ratio * w)
                center_y = int(ball.get('center_y', 0) * h)
                width = int(ball.get('width', 0.01))
                height = int(ball.get('height', 0.01))
                confidence = ball.get('confidence', 0)
                if confidence > 0.7:
                    color = (0, 255, 0)
                elif confidence > 0.4:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                cv2.circle(frame, (center_x, center_y), 8, color, -1)
                cv2.circle(frame, (center_x, center_y), 10, (255, 255, 255), 2)
                cv2.putText(frame, f"{confidence:.2f}", (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
    
    def _draw_rim_original(self, frame: np.ndarray, frame_idx: int, rim_data: List[Dict]) -> np.ndarray:
        if(frame_idx >= len(rim_data)):
            return frame
        frame_data = rim_data[frame_idx]
        if isinstance(frame_data, dict):
            rim_detections = frame_data.get('rim_detections', [])
        else:
            rim_detections = []
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        for rim in rim_detections:
            if isinstance(rim, dict):
                center_x = int(rim.get('center_x', 0) / aspect_ratio * w)
                center_y = int(rim.get('center_y', 0) * h)
                width = int(rim.get('width', 0.01))
                height = int(rim.get('height', 0.01))
                confidence = rim.get('confidence', 0)
                if confidence > 0.7:
                    color = (0, 255, 0)
                elif confidence > 0.4:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                cv2.rectangle(frame, (center_x - width // 2, center_y - height // 2), (center_x + width // 2, center_y + height // 2), color, 2)
                cv2.putText(frame, f"{confidence:.2f}", (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
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
        
        # Draw arm angles
        self._draw_arm_angles_normalized(frame, pose, w, h, center_x, center_y)
        
        return frame

    def _draw_arm_angles_normalized(self, frame: np.ndarray, pose: Dict, w: int, h: int, center_x: int, center_y: int):
        scale_factor = min(w, h) / 12
        font_scale = 1.0  # Adjust font size
        thickness = 2      # Reduce thickness
        # Calculate left arm angle
        if all(key in pose for key in ['left_shoulder', 'left_elbow', 'left_wrist']):
            left_shoulder = pose['left_shoulder']
            left_elbow = pose['left_elbow']
            left_wrist = pose['left_wrist']
            if (left_shoulder.get('confidence', 0) > 0.3 and 
                left_elbow.get('confidence', 0) > 0.3 and 
                left_wrist.get('confidence', 0) > 0.3):
                left_angle = self._calculate_angle(
                    left_shoulder['x'], left_shoulder['y'],
                    left_elbow['x'], left_elbow['y'],
                    left_wrist['x'], left_wrist['y']
                )
                # Convert normalized coordinates to screen coordinates
                elbow_x = int(center_x + left_elbow['x'] * scale_factor)
                elbow_y = int(center_y + left_elbow['y'] * scale_factor)
                wrist_x = int(center_x + left_wrist['x'] * scale_factor)
                wrist_y = int(center_y + left_wrist['y'] * scale_factor)
                elbow_x = max(0, min(w-1, elbow_x))
                elbow_y = max(0, min(h-1, elbow_y))
                wrist_x = max(0, min(w-1, wrist_x))
                wrist_y = max(0, min(h-1, wrist_y))
                if 110 <= left_angle <= 180:
                    color = (0, 255, 0)
                elif 90 <= left_angle < 110:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                text = f"L:{left_angle:.0f}"
                # Move 40 pixels in elbow→wrist vector direction (reduced distance)
                vec_x = wrist_x - elbow_x
                vec_y = wrist_y - elbow_y
                norm = (vec_x**2 + vec_y**2)**0.5
                if norm > 0:
                    offset_x = int(vec_x / norm * 40)
                    offset_y = int(vec_y / norm * 40)
                else:
                    offset_x = 30
                    offset_y = 30
                text_pos = (elbow_x + offset_x, elbow_y + offset_y)
                cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        # Calculate right arm angle
        if all(key in pose for key in ['right_shoulder', 'right_elbow', 'right_wrist']):
            right_shoulder = pose['right_shoulder']
            right_elbow = pose['right_elbow']
            right_wrist = pose['right_wrist']
            if (right_shoulder.get('confidence', 0) > 0.3 and 
                right_elbow.get('confidence', 0) > 0.3 and 
                right_wrist.get('confidence', 0) > 0.3):
                right_angle = self._calculate_angle(
                    right_shoulder['x'], right_shoulder['y'],
                    right_elbow['x'], right_elbow['y'],
                    right_wrist['x'], right_wrist['y']
                )
                elbow_x = int(center_x + right_elbow['x'] * scale_factor)
                elbow_y = int(center_y + right_elbow['y'] * scale_factor)
                wrist_x = int(center_x + right_wrist['x'] * scale_factor)
                wrist_y = int(center_y + right_wrist['y'] * scale_factor)
                elbow_x = max(0, min(w-1, elbow_x))
                elbow_y = max(0, min(h-1, elbow_y))
                wrist_x = max(0, min(w-1, wrist_x))
                wrist_y = max(0, min(h-1, wrist_y))
                if 110 <= right_angle <= 180:
                    color = (0, 255, 0)
                elif 90 <= right_angle < 110:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                text = f"R:{right_angle:.0f}"
                # Move 60 pixels in elbow→wrist vector direction (right side further)
                vec_x = wrist_x - elbow_x
                vec_y = wrist_y - elbow_y
                norm = (vec_x**2 + vec_y**2)**0.5
                if norm > 0:
                    offset_x = int(vec_x / norm * 60)
                    offset_y = int(vec_y / norm * 60)
                else:
                    offset_x = -60
                    offset_y = 30
                text_pos = (elbow_x + offset_x, elbow_y + offset_y)
                cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def _draw_ball_normalized(self, frame: np.ndarray, frame_idx: int, ball_data: List[Dict]) -> np.ndarray:
        """Draw normalized ball (centered on screen)"""
        if frame_idx >= len(ball_data):
            return frame
        
        # Get normalized ball data from normalized_data
        if hasattr(self, 'normalized_data') and frame_idx < len(self.normalized_data):
            normalized_frame = self.normalized_data[frame_idx]
            ball_info = normalized_frame.get('normalized_ball', {})
            ball_detected = normalized_frame.get('ball_detected', False)
        else:
            ball_info = ball_data[frame_idx] if frame_idx < len(ball_data) else {}
            ball_detected = True
        
        if not ball_detected or not ball_info:
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
        """
        Draw phase label in top left corner with small font.
        
        Args:
            frame: The video frame to draw on
            frame_idx: Current frame index
            data_type: Label for the data type ("Original", "Normalized", etc.)
            shooting_phases: List of detected shooting phases
            
        Returns:
            Frame with added phase label
        """
        # Handle None case for shooting_phases
        if shooting_phases is None or frame_idx >= len(shooting_phases):
            return frame
        
        phase = shooting_phases[frame_idx]
        
        # Debug: Print phase information (every 10 frames)
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx} ({data_type}): Phase = {phase}")
        
        # Label text (just the phase, without data_type to keep it small)
        label_text = f"{phase}"
        
        # Font settings - small size
        font_scale = 0.4
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        # Add padding to the background rectangle
        padding = 2
        bg_width = text_width + padding * 2
        bg_height = text_height + padding * 2
        
        # Position at top left corner
        bg_x = 5
        bg_y = 5
        
        # Phase-specific color coding
        phase_colors = {
            "General": (128, 128, 128),    # Gray
            "Set-up": (0, 153, 255),       # Orange
            "Loading": (0, 128, 255),      # Orange-Yellow
            "Rising": (0, 255, 255),       # Yellow
            "Release": (0, 255, 0),        # Green
            "Follow-through": (255, 0, 0)  # Blue (BGR color order)
        }
        
        # Get color for current phase (default to white if not in dictionary)
        phase_color = phase_colors.get(phase, (255, 255, 255))
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                    (bg_x, bg_y), 
                    (bg_x + bg_width, bg_y + bg_height), 
                    (0, 0, 0), -1)
        
        # Draw colored indicator at left edge
        indicator_width = 2
        cv2.rectangle(frame, 
                    (bg_x, bg_y), 
                    (bg_x + indicator_width, bg_y + bg_height), 
                    phase_color, -1)
        
        # Draw text
        text_x = bg_x + indicator_width + padding
        text_y = bg_y + text_height + padding - 1  # -1 to adjust vertical position
        cv2.putText(frame, label_text, (text_x, text_y), 
                font, font_scale, (255, 255, 255), font_thickness)
        
        return frame
        
    def _draw_selected_hand_label(self, frame: np.ndarray, selected_hand: str = None, confidence: float = 0.0) -> np.ndarray:
        """Draw selected hand label in top-right corner with small font"""
        if selected_hand is None:
            return frame
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Create compact label text (combined into one line)
        label_text = f"{selected_hand.upper()} ({confidence:.0f}%)"
        
        # Font settings - small size
        font_scale = 0.4
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        # Add padding to the background rectangle
        padding = 2
        bg_width = text_width + padding * 2
        bg_height = text_height + padding * 2
        
        # Position in top-right corner
        bg_x = w - bg_width - 5
        bg_y = 5
        
        # Hand color based on left/right
        hand_color = (0, 128, 255) if selected_hand.lower() == "left" else (255, 128, 0)
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                    (bg_x, bg_y), 
                    (bg_x + bg_width, bg_y + bg_height), 
                    (0, 0, 0), -1)
        
        # Draw colored indicator at right edge
        indicator_width = 2
        cv2.rectangle(frame, 
                    (bg_x + bg_width - indicator_width, bg_y), 
                    (bg_x + bg_width, bg_y + bg_height), 
                    hand_color, -1)
        
        # Draw text
        text_x = bg_x + padding
        text_y = bg_y + text_height + padding - 1
        cv2.putText(frame, label_text, (text_x, text_y), 
                font, font_scale, (255, 255, 255), font_thickness)
        
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
        
        # Vector AB
        ab_x = ax - bx
        ab_y = ay - by
        
        # Vector CB
        cb_x = cx - bx
        cb_y = cy - by
        
        # Dot product
        dot_product = ab_x * cb_x + ab_y * cb_y
        
        # Magnitudes
        ab_magnitude = np.sqrt(ab_x**2 + ab_y**2)
        cb_magnitude = np.sqrt(cb_x**2 + cb_y**2)
        
        # Avoid division by zero
        if ab_magnitude == 0 or cb_magnitude == 0:
            return 0.0
        
        # Cosine of angle
        cos_angle = dot_product / (ab_magnitude * cb_magnitude)
        
        # Clamp to valid range
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle

    def select_primary_hand(self, normalized_data: List[Dict]) -> Tuple[str, float]:
        """
        Select the primary hand based on 2-stage algorithm:
        1. Proximity to ball (primary criterion)
        2. Detection stability (secondary criterion when proximity is similar)
        
        Args:
            normalized_data: List of normalized frame data
            
        Returns:
            Tuple of (selected_hand, confidence_score)
        """
        print(f"\n🤚 Selecting primary hand for phase detection...")
        print(f"  Hand selection threshold: {self.hand_selection_threshold}")
        print(f"  Minimum stable frames: {self.hand_selection_min_frames}")
        
        left_hand_stats = {"close_frames": 0, "total_detected": 0, "wrist_detected": 0, "elbow_detected": 0}
        right_hand_stats = {"close_frames": 0, "total_detected": 0, "wrist_detected": 0, "elbow_detected": 0}
        
        # Stage 1: Collect proximity and detection statistics
        for i, frame_data in enumerate(normalized_data):
            pose = frame_data.get('normalized_pose', {})
            ball_info = frame_data.get('normalized_ball', {})
            
            if not ball_info or not pose:
                continue
            
            # Get ball position
            ball_x = ball_info.get('center_x', 0)
            ball_y = ball_info.get('center_y', 0)

            # Check left hand
            left_wrist = pose.get('left_wrist')
            left_elbow = pose.get('left_elbow')
            if left_wrist:
                left_wrist_x = left_wrist.get('x', 0)
                left_wrist_y = left_wrist.get('y', 0)
                left_distance = ((ball_x - left_wrist_x)**2 + (ball_y - left_wrist_y)**2)**0.5

                if left_distance < self.hand_selection_threshold:
                    # Stability calculated only from frames where ball was close
                    if left_wrist and isinstance(left_wrist, dict) and 'x' in left_wrist and 'y' in left_wrist:
                        left_hand_stats["wrist_detected"] += 1
                    if left_elbow and isinstance(left_elbow, dict) and 'x' in left_elbow and 'y' in left_elbow:
                        left_hand_stats["elbow_detected"] += 1
                left_hand_stats["total_detected"] += 1

            # Check right hand
            right_wrist = pose.get('right_wrist')
            right_elbow = pose.get('right_elbow')
            if right_wrist:
                right_wrist_x = right_wrist.get('x', 0)
                right_wrist_y = right_wrist.get('y', 0)
                right_distance = ((ball_x - right_wrist_x)**2 + (ball_y - right_wrist_y)**2)**0.5

                if right_distance < self.hand_selection_threshold:
                    right_hand_stats["close_frames"] += 1
                    # Stability calculated only from frames where ball was close
                    if right_wrist and isinstance(right_wrist, dict) and 'x' in right_wrist and 'y' in right_wrist:
                        right_hand_stats["wrist_detected"] += 1
                    if right_elbow and isinstance(right_elbow, dict) and 'x' in right_elbow and 'y' in right_elbow:
                        right_hand_stats["elbow_detected"] += 1
                right_hand_stats["total_detected"] += 1

        # Calculate proximity ratios
        left_proximity_ratio = 0.0
        right_proximity_ratio = 0.0

        if left_hand_stats["total_detected"] > 0:
            left_proximity_ratio = left_hand_stats["close_frames"] / left_hand_stats["total_detected"]
        if right_hand_stats["total_detected"] > 0:
            right_proximity_ratio = right_hand_stats["close_frames"] / right_hand_stats["total_detected"]

        # Calculate detection stability scores
        left_stability_score = 0.0
        right_stability_score = 0.0

        if left_hand_stats["close_frames"] > 0:
            wrist_ratio = left_hand_stats["wrist_detected"] / left_hand_stats["close_frames"]
            elbow_ratio = left_hand_stats["elbow_detected"] / left_hand_stats["close_frames"]
            left_stability_score = (wrist_ratio + elbow_ratio) / 2  # Average of wrist and elbow detection

        if right_hand_stats["close_frames"] > 0:
            wrist_ratio = right_hand_stats["wrist_detected"] / right_hand_stats["close_frames"]
            elbow_ratio = right_hand_stats["elbow_detected"] / right_hand_stats["close_frames"]
            right_stability_score = (wrist_ratio + elbow_ratio) / 2  # Average of wrist and elbow detection

        print(f"  Left hand - Proximity: {left_proximity_ratio:.3f}, Stability: {left_stability_score:.3f}")
        print(f"  Right hand - Proximity: {right_proximity_ratio:.3f}, Stability: {right_stability_score:.3f}")

        # Stage 1: Check if proximity difference is significant
        proximity_difference = abs(left_proximity_ratio - right_proximity_ratio)
        proximity_threshold = 0.2  # 10% difference threshold

        if proximity_difference > proximity_threshold:
            # Significant difference in proximity - use proximity as primary criterion
            selected_hand = "left" if left_proximity_ratio > right_proximity_ratio else "right"
            confidence = max(left_proximity_ratio, right_proximity_ratio) * 100
            print(f"  Stage 1 decision: {selected_hand} (proximity difference: {proximity_difference:.3f})")
        else:
            # Similar proximity - use detection stability as secondary criterion
            selected_hand = "left" if left_stability_score > right_stability_score else "right"
            confidence = max(left_stability_score, right_stability_score) * 100
            print(f"  Stage 2 decision: {selected_hand} (similar proximity, using stability)")

        print(f"  Selected hand: {selected_hand} (confidence: {confidence:.1f}%)")
        
        return selected_hand, confidence

    def get_selected_hand_keypoints(self, pose: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        Get keypoints for the selected hand.
        
        Args:
            pose: Pose data for current frame
            
        Returns:
            Tuple of (shoulder, elbow, wrist) for selected hand
        """
        if self.selected_hand == "left":
            shoulder = pose.get('left_shoulder', {})
            elbow = pose.get('left_elbow', {})
            wrist = pose.get('left_wrist', {})
        else:  # right
            shoulder = pose.get('right_shoulder', {})
            elbow = pose.get('right_elbow', {})
            wrist = pose.get('right_wrist', {})
        
        return shoulder, elbow, wrist
    
    def get_selected_hand_position(self, pose: Dict) -> Tuple[float, float]:
        """
        Get position of the selected hand.
        
        Args:
            pose: Pose data for current frame
            
        Returns:
            Tuple of (x, y) coordinates for selected hand wrist
        """
        if self.selected_hand == "left":
            wrist = pose.get('left_wrist', {})
        else:  # right
            wrist = pose.get('right_wrist', {})
        
        return wrist.get('x', 0), wrist.get('y', 0)

def main():
    """Main execution function"""
    analyzer = BasketballShootingAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 