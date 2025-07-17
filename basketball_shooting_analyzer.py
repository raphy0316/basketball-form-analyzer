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
        self.standard_video_dir = os.path.join(self.video_dir, "Standard")
        self.edgecase_video_dir = os.path.join(self.video_dir, "EdgeCase")
        self.extracted_data_dir = os.path.join(self.references_dir, "extracted_data")
        self.results_dir = os.path.join(self.references_dir, "results")
        self.visualized_video_dir = os.path.join(self.references_dir, "visualized_video")
        
        # Create directories
        for dir_path in [self.video_dir, self.standard_video_dir, self.edgecase_video_dir, 
                        self.extracted_data_dir, self.results_dir, self.visualized_video_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Data storage
        self.pose_data = []
        self.ball_data = []
        self.rim_data = []
        self.normalized_data = []
        self.phases = []
        self.phase_statistics = {}
        self.selected_video = None
        self.available_videos = []
        
        # Video properties for dynamic frame calculation
        self.video_fps = None
        self.video_total_frames = None

    def get_video_properties(self, video_path: str) -> Tuple[float, int]:
        """Get video FPS and total frame count"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return 30.0, 0  # Default values
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return fps, total_frames

    def calculate_dynamic_min_frames(self, fps: float, min_duration_ms: float = 100.0) -> int:
        """Calculate minimum frame duration based on FPS and desired minimum duration in milliseconds"""
        min_frames = max(1, int(fps * min_duration_ms / 1000.0))
        return min_frames
    
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
        test_video_dir = os.path.join(self.video_dir, "test")
        if os.path.exists(test_video_dir):
            # Check combined_output.mov
            combined_video = os.path.join(test_video_dir, "combined_output.mov")
            if os.path.exists(combined_video):
                videos.append(combined_video)
            
            # Check clips folder
            clips_dir = os.path.join(test_video_dir, "clips")
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
        
        # Get video properties for dynamic frame calculation
        self.video_fps, self.video_total_frames = self.get_video_properties(video_path)
        print(f"📹 Video properties: {self.video_total_frames} frames, {self.video_fps:.2f} fps")
        
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
        """Segment shooting movement into 6 steps using original data"""
        print(f"\n📐 STEP 3: Segment shooting phases (using original data)")
        print("=" * 50)
        
        if not self.pose_data or not self.ball_data:
            print("❌ Original pose or ball data not found.")
            return
        
        self.phases = []
        current_phase = "General" # Start with a general phase
        phase_start_frame = 0
        
        # Track phase history for cancellation
        phase_history = []  # List of (phase, start_frame, end_frame)
        current_phase_start = 0
        
        # Setup for noise filtering with dynamic frame calculation
        if self.video_fps is not None:
            # Calculate minimum phase duration based on FPS (100ms minimum duration)
            min_phase_duration = self.calculate_dynamic_min_frames(self.video_fps, 100.0)
            print(f"🎯 Dynamic minimum phase duration: {min_phase_duration} frames ({100.0}ms at {self.video_fps:.2f} fps)")
        else:
            # Fallback to fixed duration if FPS is not available
            min_phase_duration = 3
            print(f"⚠️ Using fallback minimum phase duration: {min_phase_duration} frames (FPS not available)")
        
        noise_threshold = 4  # Changes of 4 frames or less are considered noise
        
        for i, frame_data in enumerate(self.pose_data):
            pose = frame_data.get('pose', {})
            
            # Extract necessary keypoints from original data
            left_knee = pose.get('left_knee', {'y': 0})
            right_knee = pose.get('right_knee', {'y': 0})
            left_wrist = pose.get('left_wrist', {'y': 0})
            right_wrist = pose.get('right_wrist', {'y': 0})
            left_hip = pose.get('left_hip', {'y': 0})
            right_hip = pose.get('right_hip', {'y': 0})
            left_ankle = pose.get('left_ankle', {'y': 0})
            right_ankle = pose.get('right_ankle', {'y': 0})
            
            # Calculate average values using original coordinates
            knee_y = (left_knee['y'] + right_knee['y']) / 2
            hip_y = (left_hip['y'] + right_hip['y']) / 2
            ankle_y = (left_ankle['y'] + right_ankle['y']) / 2
            
            # Get ball position for wrist selection
            ball_info = None
            if i < len(self.ball_data):
                ball_frame_data = self.ball_data[i]
                if isinstance(ball_frame_data, dict) and ball_frame_data.get('ball_detections'):
                    ball_detections = ball_frame_data['ball_detections']
                    if ball_detections and isinstance(ball_detections[0], dict):
                        ball_info = ball_detections[0]
            
            ball_y = ball_info.get('center_y', 0) if ball_info else 0
            
            # Select the closest wrist to the ball
            left_wrist_x = left_wrist.get('x', 0)
            left_wrist_y = left_wrist['y']
            right_wrist_x = right_wrist.get('x', 0)
            right_wrist_y = right_wrist['y']
            
            # Calculate Euclidean distances to ball
            ball_x = ball_info.get('center_x', 0) if ball_info else 0
            left_distance = ((ball_x - left_wrist_x)**2 + (ball_y - left_wrist_y)**2)**0.5 if ball_info else float('inf')
            right_distance = ((ball_x - right_wrist_x)**2 + (ball_y - right_wrist_y)**2)**0.5 if ball_info else float('inf')
            
            # Use the wrist closer to the ball
            if left_distance <= right_distance:
                wrist_x = left_wrist_x
                wrist_y = left_wrist_y
                selected_wrist = "left"
            else:
                wrist_x = right_wrist_x
                wrist_y = right_wrist_y
                selected_wrist = "right"
            
            # Calculate change amounts compared to previous frames
            if i > 0:
                prev_frame = self.pose_data[i-1]
                prev_pose = prev_frame.get('pose', {})
                
                prev_knee_y = (prev_pose.get('left_knee', {'y': 0})['y'] + 
                              prev_pose.get('right_knee', {'y': 0})['y']) / 2
                prev_hip_y = (prev_pose.get('left_hip', {'y': 0})['y'] + 
                             prev_pose.get('right_hip', {'y': 0})['y']) / 2
                
                # Get previous ball position for wrist selection
                prev_ball_info = None
                if i-1 < len(self.ball_data):
                    prev_ball_frame_data = self.ball_data[i-1]
                    if isinstance(prev_ball_frame_data, dict) and prev_ball_frame_data.get('ball_detections'):
                        prev_ball_detections = prev_ball_frame_data['ball_detections']
                        if prev_ball_detections and isinstance(prev_ball_detections[0], dict):
                            prev_ball_info = prev_ball_detections[0]
                
                prev_ball_y = prev_ball_info.get('center_y', 0) if prev_ball_info else 0
                
                # Select the closest wrist to the ball in previous frame
                prev_left_wrist_x = prev_pose.get('left_wrist', {'x': 0, 'y': 0})['x']
                prev_left_wrist_y = prev_pose.get('left_wrist', {'x': 0, 'y': 0})['y']
                prev_right_wrist_x = prev_pose.get('right_wrist', {'x': 0, 'y': 0})['x']
                prev_right_wrist_y = prev_pose.get('right_wrist', {'x': 0, 'y': 0})['y']
                
                # Calculate Euclidean distances to ball in previous frame
                prev_ball_x = prev_ball_info.get('center_x', 0) if prev_ball_info else 0
                prev_left_distance = ((prev_ball_x - prev_left_wrist_x)**2 + (prev_ball_y - prev_left_wrist_y)**2)**0.5 if prev_ball_info else float('inf')
                prev_right_distance = ((prev_ball_x - prev_right_wrist_x)**2 + (prev_ball_y - prev_right_wrist_y)**2)**0.5 if prev_ball_info else float('inf')
                
                # Use the wrist closer to the ball in previous frame
                if prev_left_distance <= prev_right_distance:
                    prev_wrist_x = prev_left_wrist_x
                    prev_wrist_y = prev_left_wrist_y
                else:
                    prev_wrist_x = prev_right_wrist_x
                    prev_wrist_y = prev_right_wrist_y
                
                d_knee_y = knee_y - prev_knee_y
                d_wrist_y = wrist_y - prev_wrist_y
                d_hip_y = hip_y - prev_hip_y
            else:
                d_knee_y = d_wrist_y = d_hip_y = 0
            
            # Check if current phase transitions to next phase
            next_phase = self._check_phase_transition_original(current_phase, i, knee_y, wrist_y, hip_y, ankle_y, 
                                                    d_knee_y, d_wrist_y, d_hip_y)
            
            # Check minimum phase duration (except for Release and later phases)
            if next_phase != current_phase:
                # Apply minimum duration only for phases before Rising
                if current_phase in ["General", "Set-up", "Loading"]:
                    # Need minimum duration for early phases
                    if (i - phase_start_frame) >= min_phase_duration:
                        if self._is_trend_based_transition(i, current_phase, next_phase, noise_threshold):
                            # Record current phase in history before changing
                            if current_phase != "General":
                                phase_history.append((current_phase, current_phase_start, i))
                            
                            current_phase = next_phase
                            phase_start_frame = i
                            current_phase_start = i
                            print(f"Frame {i}: {current_phase} phase started")
                        else:
                            if i % 10 == 0:
                                print(f"Frame {i}: Noise detected, {current_phase} maintained")
                else:
                    # No minimum duration for Release and later phases
                    if self._is_trend_based_transition(i, current_phase, next_phase, noise_threshold):
                        # Record current phase in history before changing
                        if current_phase != "General":
                            phase_history.append((current_phase, current_phase_start, i))
                        
                        current_phase = next_phase
                        phase_start_frame = i
                        current_phase_start = i
                        print(f"Frame {i}: {current_phase} phase started")
                    else:
                        if i % 10 == 0:
                            print(f"Frame {i}: Noise detected, {current_phase} maintained")
            
            self.phases.append(current_phase)
        
        # Final trend-based organization
        self._finalize_phases_by_trend(noise_threshold)
        
        # Process cancellations: Replace cancelled phases with Set-up
        self._process_cancellations()
        
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
    
    def _is_cancellation_condition(self, current_phase: str, frame_idx: int, knee_y: float, 
                               wrist_y: float, hip_y: float, ankle_y: float,
                                 d_knee_y: float, d_wrist_y: float, d_hip_y: float) -> bool:
        """Check if current phase should be cancelled and return to Set-up"""
        
        # Get ball data
        ball_info = None
        if frame_idx < len(self.ball_data):
            ball_frame_data = self.ball_data[frame_idx]
            if isinstance(ball_frame_data, dict) and ball_frame_data.get('ball_detections'):
                ball_detections = ball_frame_data['ball_detections']
                if ball_detections and isinstance(ball_detections[0], dict):
                    ball_info = ball_detections[0]
        
        ball_x = ball_info.get('center_x', 0) if ball_info else 0
        ball_y = ball_info.get('center_y', 0) if ball_info else 0
        ball_detected = ball_info is not None
        
        # Get pose data
        pose = self.pose_data[frame_idx].get('pose', {}) if frame_idx < len(self.pose_data) else {}
        
        # Extract keypoints
        left_shoulder = pose.get('left_shoulder', {'x': 0, 'y': 0})
        right_shoulder = pose.get('right_shoulder', {'x': 0, 'y': 0})
        left_elbow = pose.get('left_elbow', {'x': 0, 'y': 0})
        right_elbow = pose.get('right_elbow', {'x': 0, 'y': 0})
        left_wrist = pose.get('left_wrist', {'x': 0, 'y': 0})
        right_wrist = pose.get('right_wrist', {'x': 0, 'y': 0})
        
        # Calculate shoulder position
        left_shoulder_y = left_shoulder.get('y', 0)
        right_shoulder_y = right_shoulder.get('y', 0)
        shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        
        # Calculate wrist position (closest to ball)
        left_wrist_x = left_wrist.get('x', 0)
        left_wrist_y = left_wrist.get('y', 0)
        right_wrist_x = right_wrist.get('x', 0)
        right_wrist_y = right_wrist.get('y', 0)
        
        # Calculate Euclidean distances to ball
        left_distance = ((ball_x - left_wrist_x)**2 + (ball_y - left_wrist_y)**2)**0.5 if ball_detected else float('inf')
        right_distance = ((ball_x - right_wrist_x)**2 + (ball_y - right_wrist_y)**2)**0.5 if ball_detected else float('inf')
        
        # Use the wrist closer to the ball
        if left_distance <= right_distance:
            wrist_x = left_wrist_x
            wrist_y = left_wrist_y
        else:
            wrist_x = right_wrist_x
            wrist_y = right_wrist_y
        
        # Calculate Euclidean distance between ball and wrist
        ball_wrist_distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5 if ball_detected else float('inf')
        
        # Check cancellation conditions based on current phase
        if current_phase == "Loading":
            # Loading cancellation: Ball missed
            if ball_detected:
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                close_threshold = ball_radius * 1.3
                
                if ball_wrist_distance > close_threshold:
                    if frame_idx % 10 == 0:
                        print(f"Frame {frame_idx}: Loading→Set-up: Ball missed (cancellation)")
                    return True
        
        elif current_phase == "Rising":
            # Rising cancellation: Hand moving down relative to hip
            if frame_idx > 0:
                prev_pose = self.pose_data[frame_idx-1].get('pose', {})
                prev_left_hip = prev_pose.get('left_hip', {'y': 0})
                prev_right_hip = prev_pose.get('right_hip', {'y': 0})
                prev_left_elbow = prev_pose.get('left_elbow', {'y': 0})
                prev_right_elbow = prev_pose.get('right_elbow', {'y': 0})
                prev_hip_y = (prev_left_hip['y'] + prev_right_hip['y']) / 2
                prev_elbow_y = (prev_left_elbow['y'] + prev_right_elbow['y']) / 2
                
                # Calculate elbow position for current frame
                elbow_y = (left_elbow['y'] + right_elbow['y']) / 2
                
                # Calculate relative movement (compared to hip)
                d_wrist_relative = d_wrist_y - (hip_y - prev_hip_y)
                d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                d_ball_relative = 0
                if ball_detected and frame_idx > 0:
                    prev_ball_info = None
                    if frame_idx-1 < len(self.ball_data):
                        prev_ball_frame_data = self.ball_data[frame_idx-1]
                        if isinstance(prev_ball_frame_data, dict) and prev_ball_frame_data.get('ball_detections'):
                            prev_ball_detections = prev_ball_frame_data['ball_detections']
                            if prev_ball_detections and isinstance(prev_ball_detections[0], dict):
                                prev_ball_info = prev_ball_detections[0]
                    if prev_ball_info:
                        prev_ball_y = prev_ball_info.get('center_y', 0)
                        d_ball_relative = ball_y - prev_ball_y - (hip_y - prev_hip_y)
                
                wrist_moving_down_relative = d_wrist_relative > 2.0  # 손목이 엉덩이 기준으로 아래로 이동
                elbow_moving_down_relative = d_elbow_relative > 2.0  # 팔꿈치가 엉덩이 기준으로 아래로 이동
                
                # Rising cancellation: Hand moving down relative to hip
                if ball_detected:
                    # 공이 감지되는 상황: 공, 손목, 팔꿈치가 모두 엉덩이 기준으로 아래로 이동하면 Set-up으로
                    ball_moving_down_relative = d_ball_relative > 2.0  # 공이 엉덩이 기준으로 아래로 이동
                    
                    if wrist_moving_down_relative and elbow_moving_down_relative and ball_moving_down_relative:
                        if frame_idx % 10 == 0:
                            print(f"Frame {frame_idx}: Rising→Set-up: All moving down relative to hip (cancellation)")
                        return "Set-up"
                else:
                    # 공이 감지되지 않는 상황: 손목, 팔꿈치가 엉덩이 기준으로 아래로 이동하면 Set-up으로
                    if wrist_moving_down_relative and elbow_moving_down_relative:
                        if frame_idx % 10 == 0:
                            print(f"Frame {frame_idx}: Rising→Set-up: Hand moving down relative to hip (cancellation)")
                        return "Set-up"
        
        elif current_phase == "Release":
            # Release cancellation: Ball released but improper form
            if ball_detected:
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                close_threshold = ball_radius * 1.3
                
                distance = abs(ball_y - wrist_y)
                ball_released = distance > close_threshold
                
                if ball_released:
                    # Calculate angles
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
                    
                    wrist_above_shoulder = wrist_y < shoulder_y
                    ball_above_shoulder = ball_y < shoulder_y
                    
                    # Improper form: return to General (which will be converted to Set-up)
                    if not ((left_angle >= 130 or right_angle >= 130) and wrist_above_shoulder and ball_above_shoulder):
                        if frame_idx % 10 == 0:
                            print(f"Frame {frame_idx}: Release→Set-up: Improper form (cancellation)")
                        return True
         
        return False
    
    def _process_cancellations(self):
        """Process cancellations by replacing cancelled phases with Set-up"""
        if not self.phases:
            return
        
        print("\n🔄 Processing cancellations...")
        
        # Find cancellation points (transitions to Set-up from other phases)
        cancellation_points = []
        for i in range(1, len(self.phases)):
            if self.phases[i] == "Set-up" and self.phases[i-1] in ["Loading", "Rising", "Release"]:
                cancellation_points.append(i)
        
        if not cancellation_points:
            print("  No cancellations found.")
            return
        
        print(f"  Found {len(cancellation_points)} cancellation points.")
        
        # Process each cancellation point
        for cancel_point in cancellation_points:
            # Find the start of the cancelled sequence (look backwards for consecutive cancelled phases)
            start_point = cancel_point - 1
            while start_point >= 0 and self.phases[start_point] in ["Loading", "Rising", "Release"]:
                start_point -= 1
            
            # Replace the entire cancelled sequence with Set-up
            for i in range(start_point + 1, cancel_point):
                if self.phases[i] in ["Loading", "Rising", "Release"]:
                    self.phases[i] = "Set-up"
                    print(f"    Frame {i}: {self.phases[i]} → Set-up (cancelled)")
        
        # Additional processing: Handle multiple consecutive cancellations
        # If there are multiple cancellation points close to each other, fill gaps with Set-up
        if len(cancellation_points) > 1:
            for i in range(len(cancellation_points) - 1):
                current_cancel = cancellation_points[i]
                next_cancel = cancellation_points[i + 1]
                
                # If there's a gap between cancellation points, fill with Set-up
                if next_cancel - current_cancel > 1:
                    for j in range(current_cancel + 1, next_cancel):
                        if self.phases[j] not in ["Set-up", "General"]:
                            self.phases[j] = "Set-up"
                            print(f"    Frame {j}: {self.phases[j]} → Set-up (gap fill)")
        
        print("  Cancellation processing completed.")
    
    def _check_phase_transition_original(self, current_phase: str, frame_idx: int, knee_y: float, 
                                       wrist_y: float, hip_y: float, ankle_y: float,
                                       d_knee_y: float, d_wrist_y: float, d_hip_y: float) -> str:
        """Check phase transition conditions using original data"""
        
        # Setup for noise filtering with dynamic frame calculation
        if self.video_fps is not None:
            # Calculate minimum phase duration based on FPS (100ms minimum duration)
            min_phase_duration = self.calculate_dynamic_min_frames(self.video_fps, 100.0)
        else:
            # Fallback to fixed duration if FPS is not available
            min_phase_duration = 3
        
        # Check for cancellation conditions first
        if self._is_cancellation_condition(current_phase, frame_idx, knee_y, wrist_y, hip_y, ankle_y, 
                                         d_knee_y, d_wrist_y, d_hip_y):
            return "Set-up"  # Always return to Set-up for cancellations
        
        # Get ball data from original data
        ball_info = None
        if frame_idx < len(self.ball_data):
            ball_frame_data = self.ball_data[frame_idx]
            if isinstance(ball_frame_data, dict) and ball_frame_data.get('ball_detections'):
                ball_detections = ball_frame_data['ball_detections']
                if ball_detections and isinstance(ball_detections[0], dict):
                    ball_info = ball_detections[0]
        
        # Previous frame ball data
        prev_ball_info = None
        if frame_idx > 0 and frame_idx < len(self.ball_data):
            prev_ball_frame_data = self.ball_data[frame_idx-1]
            if isinstance(prev_ball_frame_data, dict) and prev_ball_frame_data.get('ball_detections'):
                prev_ball_detections = prev_ball_frame_data['ball_detections']
                if prev_ball_detections and isinstance(prev_ball_detections[0], dict):
                    prev_ball_info = prev_ball_detections[0]
        
        # Extract ball-related information from original coordinates
        ball_x = ball_info.get('center_x', 0) if ball_info else 0
        ball_y = ball_info.get('center_y', 0) if ball_info else 0
        ball_detected = ball_info is not None
        
        # Calculate ball change amount compared to previous frame
        d_ball_y = 0
        if prev_ball_info:
            prev_ball_y = prev_ball_info.get('center_y', 0)
            d_ball_y = ball_y - prev_ball_y
        
        # Get pose data for current frame from original data
        pose = self.pose_data[frame_idx].get('pose', {}) if frame_idx < len(self.pose_data) else {}
        
        # Extract keypoints
        left_shoulder = pose.get('left_shoulder', {'x': 0, 'y': 0})
        right_shoulder = pose.get('right_shoulder', {'x': 0, 'y': 0})
        left_elbow = pose.get('left_elbow', {'x': 0, 'y': 0})
        right_elbow = pose.get('right_elbow', {'x': 0, 'y': 0})
        left_wrist = pose.get('left_wrist', {'x': 0, 'y': 0})
        right_wrist = pose.get('right_wrist', {'x': 0, 'y': 0})
        
        # Calculate shoulder position
        left_shoulder_y = left_shoulder.get('y', 0)
        right_shoulder_y = right_shoulder.get('y', 0)
        shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        
        # Calculate elbow angles
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
        
        # Calculate wrist position (closest to ball)
        left_wrist_x = left_wrist.get('x', 0)
        left_wrist_y = left_wrist.get('y', 0)
        right_wrist_x = right_wrist.get('x', 0)
        right_wrist_y = right_wrist.get('y', 0)
        
        # Calculate Euclidean distances to ball
        left_distance = ((ball_x - left_wrist_x)**2 + (ball_y - left_wrist_y)**2)**0.5 if ball_detected else float('inf')
        right_distance = ((ball_x - right_wrist_x)**2 + (ball_y - right_wrist_y)**2)**0.5 if ball_detected else float('inf')
        
        # Use the wrist closer to the ball
        if left_distance <= right_distance:
            wrist_x = left_wrist_x
            wrist_y = left_wrist_y
        else:
            wrist_x = right_wrist_x
            wrist_y = right_wrist_y
        
        # Calculate Euclidean distance between ball and wrist (original pixel coordinates)
        ball_wrist_distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5 if ball_detected else float('inf')
        
        # 1. General → Set-up: The ball is held in hand based on distance only
        if current_phase == "General":
            # Debug: Always print General phase info
            if frame_idx % 5 == 0:  # Print every 5 frames for debugging
                print(f"Frame {frame_idx}: General phase - ball_detected={ball_detected}, ball_y={ball_y:.1f}, wrist_y={wrist_y:.1f}")
            
            # Check current frame ball-hand distance
            if ball_detected:
                # Calculate ball radius from width and height (original pixel coordinates)
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                
                # Dynamic threshold based on ball radius - Close contact only
                # Close contact: 1.3 * ball radius (tight grip) - pixel units
                close_threshold = ball_radius * 1.3
                
                distance = ((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)**0.5
                
                # Debug: Print distance info
                if frame_idx % 5 == 0:
                    print(f"Frame {frame_idx}: Distance={distance:.1f}, ball_radius={ball_radius:.1f}")
                    print(f"  Threshold: close={close_threshold:.1f}")
                
                if distance < close_threshold:
                    print(f"Frame {frame_idx}: General→Set-up: Close contact (distance={distance:.1f}, threshold={close_threshold:.1f})")
                    return "Set-up"
                else:
                    if frame_idx % 5 == 0:
                        print(f"Frame {frame_idx}: Distance too far ({distance:.1f} > {close_threshold:.1f})")
            else:
                if frame_idx % 5 == 0:
                    print(f"Frame {frame_idx}: Ball not detected in General phase")
        
        # 2. Set-up → Loading: Hip AND shoulder are moving downward
        if current_phase == "Set-up":
            conditions = []
            
            # Calculate hip and shoulder positions
            left_hip = pose.get('left_hip', {'y': 0})
            right_hip = pose.get('right_hip', {'y': 0})
            left_shoulder = pose.get('left_shoulder', {'y': 0})
            right_shoulder = pose.get('right_shoulder', {'y': 0})
            
            hip_y = (left_hip['y'] + right_hip['y']) / 2
            shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            
            # Calculate hip and shoulder changes from previous frame
            if frame_idx > 0:
                prev_pose = self.pose_data[frame_idx-1].get('pose', {})
                prev_left_hip = prev_pose.get('left_hip', {'y': 0})
                prev_right_hip = prev_pose.get('right_hip', {'y': 0})
                prev_left_shoulder = prev_pose.get('left_shoulder', {'y': 0})
                prev_right_shoulder = prev_pose.get('right_shoulder', {'y': 0})
                
                prev_hip_y = (prev_left_hip['y'] + prev_right_hip['y']) / 2
                prev_shoulder_y = (prev_left_shoulder['y'] + prev_right_shoulder['y']) / 2
                
                d_hip_y = hip_y - prev_hip_y
                d_shoulder_y = shoulder_y - prev_shoulder_y
            else:
                d_hip_y = d_shoulder_y = 0
            
            # Hip moving downward (y-coordinate increasing) - pixel units
            if d_hip_y > 2.0:  # hip_y increasing means moving down (pixel threshold)
                conditions.append("hip_down")
            
            # Shoulder moving downward - pixel units
            if d_shoulder_y > 2.0:  # shoulder_y increasing means moving down (pixel threshold)
                conditions.append("shoulder_down")
            
            # BOTH hip AND shoulder must be moving down
            if len(conditions) == 2:
                if frame_idx % 10 == 0:
                    print(f"Frame {frame_idx}: Set-up→Loading conditions: {conditions}")
                return "Loading"
        
        # 3. Loading → Rising: Wrist, elbow가 모두 위로 움직이면 Rising으로 전이 (공 조건 제외)
        if current_phase == "Loading":
            conditions = []
            
            # Calculate hip position for relative movement
            left_hip = pose.get('left_hip', {'y': 0})
            right_hip = pose.get('right_hip', {'y': 0})
            hip_y = (left_hip['y'] + right_hip['y']) / 2
            
            # Calculate elbow change from original data
            if frame_idx > 0:
                prev_pose = self.pose_data[frame_idx-1].get('pose', {})
                prev_left_elbow = prev_pose.get('left_elbow', {'y': 0})
                prev_right_elbow = prev_pose.get('right_elbow', {'y': 0})
                prev_hip = prev_pose.get('left_hip', {'y': 0})
                prev_right_hip = prev_pose.get('right_hip', {'y': 0})
                
                prev_elbow_y = (prev_left_elbow['y'] + prev_right_elbow['y']) / 2
                prev_hip_y = (prev_hip['y'] + prev_right_hip['y']) / 2
                elbow_y = (left_elbow['y'] + right_elbow['y']) / 2
                
                # Calculate relative movement (compared to hip)
                d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                d_wrist_relative = d_wrist_y - (hip_y - prev_hip_y)
                # d_ball_relative = d_ball_y - (hip_y - prev_hip_y) if ball_detected else 0
                
                # Wrist moving upward relative to hip (y decreasing) - pixel units
                if d_wrist_relative < -2.0:
                    conditions.append("wrist_up_relative")
                
                # Elbow moving upward relative to hip - pixel units
                if d_elbow_relative < -2.0:
                    conditions.append("elbow_up_relative")
                
                # 공(ball) 조건은 제외
                # if ball_detected and d_ball_relative < -2.0:
                #     conditions.append("ball_up_relative")
                
                # 손목, 팔꿈치 둘 다 만족하면 Rising
                if len(conditions) == 2:
                    if frame_idx % 10 == 0:
                        print(f"Frame {frame_idx}: Loading→Rising conditions: {conditions}")
                    return "Rising"
        
        # 3.5. Set-up → Rising: Skip Loading phase if Rising conditions are met directly (relative to hip)
        if current_phase == "Set-up":
            conditions = []
            
            # Calculate hip position for relative movement
            left_hip = pose.get('left_hip', {'y': 0})
            right_hip = pose.get('right_hip', {'y': 0})
            hip_y = (left_hip['y'] + right_hip['y']) / 2
            
            # Calculate elbow change from original data
            if frame_idx > 0:
                prev_pose = self.pose_data[frame_idx-1].get('pose', {})
                prev_left_elbow = prev_pose.get('left_elbow', {'y': 0})
                prev_right_elbow = prev_pose.get('right_elbow', {'y': 0})
                prev_hip = prev_pose.get('left_hip', {'y': 0})
                prev_right_hip = prev_pose.get('right_hip', {'y': 0})
                
                prev_elbow_y = (prev_left_elbow['y'] + prev_right_elbow['y']) / 2
                prev_hip_y = (prev_hip['y'] + prev_right_hip['y']) / 2
                elbow_y = (left_elbow['y'] + right_elbow['y']) / 2
                d_elbow_y = elbow_y - prev_elbow_y
                
                # Calculate relative movement (compared to hip)
                d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                d_wrist_relative = d_wrist_y - (hip_y - prev_hip_y)
                d_ball_relative = d_ball_y - (hip_y - prev_hip_y) if ball_detected else 0
                
                # Wrist moving upward relative to hip (y decreasing) - pixel units
                if d_wrist_relative < -2.0:
                    conditions.append("wrist_up_relative")
                
                # Elbow moving upward relative to hip - pixel units
                if d_elbow_relative < -2.0:
                    conditions.append("elbow_up_relative")
                
                # Ball moving upward relative to hip - pixel units
                if ball_detected and d_ball_relative < -2.0:
                    conditions.append("ball_up_relative")
                
                # All three conditions must be met to skip Loading and go directly to Rising
                if len(conditions) == 3:
                    if frame_idx % 10 == 0:
                        print(f"Frame {frame_idx}: Set-up→Rising (skip Loading) conditions: {conditions}")
                    return "Rising"
        
        if current_phase == "Set-up" or current_phase == "Loading" :
            if ball_detected:
                # Calculate ball radius from width and height (original pixel coordinates)
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                
                # Dynamic threshold based on ball radius - Close contact only
                # Close contact: 1.3 * ball radius (tight grip) - pixel units
                close_threshold = ball_radius * 1.3
                
                distance = abs(ball_y - wrist_y)
                
                # Debug: Print distance info
                if frame_idx % 5 == 0:
                    print(f"Frame {frame_idx}: Distance={distance:.1f}, ball_radius={ball_radius:.1f}")
                    print(f"  Threshold: close={close_threshold:.1f}")
                
                if distance > close_threshold:
                    print(f"Frame {frame_idx}: Missed Ball: Close contact (distance={distance:.1f}, threshold={close_threshold:.1f})")
                    # Check minimum frame duration for General transition
                    if frame_idx >= min_phase_duration:
                        return "General"
                    else:
                        return current_phase


        # 4. Rising → Release: Ball is released with proper form
        if current_phase == "Rising":
            # Check for cancellation first (Rising → Set-up)
            # Calculate relative movement compared to hip
            if frame_idx > 0:
                prev_pose = self.pose_data[frame_idx-1].get('pose', {})
                prev_left_hip = prev_pose.get('left_hip', {'y': 0})
                prev_right_hip = prev_pose.get('right_hip', {'y': 0})
                prev_left_elbow = prev_pose.get('left_elbow', {'y': 0})
                prev_right_elbow = prev_pose.get('right_elbow', {'y': 0})
                prev_hip_y = (prev_left_hip['y'] + prev_right_hip['y']) / 2
                prev_elbow_y = (prev_left_elbow['y'] + prev_right_elbow['y']) / 2
                
                # Calculate elbow position for current frame
                elbow_y = (left_elbow['y'] + right_elbow['y']) / 2
                
                # Calculate relative movement (compared to hip)
                d_wrist_relative = d_wrist_y - (hip_y - prev_hip_y)
                d_elbow_relative = (elbow_y - prev_elbow_y) - (hip_y - prev_hip_y)
                d_ball_relative = d_ball_y - (hip_y - prev_hip_y) if ball_detected else 0
                
                wrist_moving_down_relative = d_wrist_relative > 2.0  # 손목이 엉덩이 기준으로 아래로 이동
                elbow_moving_down_relative = d_elbow_relative > 2.0  # 팔꿈치가 엉덩이 기준으로 아래로 이동
                
                # Rising cancellation: Hand moving down relative to hip
                if ball_detected:
                    # 공이 감지되는 상황: 공, 손목, 팔꿈치가 모두 엉덩이 기준으로 아래로 이동하면 Set-up으로
                    ball_moving_down_relative = d_ball_relative > 2.0  # 공이 엉덩이 기준으로 아래로 이동
                    
                    if wrist_moving_down_relative and elbow_moving_down_relative and ball_moving_down_relative:
                        if frame_idx % 10 == 0:
                            print(f"Frame {frame_idx}: Rising→Set-up: All moving down relative to hip (cancellation)")
                        return "Set-up"
                else:
                    # 공이 감지되지 않는 상황: 손목, 팔꿈치가 엉덩이 기준으로 아래로 이동하면 Set-up으로
                    if wrist_moving_down_relative and elbow_moving_down_relative:
                        if frame_idx % 10 == 0:
                            print(f"Frame {frame_idx}: Rising→Set-up: Hand moving down relative to hip (cancellation)")
                        return "Set-up"
            
            # Normal Rising → Release transition
            if ball_detected:
                # Calculate ball radius from width and height (original pixel coordinates)
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                
                # Dynamic threshold based on ball radius - Close contact only
                # Close contact: 1.3 * ball radius (tight grip) - pixel units
                close_threshold = ball_radius * 1.3
                
                distance = abs(ball_y - wrist_y)
                
                # Check if wrist is above shoulder
                wrist_above_shoulder = wrist_y < shoulder_y
                
                # Debug: Print distance info
                if frame_idx % 5 == 0:
                    print(f"Frame {frame_idx}: Distance={distance:.1f}, ball_radius={ball_radius:.1f}")
                    print(f"  Threshold: close={close_threshold:.1f}")
                    print(f"  Wrist above shoulder: {wrist_above_shoulder}")
                
                # Check if ball is released (distance > threshold)
                ball_released = distance > close_threshold
                
                # Enhanced Release conditions
                if ball_released:
                    # Ball is released - check for proper shooting form
                    # Check if ball is above shoulder
                    ball_above_shoulder = ball_y < shoulder_y
                    
                    if (left_angle >= 110 or right_angle >= 110) and wrist_above_shoulder and ball_above_shoulder:
                        if frame_idx % 10 == 0:
                            print(f"Frame {frame_idx}: Rising→Release: Proper release (angle={max(left_angle, right_angle):.1f}, wrist_above_shoulder={wrist_above_shoulder}, ball_above_shoulder={ball_above_shoulder})")
                        return "Release"
                    else:
                        # Ball released but improper form - return to Set-up (not General)
                        if frame_idx % 10 == 0:
                            print(f"Frame {frame_idx}: Rising→Set-up: Ball released but improper form (cancellation)")
                        return "Set-up"
                else:
                    # Ball still in hand - check for normal release conditions
                    # Check if ball is above shoulder
                    ball_above_shoulder = ball_y < shoulder_y
                    
                    if (left_angle >= 110 or right_angle >= 110) and distance > close_threshold and ball_above_shoulder:
                        if frame_idx % 10 == 0:
                            print(f"Frame {frame_idx}: Rising→Release: Normal release (angle={max(left_angle, right_angle):.1f}, ball_above_shoulder={ball_above_shoulder})")
                        return "Release"
        
        # 5. Release → Follow-through: Ball has fully left the hand
        if current_phase == "Release":
            conditions = []
            
            # Ball has fully left the hand (distance > threshold)
            if ball_detected:
                # Dynamic threshold based on ball radius with multiple levels
                ball_info = self.normalized_data[frame_idx].get('normalized_ball', {})
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                close_threshold = ball_radius * 1.5  # Ball still near hand
                medium_threshold = ball_radius * 2.5  # Ball moderately away
                far_threshold = ball_radius * 4.0     # Ball clearly left hand
                
                if ball_wrist_distance > far_threshold:
                    conditions.append("ball_clearly_left_hand")
                elif ball_wrist_distance > medium_threshold:
                    conditions.append("ball_moderately_away")
                elif ball_wrist_distance > close_threshold:
                    conditions.append("ball_slightly_away")
            
            # Any ball distance condition is met
            if len(conditions) >= 1:
                if frame_idx % 10 == 0:
                    print(f"Frame {frame_idx}: Release→Follow-through conditions: {conditions}")
                return "Follow-through"
        
        # 6. Follow-through → General: Wrist below eyes relative to hip + Ball caught check
        if current_phase == "Follow-through":
            # Check if ball is caught (return to Set-up)
            if ball_detected:
                # Calculate ball radius and threshold
                ball_info = self.normalized_data[frame_idx].get('normalized_ball', {})
                ball_width = ball_info.get('width', 0)
                ball_height = ball_info.get('height', 0)
                ball_radius = (ball_width + ball_height) / 4
                close_threshold = ball_radius * 1.3
                
                if ball_wrist_distance <= close_threshold:
                    if frame_idx % 10 == 0:
                        print(f"Frame {frame_idx}: Follow-through→Set-up: Ball caught (distance={ball_wrist_distance:.1f})")
                    return "Set-up"
            
            # Check if wrist is below eyes relative to hip
            if frame_idx > 0:
                # Get eye positions (use lowest eye)
                left_eye = pose.get('left_eye', {'y': 0})
                right_eye = pose.get('right_eye', {'y': 0})
                eye_y = max(left_eye.get('y', 0), right_eye.get('y', 0))  # Lowest eye
                
                # Get wrist positions (use highest wrist)
                left_wrist = pose.get('left_wrist', {'y': 0})
                right_wrist = pose.get('right_wrist', {'y': 0})
                wrist_y = min(left_wrist.get('y', 0), right_wrist.get('y', 0))  # Highest wrist
                
                # Get hip position
                left_hip = pose.get('left_hip', {'y': 0})
                right_hip = pose.get('right_hip', {'y': 0})
                hip_y = (left_hip.get('y', 0) + right_hip.get('y', 0)) / 2
                
                # Calculate relative positions to hip
                eye_relative_to_hip = eye_y - hip_y
                wrist_relative_to_hip = wrist_y - hip_y
                
                # Check if wrist is below eyes relative to hip
                if wrist_relative_to_hip > eye_relative_to_hip:
                    if frame_idx % 10 == 0:
                        print(f"Frame {frame_idx}: Follow-through→General: Wrist below eyes relative to hip (wrist_rel={wrist_relative_to_hip:.1f}, eye_rel={eye_relative_to_hip:.1f})")
                    # Check minimum frame duration for General transition
                    if frame_idx >= min_phase_duration:
                        return "General"
                    else:
                        return current_phase
        
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
            original_rim_data = self.rim_data
            normalized_ball_data = [frame['normalized_ball'] for frame in self.normalized_data]  # Normalized ball data
            print(original_rim_data)
            self.create_dual_analysis_video(
                video_path=video_path,
                output_path=output_video,
                original_pose_data=original_pose_data,
                normalized_pose_data=normalized_pose_data,
                original_ball_data=original_ball_data,
                normalized_ball_data=normalized_ball_data,
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
        
        # Draw arm angles
        self._draw_arm_angles_original(frame, pose, w, h)
        
        return frame

    def _draw_arm_angles_original(self, frame: np.ndarray, pose: Dict, w: int, h: int):
        font_scale = 1.0  # 글자 크기 조정
        thickness = 2      # 두께 줄임
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
                elbow_x = int(left_elbow['x'])
                elbow_y = int(left_elbow['y'])
                wrist_x = int(left_wrist['x'])
                wrist_y = int(left_wrist['y'])
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
                # 팔꿈치→손목 벡터 방향으로 40픽셀 이동 (거리 줄임)
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
                elbow_x = int(right_elbow['x'])
                elbow_y = int(right_elbow['y'])
                wrist_x = int(right_wrist['x'])
                wrist_y = int(right_wrist['y'])
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
                # 팔꿈치→손목 벡터 방향으로 60픽셀 이동 (오른쪽은 더 멀리)
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
    
    def _draw_rim_original(self, frame: np.ndarray, frame_idx: int, rim_data: List[Dict]) -> np.ndarray:
        """Draw original absolute coordinates rim"""
        if(frame_idx >= len(rim_data)):
            return frame
        # Check data structure and extract rim data
        frame_data = rim_data[frame_idx]
        if isinstance(frame_data, dict):
            rim_detections = frame_data.get('rim_detections', [])
        else:
            rim_detections = []
        for rim in rim_detections:
            if isinstance(rim, dict):
                # Use original absolute coordinates
                center_x = int(rim.get('center_x', 0))
                center_y = int(rim.get('center_y', 0))
                width = int(rim.get('width', 10))
                height = int(rim.get('height', 10))
                confidence = rim.get('confidence', 0)
                
                # Change color based on confidence
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green
                elif confidence > 0.4:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                # Draw rim
                cv2.rectangle(frame, 
                              (center_x - width // 2, center_y - height // 2), 
                              (center_x + width // 2, center_y + height // 2), 
                              color, 2)
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
        
        # Draw arm angles
        self._draw_arm_angles_normalized(frame, pose, w, h, center_x, center_y)
        
        return frame

    def _draw_arm_angles_normalized(self, frame: np.ndarray, pose: Dict, w: int, h: int, center_x: int, center_y: int):
        scale_factor = min(w, h) / 12
        font_scale = 1.0  # 글자 크기 조정
        thickness = 2      # 두께 줄임
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
                # 정규화 좌표를 화면 좌표로 변환
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
                # 팔꿈치→손목 벡터 방향으로 40픽셀 이동 (거리 줄임)
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
                # 팔꿈치→손목 벡터 방향으로 60픽셀 이동 (오른쪽은 더 멀리)
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