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

# Import shot detection module
from shot_detection.shot_detector import ShotDetector

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

        # Shot detection (replaces all shot tracking variables)
        self.shot_detector = ShotDetector()
        self.fallback_torso_length = None  # Fallback torso for non-shot frames
        self.shot_normalization_data = {}  # Shot-specific normalization data (direction, hip, torso)

        # Detectors (inject shot_detector into phase detectors)
        self.ball_detector = BallBasedPhaseDetector()
        self.torso_detector = TorsoBasedPhaseDetector()
        self.hybrid_fps_detector = HybridFPSPhaseDetector(shot_detector=self.shot_detector)
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
        """Normalize pose data using shot-based torso, direction, and hip coordinates"""
        print(f"\n🔄 STEP 4: Normalize data using shot-based measurements")
        print("=" * 50)
        
        if not self.pose_data:
            print("❌ Pose data not found.")
            return
        
        # Use selected_video if video_path is None
        if video_path is None:
            video_path = self.selected_video
        
        # Step 1: Check if we have shot-based data
        print("📊 Checking shot-based data...")
        has_shots = len(self.shot_detector.shots) > 0
        if has_shots:
            print(f"   ✅ Found {len(self.shot_detector.shots)} shots with individual measurements")
            for shot in self.shot_detector.shots:
                print(f"   📏 Shot {shot['shot_id']}: frames {shot['start_frame']}-{shot['end_frame']}, torso: {shot['fixed_torso']:.4f}")
        else:
            print("   ⚠️ No shot data found, using global measurements")
        
        # Step 2: Get fallback torso measurement for non-shot frames
        print("📊 Getting fallback torso measurement...")
        fallback_torso_length = self._get_fallback_torso_from_shots()
        
        # Store fallback torso for later use
        self.fallback_torso_length = fallback_torso_length
        
        # Step 3: Process each shot individually for direction and hip coordinates
        print("🎯 Processing each shot for individual direction and hip coordinates...")
        self.shot_normalization_data = {}  # Store shot-specific normalization info
        
        if has_shots:
            for shot in self.shot_detector.shots:
                shot_id = shot['shot_id']
                start_frame = shot['start_frame']
                end_frame = shot['end_frame']
                
                print(f"   🎯 Processing Shot {shot_id} (frames {start_frame}-{end_frame})...")
                
                # Get shot-specific direction and hip coordinates
                shot_facing_direction, shot_reference_hip_side = self._determine_facing_direction_for_shot(shot)
                shot_stable_hip_x, shot_stable_hip_y = self._calculate_stable_reference_hip_for_shot(shot, shot_reference_hip_side)
                
                # Check if hip coordinates are valid
                if shot_stable_hip_x is None or shot_stable_hip_y is None:
                    print(f"      ❌ Shot {shot_id}: Hip coordinates calculation failed, skipping normalization")
                    continue
                
                # Store shot-specific normalization data
                self.shot_normalization_data[shot_id] = {
                    'facing_direction': shot_facing_direction,
                    'reference_hip_side': shot_reference_hip_side,
                    'stable_hip_x': shot_stable_hip_x,
                    'stable_hip_y': shot_stable_hip_y,
                    'fixed_torso': shot['fixed_torso']
                }
                
                print(f"      ✅ Shot {shot_id}: facing {shot_facing_direction}, {shot_reference_hip_side} hip, torso {shot['fixed_torso']:.4f}")
                
                # Set facing direction for visualization (use first shot's direction)
                if shot_id == 1:
                    self.facing_direction = shot_facing_direction
        else:
            # Fallback: use global direction and hip coordinates
            print("   ⚠️ No shots available, using global direction and hip coordinates")
            global_facing_direction, global_reference_hip_side = self._determine_facing_direction()
            global_stable_hip_x, global_stable_hip_y = self._calculate_stable_reference_hip_from_phase_detection(global_reference_hip_side)
            
            # Check if global hip coordinates are valid
            if global_stable_hip_x is None or global_stable_hip_y is None:
                print("   ❌ Global hip coordinates calculation failed, normalization failed")
                return
            
            self.shot_normalization_data['global'] = {
                'facing_direction': global_facing_direction,
                'reference_hip_side': global_reference_hip_side,
                'stable_hip_x': global_stable_hip_x,
                'stable_hip_y': global_stable_hip_y,
                'fixed_torso': fallback_torso_length
            }
            
            # Set global facing direction for visualization
            self.facing_direction = global_facing_direction
        
        print(f"✅ Fallback torso length: {fallback_torso_length:.4f}")
        
        # Step 4: Normalize all frames using shot-specific values
        print("🔄 Normalizing all frames with shot-specific values...")
        self.normalized_data = []
        
        total_frames = len(self.pose_data)
        for i, frame_data in enumerate(self.pose_data):
            pose = frame_data.get('pose', {})
            
            # Step 1: Get shot information and normalization data for this frame
            current_shot = self._get_shot_for_frame(i)
            
            if current_shot and current_shot['shot_id'] in self.shot_normalization_data:
                # Use shot-specific normalization data
                shot_id = current_shot['shot_id']
                shot_norm_data = self.shot_normalization_data[shot_id]
                
                frame_torso_length = shot_norm_data['fixed_torso']
                facing_direction = shot_norm_data['facing_direction']
                reference_hip_side = shot_norm_data['reference_hip_side']
                stable_reference_hip_x = shot_norm_data['stable_hip_x']
                stable_reference_hip_y = shot_norm_data['stable_hip_y']
                
                # print(f"   Frame {i}: Using shot {shot_id} normalization (torso: {frame_torso_length:.4f}, facing: {facing_direction})")
            else:
                # Use fallback normalization data
                if 'global' in self.shot_normalization_data:
                    global_norm_data = self.shot_normalization_data['global']
                    frame_torso_length = global_norm_data['fixed_torso']
                    facing_direction = global_norm_data['facing_direction']
                    reference_hip_side = global_norm_data['reference_hip_side']
                    stable_reference_hip_x = global_norm_data['stable_hip_x']
                    stable_reference_hip_y = global_norm_data['stable_hip_y']
                else:
                    # Last resort fallback
                    frame_torso_length = fallback_torso_length
                    facing_direction = 'right'
                    reference_hip_side = 'right'
                    stable_reference_hip_x = 0.5
                    stable_reference_hip_y = 0.5
                
                # print(f"   Frame {i}: Using fallback normalization (torso: {frame_torso_length:.4f}, facing: {facing_direction})")
            
            # Step 2: Scale normalization using frame-specific torso length
            current_scaling_factor = frame_torso_length
            scale_normalized_pose = {}
            for key, kp in pose.items():
                if isinstance(kp, dict) and 'x' in kp and 'y' in kp:
                    # Scale normalization only
                    norm_x = kp['x'] / current_scaling_factor
                    norm_y = kp['y'] / current_scaling_factor
                    
                    scale_normalized_pose[key] = {
                        'x': norm_x,
                        'y': norm_y,
                        'confidence': kp.get('confidence', 0)
                    }
                # Missing keypoints are not added (automatically excluded)
            
            # Step 3: Coordinate normalization using shot-specific stable reference hip as origin (x=0)
            coordinate_normalized_pose = {}
            hip_center_valid = True  # Using stable values, so always valid
            
            for key, kp in scale_normalized_pose.items():
                if isinstance(kp, dict) and 'x' in kp and 'y' in kp:
                    # Position normalization: stable reference hip becomes (0,0)
                    norm_x = kp['x'] - (stable_reference_hip_x / current_scaling_factor)
                    norm_y = kp['y'] - (stable_reference_hip_y / current_scaling_factor)
                    
                    coordinate_normalized_pose[key] = {
                        'x': norm_x,
                        'y': norm_y,
                        'confidence': kp.get('confidence', 0)
                    }
            
            
            # Final pose with horizontal flip applied for left-facing shooters
            normalized_pose = {}
            for key, kp in coordinate_normalized_pose.items():
                if isinstance(kp, dict) and 'x' in kp and 'y' in kp:
                    norm_x = kp['x']
                    norm_y = kp['y']
                    
                    # Apply horizontal flip for left-facing shooters to make them face right
                    if facing_direction == 'left':
                        norm_x = -norm_x  # Flip horizontally
                    
                    normalized_pose[key] = {
                        'x': norm_x,
                        'y': norm_y,
                        'confidence': kp.get('confidence', 0)
                    }
            
            # Step 4: Normalize ball position with same shot-specific transformations
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
                        
                        # Apply same shot-specific transformations as pose: Scale → Coordinate → Direction
                        # 1. Scale normalization
                        norm_ball_x = ball_x / current_scaling_factor
                        norm_ball_y = ball_y / current_scaling_factor
                        
                        # 2. Coordinate normalization (shot-specific reference hip becomes (0,0))
                        norm_ball_x = norm_ball_x - (stable_reference_hip_x / current_scaling_factor)
                        norm_ball_y = norm_ball_y - (stable_reference_hip_y / current_scaling_factor)
                        
                        # 3. Direction normalization (horizontal flip for left-facing)
                        if facing_direction == 'left':
                            norm_ball_x = -norm_ball_x  # Flip horizontally
                        
                        normalized_ball = {
                            'center_x': norm_ball_x,
                            'center_y': norm_ball_y,
                            'width': ball.get('width', 0.01) / current_scaling_factor,
                            'height': ball.get('height', 0.01) / current_scaling_factor
                        }
                        ball_detected = True
            
            # Get shot information for this frame
            shot_id = None
            if i < len(self.shot_detector.frame_shots):
                shot_id = self.shot_detector.frame_shots[i]
            
            normalized_frame = {
                'frame_index': i,
                'normalized_pose': normalized_pose,
                'normalized_ball': normalized_ball,
                'stable_reference_hip': [stable_reference_hip_x, stable_reference_hip_y],
                'scaling_factor': current_scaling_factor,  # 실제 사용된 scaling factor
                'facing_direction': facing_direction,
                'reference_hip_side': reference_hip_side,
                'ball_detected': ball_detected,
                'hip_center_valid': hip_center_valid,
                'consecutive_missing_hip': 0,  # Always 0 since using stable values
                'shot': shot_id,
                'shot_normalization_applied': current_shot is not None  # Whether shot-specific normalization was applied
            }
            
            self.normalized_data.append(normalized_frame)
            
            # Show progress every 100 frames
            if (i + 1) % 100 == 0 or (i + 1) == total_frames:
                progress = ((i + 1) / total_frames) * 100
                print(f"   📊 Progress: {i + 1}/{total_frames} frames ({progress:.1f}%)")
        
        # Print statistics
        detected_frames = sum(1 for frame in self.normalized_data if frame['ball_detected'])
        total_frames = len(self.normalized_data)
        valid_hip_frames = sum(1 for frame in self.normalized_data if frame['hip_center_valid'])
        
        print(f"✅ Normalization completed: {len(self.normalized_data)} frames")
        print(f"Detected ball frames: {detected_frames}/{total_frames} ({detected_frames/total_frames*100:.1f}%)")
        print(f"Valid hip_center frames: {valid_hip_frames}/{total_frames} ({valid_hip_frames/total_frames*100:.1f}%)")
        
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
    
    def _calculate_stable_torso_length(self) -> float:
        """Calculate stable torso length using same method as phase detection (first 4 frames)"""
        confidence_threshold = 0.3  # Same as phase detection
        fps = getattr(self, 'video_fps', 30.0)
        required_frames = max(3, int(4 * (fps / 30.0)))  # 30fps 기준 4프레임
        
        print(f"   Calculating from first {required_frames} frames (FPS: {fps})")
        
        torso_values = []
        frames_used = []
        
        for i in range(min(required_frames, len(self.pose_data))):
            frame_data = self.pose_data[i]
            pose = frame_data.get('pose', {})
            
            # Get keypoints
            left_shoulder = pose.get('left_shoulder', {})
            right_shoulder = pose.get('right_shoulder', {})
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            
            valid_torso_lengths = []
            
            # Check left side torso (신뢰도 기반)
            if (isinstance(left_shoulder, dict) and isinstance(left_hip, dict) and
                'x' in left_shoulder and 'y' in left_shoulder and
                'x' in left_hip and 'y' in left_hip):
                
                left_shoulder_conf = left_shoulder.get('confidence', 1.0)
                left_hip_conf = left_hip.get('confidence', 1.0)
                left_avg_conf = (left_shoulder_conf + left_hip_conf) / 2
                
                if left_avg_conf >= confidence_threshold:
                    left_torso_length = ((left_shoulder['x'] - left_hip['x'])**2 + 
                                       (left_shoulder['y'] - left_hip['y'])**2)**0.5
                    if left_torso_length > 0:
                        valid_torso_lengths.append(left_torso_length)
                        print(f"   Frame {i}: Left torso {left_torso_length:.4f} (conf: {left_avg_conf:.3f}) ✓")
                else:
                    print(f"   Frame {i}: Left torso excluded (conf: {left_avg_conf:.3f} < {confidence_threshold}) ✗")
            
            # Check right side torso (신뢰도 기반)
            if (isinstance(right_shoulder, dict) and isinstance(right_hip, dict) and
                'x' in right_shoulder and 'y' in right_shoulder and
                'x' in right_hip and 'y' in right_hip):
                
                right_shoulder_conf = right_shoulder.get('confidence', 1.0)
                right_hip_conf = right_hip.get('confidence', 1.0)
                right_avg_conf = (right_shoulder_conf + right_hip_conf) / 2
                
                if right_avg_conf >= confidence_threshold:
                    right_torso_length = ((right_shoulder['x'] - right_hip['x'])**2 + 
                                        (right_shoulder['y'] - right_hip['y'])**2)**0.5
                    if right_torso_length > 0:
                        valid_torso_lengths.append(right_torso_length)
                        print(f"   Frame {i}: Right torso {right_torso_length:.4f} (conf: {right_avg_conf:.3f}) ✓")
                else:
                    print(f"   Frame {i}: Right torso excluded (conf: {right_avg_conf:.3f} < {confidence_threshold}) ✗")
            
            # Calculate frame torso (average of valid measurements)
            if len(valid_torso_lengths) > 0:
                frame_torso = np.mean(valid_torso_lengths)
                torso_values.append(frame_torso)
                frames_used.append(i)
                print(f"   Frame {i}: Final torso {frame_torso:.4f} (average of {len(valid_torso_lengths)} measurements)")
        
        if len(torso_values) >= 3:  # Minimum 3 frames
            stable_torso = np.mean(torso_values)
            print(f"   ✅ Stable torso: {stable_torso:.4f} (from frames {frames_used})")
            return stable_torso
        else:
            print(f"   ⚠️ Not enough valid torso measurements ({len(torso_values)}/3), using default")
            return 0.1  # Default fallback
    
    def _determine_facing_direction(self) -> tuple:
        """Determine facing direction from 4 frames before first phase transition"""
        if not hasattr(self, 'phase_detector') or not self.phase_detector:
            print("   ⚠️ Phase detector not available, using first 4 frames")
            return self._determine_facing_direction_from_start()
        
        # Get the frame index where first meaningful transition occurred
        first_transition_frame = getattr(self.phase_detector, 'first_transition_frame', None)
        print(f"   🔍 DEBUG: phase_detector.first_transition_frame = {first_transition_frame}")  # 중요한 디버깅 로그 유지
        
        if first_transition_frame is None:
            print("   ⚠️ No phase transition detected, using first 4 frames")
            return self._determine_facing_direction_from_start()
        
        fps = getattr(self, 'video_fps', 30.0)
        required_frames = max(3, int(4 * (fps / 30.0)))
        
        # Check if we have enough frames before the first transition
        if first_transition_frame < required_frames:
            print(f"   ⚠️ First transition too early (frame {first_transition_frame} < {required_frames} required), using fallback method")
            return self._determine_facing_direction_from_start()
        
        # Analyze frames immediately before the first transition (4 frames)
        start_frame = first_transition_frame - required_frames
        end_frame = first_transition_frame
        
        print(f"   Analyzing frames {start_frame}-{end_frame-1} (before first transition at frame {first_transition_frame})...")
        
        direction_votes = []  # 'right' or 'left'
        
        # First attempt: 4 frames before transition
        for i in range(start_frame, min(end_frame, len(self.pose_data))):
            frame_data = self.pose_data[i]
            pose = frame_data.get('pose', {})
            
            # Get hip and arm positions
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            left_wrist = pose.get('left_wrist', {})
            right_wrist = pose.get('right_wrist', {})
            left_shoulder = pose.get('left_shoulder', {})
            right_shoulder = pose.get('right_shoulder', {})
            
            # Determine center reference point (prefer hip, fallback to shoulder)
            center_x = None
            reference_type = None
            
            # Try hip center first
            if (isinstance(left_hip, dict) and isinstance(right_hip, dict) and
                'x' in left_hip and 'x' in right_hip):
                center_x = (left_hip['x'] + right_hip['x']) / 2
                reference_type = "Hip center"
            # Fallback to shoulder center
            elif (isinstance(left_shoulder, dict) and isinstance(right_shoulder, dict) and
                  'x' in left_shoulder and 'x' in right_shoulder):
                center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                reference_type = "Shoulder center"
            
            if center_x is None:
                continue
            
            # Check arm positions relative to center
            arm_direction = None
            if isinstance(left_wrist, dict) and 'x' in left_wrist:
                if left_wrist['x'] > center_x:  # 왼팔이 중심보다 오른쪽에 있음 → 오른쪽을 보고 있음
                    arm_direction = 'right'
                else:
                    arm_direction = 'left'
            elif isinstance(right_wrist, dict) and 'x' in right_wrist:
                if right_wrist['x'] < center_x:  # 오른팔이 중심보다 왼쪽에 있음 → 왼쪽을 보고 있음
                    arm_direction = 'left'
                else:
                    arm_direction = 'right'
            
            if arm_direction:
                direction_votes.append(arm_direction)
        
        # If not enough votes, extend to 8 frames before transition
        if len(direction_votes) < 3:
            print(f"   ⚠️ Only {len(direction_votes)} valid votes in 4-frame window, extending to 8 frames...")
            
            # Calculate 8 frames before transition
            extended_required_frames = max(6, int(8 * (fps / 30.0)))
            extended_start_frame = first_transition_frame - extended_required_frames
            extended_end_frame = first_transition_frame
            
            # Reset for extended search
            direction_votes = []
            
            for i in range(extended_start_frame, min(extended_end_frame, len(self.pose_data))):
                frame_data = self.pose_data[i]
                pose = frame_data.get('pose', {})
                
                # Get hip and arm positions
                left_hip = pose.get('left_hip', {})
                right_hip = pose.get('right_hip', {})
                left_wrist = pose.get('left_wrist', {})
                right_wrist = pose.get('right_wrist', {})
                left_shoulder = pose.get('left_shoulder', {})
                right_shoulder = pose.get('right_shoulder', {})
                
                # Determine center reference point (prefer hip, fallback to shoulder)
                center_x = None
                reference_type = None
                
                # Try hip center first
                if (isinstance(left_hip, dict) and isinstance(right_hip, dict) and
                    'x' in left_hip and 'x' in right_hip):
                    center_x = (left_hip['x'] + right_hip['x']) / 2
                    reference_type = "Hip center"
                # Fallback to shoulder center
                elif (isinstance(left_shoulder, dict) and isinstance(right_shoulder, dict) and
                      'x' in left_shoulder and 'x' in right_shoulder):
                    center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                    reference_type = "Shoulder center"
                
                if center_x is None:
                    continue
                
                # Check arm positions relative to center
                arm_direction = None
                if isinstance(left_wrist, dict) and 'x' in left_wrist:
                    if left_wrist['x'] > center_x:  # 왼팔이 중심보다 오른쪽에 있음 → 오른쪽을 보고 있음
                        arm_direction = 'right'
                    else:
                        arm_direction = 'left'
                elif isinstance(right_wrist, dict) and 'x' in right_wrist:
                    if right_wrist['x'] < center_x:  # 오른팔이 중심보다 왼쪽에 있음 → 왼쪽을 보고 있음
                        arm_direction = 'left'
                    else:
                        arm_direction = 'right'
                
                if arm_direction:
                    direction_votes.append(arm_direction)
        
        # Determine final direction
        if direction_votes:
            from collections import Counter
            direction_count = Counter(direction_votes)
            facing_direction = direction_count.most_common(1)[0][0]
            
            # Determine reference hip
            reference_hip_side = 'right' if facing_direction == 'right' else 'left'
            
            print(f"   Direction votes: {dict(direction_count)}")
            print(f"   ✅ Final decision: facing {facing_direction} (reference: {reference_hip_side} hip) (from {len(direction_votes)} votes)")
            
            return facing_direction, reference_hip_side
        else:
            print(f"   ⚠️ Could not determine direction, defaulting to facing right")
            return 'right', 'right'
    
    def _determine_facing_direction_from_start(self) -> tuple:
        """Fallback method: determine facing direction from first 4 frames"""
        fps = getattr(self, 'video_fps', 30.0)
        required_frames = max(3, int(4 * (fps / 30.0)))
        
        print(f"   📊 Determining direction from first {required_frames} frames...")
        
        direction_votes = []
        
        for i in range(min(required_frames, len(self.pose_data))):
            frame_data = self.pose_data[i]
            pose = frame_data.get('pose', {})
            
            # Get hip and arm positions
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            left_wrist = pose.get('left_wrist', {})
            right_wrist = pose.get('right_wrist', {})
            left_shoulder = pose.get('left_shoulder', {})
            right_shoulder = pose.get('right_shoulder', {})
            
            # Determine center reference point (prefer hip, fallback to shoulder)
            center_x = None
            reference_type = None
            
            # Try hip center first
            if (isinstance(left_hip, dict) and isinstance(right_hip, dict) and
                'x' in left_hip and 'x' in right_hip):
                center_x = (left_hip['x'] + right_hip['x']) / 2
                reference_type = "Hip center"
            # Fallback to shoulder center
            elif (isinstance(left_shoulder, dict) and isinstance(right_shoulder, dict) and
                  'x' in left_shoulder and 'x' in right_shoulder):
                center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                reference_type = "Shoulder center"
            
            if center_x is None:
                continue
            
            # Check arm positions relative to center
            arm_direction = None
            if isinstance(left_wrist, dict) and 'x' in left_wrist:
                if left_wrist['x'] > center_x:  # 왼팔이 중심보다 오른쪽에 있음 → 오른쪽을 보고 있음
                    arm_direction = 'right'
                else:
                    arm_direction = 'left'
            elif isinstance(right_wrist, dict) and 'x' in right_wrist:
                if right_wrist['x'] < center_x:  # 오른팔이 중심보다 왼쪽에 있음 → 왼쪽을 보고 있음
                    arm_direction = 'left'
                else:
                    arm_direction = 'right'
            
            if arm_direction:
                direction_votes.append(arm_direction)
        
        # Determine final direction
        if direction_votes:
            from collections import Counter
            direction_count = Counter(direction_votes)
            facing_direction = direction_count.most_common(1)[0][0]
            
            # Determine reference hip
            reference_hip_side = 'right' if facing_direction == 'right' else 'left'
            
            print(f"   ✅ First frames direction: {facing_direction}")
            print(f"   📊 Direction votes: {dict(direction_count)}")
            print(f"   📏 Reference hip: {reference_hip_side}")
            
            return facing_direction, reference_hip_side
        else:
            print(f"   ⚠️ Could not determine direction from first frames, using overall frames")
            return self._determine_facing_direction_from_all_frames()
    
    def _normalize_keypoint_names(self, pose: dict, facing_direction: str) -> dict:
        """Normalize keypoint names based on facing direction (left-facing becomes right-facing)"""
        if facing_direction == 'right':
            return pose  # Already facing right, no change needed
        
        # Convert left-facing to right-facing by swapping left/right keypoints
        normalized_pose = {}
        
        # Mapping for left-to-right conversion
        left_to_right_mapping = {
            'left_shoulder': 'right_shoulder',
            'right_shoulder': 'left_shoulder',
            'left_elbow': 'right_elbow', 
            'right_elbow': 'left_elbow',
            'left_wrist': 'right_wrist',
            'right_wrist': 'left_wrist',
            'left_hip': 'right_hip',
            'right_hip': 'left_hip',
            'left_knee': 'right_knee',
            'right_knee': 'left_knee',
            'left_ankle': 'right_ankle',
            'right_ankle': 'left_ankle'
        }
        
        for original_key, keypoint_data in pose.items():
            if original_key in left_to_right_mapping:
                new_key = left_to_right_mapping[original_key]
                normalized_pose[new_key] = keypoint_data
            else:
                # Keep other keypoints as-is (nose, eyes, ears, etc.)
                normalized_pose[original_key] = keypoint_data
        
        return normalized_pose
    
    def _get_torso_from_phase_detection(self) -> float:
        """Get stable torso length from phase detection result (deprecated - use shot-based torso)"""
        if hasattr(self, 'phase_detector') and self.phase_detector is not None:
            if hasattr(self.phase_detector, 'transition_reference_torso') and self.phase_detector.transition_reference_torso is not None:
                torso_length = self.phase_detector.transition_reference_torso
                print(f"   ✅ Using phase detection torso: {torso_length:.4f}")
                return torso_length
            else:
                print(f"   ⚠️ Phase detection torso not available, calculating from all frames")
                return self._calculate_torso_from_all_frames()
        else:
            print(f"   ⚠️ Phase detector not available, calculating from all frames")
            return self._calculate_torso_from_all_frames()
    
    def _get_torso_for_frame(self, frame_idx: int) -> float:
        """
        Get appropriate torso length for a specific frame
        Uses shot-based fixed torso if frame belongs to a shot, otherwise fallback
        """
        # Check if frame belongs to any shot
        if len(self.shot_detector.shots) > 0:
            for shot in self.shot_detector.shots:
                if shot['start_frame'] <= frame_idx <= shot['end_frame']:
                    return shot['fixed_torso']
        
        # Fallback: use pre-calculated fallback torso
        if hasattr(self, 'fallback_torso_length'):
            return self.fallback_torso_length
        else:
            # Last resort: use phase detection torso
            return self._get_torso_from_phase_detection()
    
    def _get_shot_for_frame(self, frame_idx: int) -> dict:
        """
        Get shot information for a specific frame
        Returns shot dict if frame belongs to a shot, None otherwise
        """
        if len(self.shot_detector.shots) > 0:
            for shot in self.shot_detector.shots:
                if shot['start_frame'] <= frame_idx <= shot['end_frame']:
                    return shot
        return None
    
    def _get_fallback_torso_from_shots(self) -> float:
        """
        Get fallback torso length from average of all shot torso measurements
        If no shots available, use phase detection torso
        """
        if hasattr(self, 'shots') and self.shots:
            # Calculate average of all shot torso measurements
            shot_torsos = [shot['fixed_torso'] for shot in self.shots]
            avg_shot_torso = np.mean(shot_torsos)
            
            print(f"   ✅ Calculated fallback torso from {len(self.shots)} shots:")
            for shot in self.shots:
                print(f"      Shot {shot['shot_id']}: {shot['fixed_torso']:.4f}")
            print(f"   📏 Average shot torso (fallback): {avg_shot_torso:.4f}")
            
            return avg_shot_torso
        else:
            # No shots available, use phase detection torso as last resort
            print("   ⚠️ No shots available for fallback, using phase detection torso")
            return self._get_torso_from_phase_detection()
    
    def _determine_facing_direction_for_shot(self, shot: dict) -> tuple:
        """Determine facing direction for a specific shot using frames before first meaningful transition within shot"""
        shot_start = shot['start_frame']
        shot_end = shot['end_frame']
        fps = getattr(self, 'video_fps', 30.0)
        required_frames = max(3, int(4 * (fps / 30.0)))
        
        # Find first meaningful transition within this shot (Set-up → Loading/Rising)
        first_transition_frame = None
        
        for i in range(shot_start, shot_end + 1):
            if i < len(self.phases):
                current_phase = self.phases[i]
                if i > shot_start and i < len(self.phases):
                    prev_phase = self.phases[i-1]
                    # Check for meaningful transition: Set-up → Loading/Rising
                    if (prev_phase == "Set-up" and 
                        current_phase in ["Loading", "Rising", "Loading-Rising"]):
                        first_transition_frame = i
                        break
        
        if first_transition_frame is None:
            first_transition_frame = shot_start
        
        # Use frames before first meaningful transition within shot (4 frames)
        start_frame = max(shot_start, first_transition_frame - required_frames)
        end_frame = first_transition_frame
        
        direction_votes = []
        
        # First attempt: 4 frames before transition
        for i in range(start_frame, min(end_frame, len(self.pose_data))):
            frame_data = self.pose_data[i]
            pose = frame_data.get('pose', {})
            
            # Get hip and arm positions
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            left_wrist = pose.get('left_wrist', {})
            right_wrist = pose.get('right_wrist', {})
            left_shoulder = pose.get('left_shoulder', {})
            right_shoulder = pose.get('right_shoulder', {})
            
            # Determine center reference point
            center_x = None
            reference_type = None
            
            # Try hip center first
            if (isinstance(left_hip, dict) and isinstance(right_hip, dict) and
                'x' in left_hip and 'x' in right_hip):
                center_x = (left_hip['x'] + right_hip['x']) / 2
                reference_type = "Hip center"
            # Fallback to shoulder center
            elif (isinstance(left_shoulder, dict) and isinstance(right_shoulder, dict) and
                  'x' in left_shoulder and 'x' in right_shoulder):
                center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                reference_type = "Shoulder center"
            
            if center_x is None:
                continue
            
            # Check arm positions relative to center
            arm_direction = None
            if isinstance(left_wrist, dict) and 'x' in left_wrist:
                if left_wrist['x'] > center_x:
                    arm_direction = 'right'
                else:
                    arm_direction = 'left'
            elif isinstance(right_wrist, dict) and 'x' in right_wrist:
                if right_wrist['x'] < center_x:
                    arm_direction = 'left'
                else:
                    arm_direction = 'right'
            
            if arm_direction:
                direction_votes.append(arm_direction)
        
        # If not enough votes, extend to 8 frames before transition
        if len(direction_votes) < 3:
            print(f"      ⚠️ Only {len(direction_votes)} valid votes in 4-frame window, extending to 8 frames...")
            
            # Calculate 8 frames before transition
            extended_required_frames = max(6, int(8 * (fps / 30.0)))
            extended_start_frame = max(shot_start, first_transition_frame - extended_required_frames)
            extended_end_frame = first_transition_frame
            
            # Reset for extended search
            direction_votes = []
            
            for i in range(extended_start_frame, min(extended_end_frame, len(self.pose_data))):
                frame_data = self.pose_data[i]
                pose = frame_data.get('pose', {})
                
                # Get hip and arm positions
                left_hip = pose.get('left_hip', {})
                right_hip = pose.get('right_hip', {})
                left_wrist = pose.get('left_wrist', {})
                right_wrist = pose.get('right_wrist', {})
                left_shoulder = pose.get('left_shoulder', {})
                right_shoulder = pose.get('right_shoulder', {})
                
                # Determine center reference point
                center_x = None
                reference_type = None
                
                # Try hip center first
                if (isinstance(left_hip, dict) and isinstance(right_hip, dict) and
                    'x' in left_hip and 'x' in right_hip):
                    center_x = (left_hip['x'] + right_hip['x']) / 2
                    reference_type = "Hip center"
                # Fallback to shoulder center
                elif (isinstance(left_shoulder, dict) and isinstance(right_shoulder, dict) and
                      'x' in left_shoulder and 'x' in right_shoulder):
                    center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                    reference_type = "Shoulder center"
                
                if center_x is None:
                    continue
                
                # Check arm positions relative to center
                arm_direction = None
                if isinstance(left_wrist, dict) and 'x' in left_wrist:
                    if left_wrist['x'] > center_x:
                        arm_direction = 'right'
                    else:
                        arm_direction = 'left'
                elif isinstance(right_wrist, dict) and 'x' in right_wrist:
                    if right_wrist['x'] < center_x:
                        arm_direction = 'left'
                    else:
                        arm_direction = 'right'
                
                if arm_direction:
                    direction_votes.append(arm_direction)
        
        # Determine final direction
        if direction_votes:
            from collections import Counter
            direction_count = Counter(direction_votes)
            facing_direction = direction_count.most_common(1)[0][0]
            reference_hip_side = 'right' if facing_direction == 'right' else 'left'
            
            print(f"      Direction votes: {dict(direction_count)}")
            print(f"      ✅ Shot {shot['shot_id']}: facing {facing_direction} (reference: {reference_hip_side} hip) (from {len(direction_votes)} votes)")
            
            return facing_direction, reference_hip_side
        else:
            print(f"      ⚠️ Could not determine direction for shot {shot['shot_id']}, defaulting to facing right")
            return 'right', 'right'
    
    def _calculate_stable_reference_hip_for_shot(self, shot: dict, reference_hip_side: str) -> tuple:
        """Calculate stable reference hip position for a specific shot using frames before first meaningful transition within shot"""
        shot_start = shot['start_frame']
        shot_end = shot['end_frame']
        fps = getattr(self, 'video_fps', 30.0)
        required_frames = max(3, int(4 * (fps / 30.0)))
        
        # Find first meaningful transition within this shot (Set-up → Loading/Rising)
        first_transition_frame = None
        
        for i in range(shot_start + 1, shot_end + 1):  # shot_start + 1부터 검사 (첫 프레임 제외)
            if i < len(self.phases):
                current_phase = self.phases[i]
                prev_phase = self.phases[i-1]
                # Check for meaningful transition: Set-up → Loading/Rising
                if (prev_phase == "Set-up" and 
                    current_phase in ["Loading", "Rising", "Loading-Rising"]):
                    first_transition_frame = i
                    break
        
        if first_transition_frame is None:
            first_transition_frame = shot_start
        
        # Use frames before first meaningful transition within shot (4 frames)
        start_frame = max(shot_start, first_transition_frame - required_frames)
        end_frame = first_transition_frame
        
        hip_x_values = []
        hip_y_values = []
        frames_used = []
        
        # First attempt: 4 frames before transition
        for frame_idx in range(start_frame, end_frame):
            if frame_idx < len(self.pose_data):
                frame_data = self.pose_data[frame_idx]
                pose = frame_data.get('pose', {})
                
                # Get the reference hip
                hip_key = f'{reference_hip_side}_hip'
                reference_hip = pose.get(hip_key, {})
                
                if (isinstance(reference_hip, dict) and 
                    'x' in reference_hip and 'y' in reference_hip):
                    
                    hip_x_values.append(reference_hip['x'])
                    hip_y_values.append(reference_hip['y'])
                    frames_used.append(frame_idx)
        
        # If not enough frames, extend to 8 frames before transition
        if len(hip_x_values) < 3:
            print(f"      ⚠️ Only {len(hip_x_values)} valid frames in 4-frame window, extending to 8 frames...")
            
            # Calculate 8 frames before transition
            extended_required_frames = max(6, int(8 * (fps / 30.0)))
            extended_start_frame = max(shot_start, first_transition_frame - extended_required_frames)
            extended_end_frame = first_transition_frame
            
            # Reset for extended search
            hip_x_values = []
            hip_y_values = []
            frames_used = []
            
            for frame_idx in range(extended_start_frame, end_frame):
                if frame_idx < len(self.pose_data):
                    frame_data = self.pose_data[frame_idx]
                    pose = frame_data.get('pose', {})
                    
                    # Get the reference hip
                    hip_key = f'{reference_hip_side}_hip'
                    reference_hip = pose.get(hip_key, {})
                    
                    if (isinstance(reference_hip, dict) and 
                        'x' in reference_hip and 'y' in reference_hip):
                        
                        hip_x_values.append(reference_hip['x'])
                        hip_y_values.append(reference_hip['y'])
                        frames_used.append(frame_idx)
        
        if len(hip_x_values) >= 3:  # Minimum 3 frames
            stable_hip_x = np.mean(hip_x_values)
            stable_hip_y = np.mean(hip_y_values)
            return stable_hip_x, stable_hip_y
        else:
            return self._calculate_stable_reference_hip(reference_hip_side)
    
    def _calculate_stable_reference_hip_from_phase_detection(self, reference_hip_side: str) -> tuple:
        """Calculate stable reference hip position from frames before first phase transition"""
        if not hasattr(self, 'phase_detector') or not self.phase_detector:
            print("   ⚠️ Phase detector not available, using first 4 frames")
            return self._calculate_stable_reference_hip(reference_hip_side)
        
        # Get the frame index where first meaningful transition occurred
        first_transition_frame = getattr(self.phase_detector, 'first_transition_frame', None)

        
        if first_transition_frame is None:
            print("   ⚠️ No phase transition detected, using first 4 frames")
            return self._calculate_stable_reference_hip(reference_hip_side)
        
        fps = getattr(self, 'video_fps', 30.0)
        required_frames = max(3, int(4 * (fps / 30.0)))
        
        # Check if we have enough frames before the first transition
        if first_transition_frame < required_frames:
            print(f"   ⚠️ First transition too early (frame {first_transition_frame} < {required_frames} required), using fallback method")
            return self._calculate_stable_reference_hip(reference_hip_side)
        
        # Use frames immediately before the first transition (4 frames)
        start_frame = first_transition_frame - required_frames
        end_frame = first_transition_frame
        reference_frames = list(range(start_frame, end_frame))
        
        hip_x_values = []
        hip_y_values = []
        frames_used = []
        
        # First attempt: 4 frames before transition
        for frame_idx in reference_frames:
            if frame_idx < len(self.pose_data):
                frame_data = self.pose_data[frame_idx]
                pose = frame_data.get('pose', {})
                
                # Get the reference hip
                hip_key = f'{reference_hip_side}_hip'
                reference_hip = pose.get(hip_key, {})
                
                if (isinstance(reference_hip, dict) and 
                    'x' in reference_hip and 'y' in reference_hip):
                    
                    hip_x_values.append(reference_hip['x'])
                    hip_y_values.append(reference_hip['y'])
                    frames_used.append(frame_idx)
        
        # If not enough frames, extend to 8 frames before transition
        if len(hip_x_values) < 3:
            print(f"   ⚠️ Only {len(hip_x_values)} valid frames in 4-frame window, extending to 8 frames...")
            
            # Calculate 8 frames before transition
            extended_required_frames = max(6, int(8 * (fps / 30.0)))
            extended_start_frame = first_transition_frame - extended_required_frames
            extended_reference_frames = list(range(extended_start_frame, end_frame))
            
            # Reset for extended search
            hip_x_values = []
            hip_y_values = []
            frames_used = []
            
            for frame_idx in extended_reference_frames:
                if frame_idx < len(self.pose_data):
                    frame_data = self.pose_data[frame_idx]
                    pose = frame_data.get('pose', {})
                    
                    # Get the reference hip
                    hip_key = f'{reference_hip_side}_hip'
                    reference_hip = pose.get(hip_key, {})
                    
                    if (isinstance(reference_hip, dict) and 
                        'x' in reference_hip and 'y' in reference_hip):
                        
                        hip_x_values.append(reference_hip['x'])
                        hip_y_values.append(reference_hip['y'])
                        frames_used.append(frame_idx)
        
        if len(hip_x_values) >= 3:  # Minimum 3 frames
            stable_hip_x = np.mean(hip_x_values)
            stable_hip_y = np.mean(hip_y_values)
            print(f"   ✅ Phase detection hip: ({stable_hip_x:.4f}, {stable_hip_y:.4f}) (from {len(frames_used)} frames)")
            return stable_hip_x, stable_hip_y
        else:
            print(f"   ⚠️ Not enough valid hip measurements ({len(hip_x_values)}/3), using first 4 frames")
            return self._calculate_stable_reference_hip(reference_hip_side)
    
    def _calculate_stable_reference_hip(self, reference_hip_side: str) -> tuple:
        """Calculate stable reference hip position from first 4 frames (2nd fallback for hip)"""
        fps = getattr(self, 'video_fps', 30.0)
        required_frames = max(3, int(4 * (fps / 30.0)))
        
        print(f"   📊 Calculating {reference_hip_side} hip from first {required_frames} frames...")
        
        hip_x_values = []
        hip_y_values = []
        frames_used = []
        
        for i in range(min(required_frames, len(self.pose_data))):
            frame_data = self.pose_data[i]
            pose = frame_data.get('pose', {})
            
            # Get the reference hip
            hip_key = f'{reference_hip_side}_hip'
            reference_hip = pose.get(hip_key, {})
            
            if (isinstance(reference_hip, dict) and 
                'x' in reference_hip and 'y' in reference_hip):
                
                hip_x_values.append(reference_hip['x'])
                hip_y_values.append(reference_hip['y'])
                frames_used.append(i)
            else:
                continue
        
        if len(hip_x_values) >= 3:
            stable_hip_x = np.mean(hip_x_values)
            stable_hip_y = np.mean(hip_y_values)
            print(f"   ✅ First frames hip: ({stable_hip_x:.4f}, {stable_hip_y:.4f})")
            return stable_hip_x, stable_hip_y
        else:
            print(f"   ❌ Could not calculate hip from first frames, normalization failed")
            return None, None

    def _save_normalized_data(self, video_path: Optional[str]):
        """Save normalized data as separate JSON file"""
        if video_path is None:
            print("❌ video_path not provided, cannot save normalized data")
            return
            
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Get shots metadata from ShotDetector
        shots_metadata = self.shot_detector.get_shots_metadata()
        
        # Save normalized pose data
        pose_normalized_file = os.path.join(self.extracted_data_dir, f"{base_name}_pose_normalized.json")
        pose_data_to_save = {
            "metadata": {
                "total_frames": len(self.normalized_data),
                "normalization_time": datetime.now().isoformat(),
                "coordinate_system": "shot_based_torso_with_direction_normalization",
                "fallback_torso_length": self.normalized_data[0]['scaling_factor'] if self.normalized_data else 0.1,
                "facing_direction": self.normalized_data[0]['facing_direction'] if self.normalized_data else 'right',
                "reference_hip_side": self.normalized_data[0]['reference_hip_side'] if self.normalized_data else 'right',
                "normalization_method": "shot_based_individual_torso",
                "shots_detected": len(shots_metadata),
                "shots": shots_metadata
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
                "coordinate_system": "shot_based_torso_with_direction_normalization",
                "fallback_torso_length": self.normalized_data[0]['scaling_factor'] if self.normalized_data else 0.1,
                "facing_direction": self.normalized_data[0]['facing_direction'] if self.normalized_data else 'right',
                "reference_hip_side": self.normalized_data[0]['reference_hip_side'] if self.normalized_data else 'right',
                "normalization_method": "shot_based_individual_torso",
                "shots_detected": len(shots_metadata),
                "shots": shots_metadata
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
        print(f"\n🎯 STEP 3: Segment shooting phases (using {detector_type} detector)")
        print("=" * 50)
        
        if not self.pose_data or not self.ball_data:
            print("❌ Original pose or ball data not found.")
            return
        
        # Select primary hand before phase detection using original data
        self.selected_hand, self.selected_hand_confidence = self.select_primary_hand_from_original_data()
        
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
        
        # Store reference to phase detector for normalization to use
        self.phase_detector = self.current_detector
        
        self.phases = []
        current_phase = "General" # Start with a general phase
        phase_start_frame = 0
        
        # Track phase history for cancellation
        phase_history = []  # List of (phase, start_frame, end_frame)
        current_phase_start = 0
        
        # Setup for noise filtering
        min_phase_duration = 3  # Must last at least 3 frames
        noise_threshold = 4  # Changes of 4 frames or less are considered noise
        
        for i, frame_data in enumerate(self.pose_data):
            pose = frame_data.get('pose', {})
            
            # Get selected hand keypoints
            selected_shoulder, selected_elbow, selected_wrist = self.get_selected_hand_keypoints(pose)
            
            # Extract necessary keypoints from original data (using selected hand)
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
            
            # Get ball position for calculations from original data
            ball_info = None
            if i < len(self.ball_data):
                ball_frame_data = self.ball_data[i]
                if isinstance(ball_frame_data, dict) and ball_frame_data.get('ball_detections'):
                    ball_detections = ball_frame_data['ball_detections']
                    if ball_detections and isinstance(ball_detections[0], dict):
                        ball_info = ball_detections[0]
            
            ball_x = ball_info.get('center_x', 0) if ball_info else 0
            ball_y = ball_info.get('center_y', 0) if ball_info else 0
            
            # Calculate movement deltas using selected hand
            if i > 0 and i-1 < len(self.pose_data):
                prev_frame_data = self.pose_data[i-1]
                prev_pose = prev_frame_data.get('pose', {})
                
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
                
            # Provide current torso to detector for threshold calculation
            current_torso = self._get_current_torso_for_thresholds()
            if hasattr(self.current_detector, 'set_current_torso'):
                self.current_detector.set_current_torso(current_torso)
                
            next_phase = self.current_detector.check_phase_transition(
                current_phase, i, self.pose_data, self.ball_data, 
                fps=fps, selected_hand=self.selected_hand
            )
            

            
            # Update rolling torso tracking (delegated to ShotDetector)  
            self.shot_detector.update_rolling_torso(i, pose)
            
            # Real-time shot detection - detect shot transitions
            self.shot_detector.detect_shot_transitions(i, next_phase)
            
            # Minimum phase duration disabled - transition immediately when conditions are met
            if next_phase != current_phase:
                            # Record current phase in history before changing
                            if current_phase != "General":
                                phase_history.append((current_phase, current_phase_start, i))
                            
                            current_phase = next_phase
                            phase_start_frame = i
                            current_phase_start = i
                # print(f"Frame {i}: {current_phase} phase started")  # 로그 제거
            
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
    
        # Find first meaningful transition after cancellation processing
        self._find_first_meaningful_transition()
        
        # Finalize ShotDetector results - assign shot info to frames
        self.shot_detector.finalize_frame_shots(len(self.phases))
        
        print(f"\n🎯 Real-time shot detection completed: {len(self.shot_detector.shots)} shots detected")
    
    def _find_first_meaningful_transition(self):
        """Find first meaningful transition from Set-up to Loading/Rising/Loading-Rising after cancellation processing"""
        if not self.phases:
            return
        
        print("\n🔍 Finding first meaningful transition from final phases...")
        
        first_transition_frame = None
        for i in range(1, len(self.phases)):
            prev_phase = self.phases[i-1]
            curr_phase = self.phases[i]
            
            # Look for Set-up → meaningful phase transitions
            if (prev_phase == "Set-up" and 
                curr_phase in ["Loading", "Rising", "Loading-Rising"]):
                first_transition_frame = i
                print(f"   ✅ First meaningful transition found: Set-up → {curr_phase} at frame {i}")
                break
        
        if first_transition_frame is not None:
            # Update phase detector with the correct first transition frame and finalize torso measurement
            if hasattr(self, 'phase_detector') and self.phase_detector:
                self.phase_detector.finalize_transition_reference_torso(first_transition_frame)
                print(f"   📝 Updated phase detector first_transition_frame to {first_transition_frame}")
        else:
            print("   ⚠️ No meaningful transition found in final phases")
    

    
    def _update_rolling_torso(self, frame_idx: int, pose: Dict):
        """
        Update rolling torso measurement with current frame data
        Only updates if torso_tracking_active is True
        """
        if not self.torso_tracking_active:
            return
        
        # Calculate current frame torso
        torso_length = self._calculate_frame_torso(pose)
        
        if torso_length > 0:
            # Add new measurement
            self.rolling_torso_values.append(torso_length)
            self.rolling_torso_frames.append(frame_idx)
            
            # Keep only last 4 measurements
            if len(self.rolling_torso_values) > 4:
                self.rolling_torso_values.pop(0)
                self.rolling_torso_frames.pop(0)
            
            # Debug output (disabled for cleaner logs)
            # if len(self.rolling_torso_values) >= 4:
            #     avg_torso = np.mean(self.rolling_torso_values)
            #     print(f"   🔄 Frame {frame_idx}: Rolling torso avg = {avg_torso:.4f} (window: {len(self.rolling_torso_values)}/4)")
    
    def _calculate_frame_torso(self, pose: Dict) -> float:
        """
        Calculate torso length for a single frame
        Returns average of left and right torso lengths if both available
        """
        required_keypoints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        for keypoint in required_keypoints:
            if keypoint not in pose:
                return 0.0
            kp_data = pose[keypoint]
            if not isinstance(kp_data, dict) or 'x' not in kp_data or 'y' not in kp_data:
                return 0.0
        
        # Calculate left torso (left shoulder to left hip)
        left_shoulder = pose['left_shoulder']
        left_hip = pose['left_hip']
        left_torso_length = 0.0
        
        if (left_shoulder.get('confidence', 0) > 0.3 and left_hip.get('confidence', 0) > 0.3):
            left_dx = left_shoulder['x'] - left_hip['x']
            left_dy = left_shoulder['y'] - left_hip['y']
            left_torso_length = np.sqrt(left_dx**2 + left_dy**2)
        
        # Calculate right torso (right shoulder to right hip)
        right_shoulder = pose['right_shoulder']
        right_hip = pose['right_hip']
        right_torso_length = 0.0
        
        if (right_shoulder.get('confidence', 0) > 0.3 and right_hip.get('confidence', 0) > 0.3):
            right_dx = right_shoulder['x'] - right_hip['x']
            right_dy = right_shoulder['y'] - right_hip['y']
            right_torso_length = np.sqrt(right_dx**2 + right_dy**2)
        
        # Return average of valid measurements
        valid_measurements = []
        if left_torso_length > 0:
            valid_measurements.append(left_torso_length)
        if right_torso_length > 0:
            valid_measurements.append(right_torso_length)
        
        if valid_measurements:
            return np.mean(valid_measurements)
        else:
            return 0.0
    
    def _get_current_torso_for_thresholds(self) -> float:
        """
        Get current torso length for phase detection thresholds
        Uses ShotDetector's torso management
        """
        # Get torso from ShotDetector (includes shot-aware torso management)
        return self.shot_detector.get_shot_torso()
    
    # Shot-related methods removed - now handled by ShotDetector
    
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
                # print(f"    Abnormal transition at frame {i}: {current_phase} → {next_phase}")  # 로그 제거
        
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
                    # print(f"    Frame {i}: {self.phases[i]} → Set-up (abnormal transition)")  # 로그 제거
        
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
        
        # Prepare shots metadata
        shots_metadata = []
        if len(self.shot_detector.shots) > 0:
            for shot in self.shot_detector.shots:
                shot_meta = {
                    "shot_id": shot['shot_id'],
                    "start_frame": shot['start_frame'],
                    "end_frame": shot['end_frame'],
                    "total_frames": shot['total_frames'],
                    "fixed_torso": shot['fixed_torso']
                }
                shots_metadata.append(shot_meta)
        
        # Configure result data
        results = {
            "metadata": {
                "video_path": video_path,
                "analysis_date": datetime.now().isoformat(),
                "total_frames": len(self.normalized_data),
                "phases_detected": list(set(self.phases)),
                "shots_detected": len(shots_metadata),
                "shots": shots_metadata,
                "normalization_method": "shot_based_individual_torso",
                "phase_detection_method": "sequential_transition",
                "hand": self.selected_hand,
                "fps" : self.video_fps,
            },
            "frames": []
        }
        
        # Only include frames that belong to shots
        shot_frames = []
        for i, frame_data in enumerate(self.normalized_data):
            # Get shot information for this frame
            shot_id = None
            if i < len(self.shot_detector.frame_shots):
                shot_id = self.shot_detector.frame_shots[i]
            
            # Only include frames that belong to a shot
            if shot_id is not None:
                frame_result = {
                    "frame_index": i,
                    "phase": self.phases[i] if i < len(self.phases) else "Unknown",
                    "shot": shot_id,
                    "normalized_pose": frame_data["normalized_pose"],
                    "normalized_ball": frame_data["normalized_ball"],
                    "scaling_factor": frame_data["scaling_factor"],
                    "ball_detected": frame_data["ball_detected"],
                    "shot_normalization_applied": frame_data.get("shot_normalization_applied", False)
                }
                shot_frames.append(frame_result)
        
        results["frames"] = shot_frames
        
        # Update metadata to reflect shot-only data
        results["metadata"]["total_frames"] = len(shot_frames)
        results["metadata"]["shot_frames_only"] = True
        
        # Save as JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved: {output_file}")
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    def generate_visualization(self, video_path: str, overwrite_mode: bool = False) -> bool:
        """Generate dual visualization video (left: original, right: normalized)"""
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Check for existing dual video file
        dual_output = os.path.join(self.visualized_video_dir, f"{base_name}_dual_analyzed.mp4")
        
        if os.path.exists(dual_output) and not overwrite_mode:
            print(f"\n⚠️ Existing dual visualization video found:")
            print(f"  - {os.path.basename(dual_output)}")
            choice = input("Overwrite? (y/n): ").strip().lower()
            if choice != 'y':
                print("Keeping existing dual visualization video.")
                return True
        
        try:
            # Load data
            original_pose_data = self.pose_data
            original_ball_data = self.ball_data
            original_rim_data = self.rim_data
            
            # Generate dual visualization (if normalized data exists)
            if hasattr(self, 'normalized_data') and self.normalized_data:
                print("\n🎬 Generating dual visualization...")
                self.create_dual_analysis_video(
                video_path=video_path,
                    output_path=dual_output,
                original_pose_data=original_pose_data,
                    normalized_pose_data=self.normalized_data,
                original_ball_data=original_ball_data,
                    normalized_ball_data=[frame.get('normalized_ball', {}) for frame in self.normalized_data],
                original_rim_data=original_rim_data,
                shooting_phases=self.phases
            )
                print(f"✅ Dual visualization: {os.path.basename(dual_output)}")
            else:
                print("\n⚠️ No normalized data available for dual visualization")
                return False
            
            return True
        except Exception as e:
            print(f"❌ Failed to generate dual visualization video: {e}")
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
                original_frame = self._draw_selected_hand_label(original_frame, self.selected_hand, self.selected_hand_confidence, frame_count)
                
                # Right: Normalized data
                if frame_count < len(normalized_pose_data):
                    normalized_frame = self._draw_pose_skeleton_normalized(normalized_frame, frame_count, normalized_pose_data)
                    normalized_frame = self._draw_ball_normalized(normalized_frame, frame_count, normalized_ball_data)
                
                if shooting_phases and frame_count < len(shooting_phases):
                    normalized_frame = self._draw_phase_label(normalized_frame, frame_count, "Normalized", shooting_phases)
                
                # Add selected hand label to normalized frame
                normalized_frame = self._draw_selected_hand_label(normalized_frame, self.selected_hand, self.selected_hand_confidence, frame_count)
                

                
                # Stack two frames side by side
                combined_frame = np.hstack([original_frame, normalized_frame])
                
                out.write(combined_frame)
                frame_count += 1
                
                # Print progress (every 10 frames) - disabled for cleaner output
                # if frame_count % 10 == 0:
                #     print(f"🎬 Processing frames: {frame_count}/{total_frames}")
            
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
                
                # Add shot information label
                frame = self._draw_shot_info_label(frame, frame_count)
                
                out.write(frame)
                frame_count += 1
                
                # Print progress (every 10 frames) - disabled for cleaner output
                # if frame_count % 10 == 0:
                #     print(f"🎬 Processing frames: {frame_count}/{total_frames}")
            
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
        
        # Debug: Print phase information (only when phase changes) - 로그 제거
        # if not hasattr(self, '_last_logged_phase') or self._last_logged_phase != phase:
        #     print(f"Frame {frame_idx} ({data_type}): Phase = {phase}")
        #     self._last_logged_phase = phase
        
        # Label text (just the phase, without data_type to keep it small)
        label_text = f"{phase}"
        
        # Font settings - larger size
        font_scale = 0.8
        font_thickness = 2
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
            "Loading-Rising": (0, 200, 255), # Light Orange-Yellow (distinct from Loading)
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
        
    def _draw_selected_hand_label(self, frame: np.ndarray, selected_hand: str = None, confidence: float = 0.0, frame_idx: int = 0) -> np.ndarray:
        """Draw selected hand label and facing direction in top-right corner with small font"""
        if selected_hand is None:
            return frame
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Get facing direction from shot-specific data for current frame
        facing_direction = 'Unknown'
        if hasattr(self.shot_detector, 'frame_shots') and frame_idx < len(self.shot_detector.frame_shots):
            shot_id = self.shot_detector.frame_shots[frame_idx]
            if shot_id is not None and hasattr(self, 'shot_normalization_data'):
                shot_data = self.shot_normalization_data.get(shot_id, {})
                facing_direction = shot_data.get('facing_direction', 'Unknown')
        
        # Fallback to global facing direction if shot-specific data not available
        if facing_direction == 'Unknown':
            facing_direction = getattr(self, 'facing_direction', 'Unknown')
        
        # Create hand label text
        hand_text = f"{selected_hand.upper()} ({confidence:.0f}%)"
        # Create direction label text  
        direction_text = f"Facing: {facing_direction.upper()}"
        
        # Get shot information for current frame
        shot_text = ""
        torso_text = ""
        if hasattr(self.shot_detector, 'frame_shots') and frame_idx < len(self.shot_detector.frame_shots):
            shot_id = self.shot_detector.frame_shots[frame_idx]
            if shot_id is not None:
                shot_text = f"Shot {shot_id}"
                # Get torso information for this shot
                if hasattr(self.shot_detector, 'shots'):
                    for shot in self.shot_detector.shots:
                        if shot['shot_id'] == shot_id:
                            torso_value = shot.get('fixed_torso', 0.0)
                            torso_text = f"Torso: {torso_value:.4f}"
                            break
                # If no shot found, check if it's a fallback
                if not torso_text and hasattr(self, 'fallback_torso_length'):
                    torso_text = f"Fallback: {self.fallback_torso_length:.4f}"
        
        # Font settings - larger size
        font_scale = 0.8
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate text size for all lines
        (hand_width, hand_height), baseline = cv2.getTextSize(hand_text, font, font_scale, font_thickness)
        (dir_width, dir_height), _ = cv2.getTextSize(direction_text, font, font_scale, font_thickness)
        if shot_text:
            (shot_width, shot_height), _ = cv2.getTextSize(shot_text, font, font_scale, font_thickness)
        else:
            shot_width, shot_height = 0, 0
        if torso_text:
            (torso_width, torso_height), _ = cv2.getTextSize(torso_text, font, font_scale, font_thickness)
        else:
            torso_width, torso_height = 0, 0
        
        # Add padding to the background rectangle
        padding = 2
        bg_width = max(hand_width, dir_width, shot_width, torso_width) + padding * 2
        bg_height = hand_height + dir_height + shot_height + torso_height + padding * 5  # 5 padding for 4 lines
        
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
        
        # Draw hand text (first line)
        hand_text_x = bg_x + padding
        hand_text_y = bg_y + hand_height + padding - 1
        cv2.putText(frame, hand_text, (hand_text_x, hand_text_y), 
                font, font_scale, (255, 255, 255), font_thickness)
        
        # Draw direction text (second line)
        dir_text_x = bg_x + padding
        dir_text_y = hand_text_y + dir_height + padding
        cv2.putText(frame, direction_text, (dir_text_x, dir_text_y), 
                font, font_scale, (255, 255, 255), font_thickness)
        
        # Draw shot text (third line) if available
        if shot_text:
            shot_text_x = bg_x + padding
            shot_text_y = dir_text_y + shot_height + padding
            cv2.putText(frame, shot_text, (shot_text_x, shot_text_y), 
                font, font_scale, (255, 255, 255), font_thickness)
        
        # Draw torso text (fourth line) if available
        if torso_text:
            torso_text_x = bg_x + padding
            torso_text_y = shot_text_y + torso_height + padding
            cv2.putText(frame, torso_text, (torso_text_x, torso_text_y), 
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

        # Stage 1: Check if proximity difference is significant
        proximity_difference = abs(left_proximity_ratio - right_proximity_ratio)
        proximity_threshold = 0.2  # 10% difference threshold

        if proximity_difference > proximity_threshold:
            # Significant difference in proximity - use proximity as primary criterion
            selected_hand = "left" if left_proximity_ratio > right_proximity_ratio else "right"
            confidence = max(left_proximity_ratio, right_proximity_ratio) * 100
        else:
            # Similar proximity - use detection stability as secondary criterion
            selected_hand = "left" if left_stability_score > right_stability_score else "right"
            confidence = max(left_stability_score, right_stability_score) * 100
        
        return selected_hand, confidence

    def select_primary_hand_from_original_data(self) -> Tuple[str, float]:
        """
        Select the primary hand based on original (non-normalized) data.
        Used during phase detection before normalization.
        
        Returns:
            Tuple of (selected_hand, confidence_score)
        """
        print(f"\n🤚 Selecting primary hand from original data for phase detection...")
        
        left_hand_stats = {"close_frames": 0, "total_detected": 0, "wrist_detected": 0, "elbow_detected": 0}
        right_hand_stats = {"close_frames": 0, "total_detected": 0, "wrist_detected": 0, "elbow_detected": 0}
        
        # Use a reasonable threshold for original data (aspect ratio corrected coordinates)
        # Original data uses 0~aspect_ratio for x, 0~1 for y coordinates  
        # So we need to scale the threshold proportionally
        original_threshold = 0.3 * 2.0  # Roughly 2x the normalized threshold to account for different coordinate system
        
        # Debug counters
        total_frames = 0
        ball_detected_frames = 0
        pose_detected_frames = 0
        
        # Stage 1: Collect proximity and detection statistics from original data
        for i in range(min(len(self.pose_data), len(self.ball_data))):
            total_frames += 1
            pose = self.pose_data[i].get('pose', {})
            
            # Use the same ball data access pattern as phase detection
            ball_info = None
            ball_frame_data = self.ball_data[i]
            if isinstance(ball_frame_data, dict) and ball_frame_data.get('ball_detections'):
                ball_detections = ball_frame_data['ball_detections']
                if ball_detections and isinstance(ball_detections[0], dict):
                    ball_info = ball_detections[0]
            
            if ball_info:
                ball_detected_frames += 1
            if pose:
                pose_detected_frames += 1
            
            if not ball_info or not pose:
                continue
            
            # Get ball position from original data
            ball_x = ball_info.get('center_x', 0)
            ball_y = ball_info.get('center_y', 0)

            # Check left hand
            left_wrist = pose.get('left_wrist')
            left_elbow = pose.get('left_elbow')
            if left_wrist:
                left_wrist_x = left_wrist.get('x', 0)
                left_wrist_y = left_wrist.get('y', 0)
                left_distance = ((ball_x - left_wrist_x)**2 + (ball_y - left_wrist_y)**2)**0.5

                if left_distance < original_threshold:
                    left_hand_stats["close_frames"] += 1
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

                if right_distance < original_threshold:
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


        # Stage 1: Check if proximity difference is significant
        proximity_difference = abs(left_proximity_ratio - right_proximity_ratio)
        proximity_threshold = 0.2  # 20% difference threshold

        if proximity_difference > proximity_threshold:
            # Significant difference in proximity - use proximity as primary criterion
            selected_hand = "left" if left_proximity_ratio > right_proximity_ratio else "right"
            confidence = max(left_proximity_ratio, right_proximity_ratio) * 100
        else:
            # Similar proximity - use detection stability as secondary criterion
            selected_hand = "left" if left_stability_score > right_stability_score else "right"
            confidence = max(left_stability_score, right_stability_score) * 100
        
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

    def _draw_shot_info_label(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw shot information label in bottom-left corner"""
        if not hasattr(self.shot_detector, 'frame_shots') or frame_idx >= len(self.shot_detector.frame_shots):
            return frame
        
        shot_id = self.shot_detector.frame_shots[frame_idx]
        if shot_id is None:
            return frame
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Get shot metadata for torso info
        shot_info = None
        for shot in self.shot_detector.shots:
            if shot['shot_id'] == shot_id:
                shot_info = shot
                break
        
        # Create shot label text with torso info
        shot_text = f"Shot {shot_id}"
        if shot_info and shot_info.get('fixed_torso'):
            torso_text = f"Torso: {shot_info['fixed_torso']:.3f}"
        else:
            torso_text = "Torso: Not Fixed"
            
        # Font settings
        font_scale = 0.6
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate text size for both lines
        (shot_text_width, shot_text_height), _ = cv2.getTextSize(shot_text, font, font_scale, font_thickness)
        (torso_text_width, torso_text_height), _ = cv2.getTextSize(torso_text, font, font_scale, font_thickness)
        
        text_width = max(shot_text_width, torso_text_width)
        text_height = shot_text_height + torso_text_height + 3  # 3px spacing between lines
        
        # Add padding to the background rectangle
        padding = 3
        bg_width = text_width + padding * 2
        bg_height = text_height + padding * 2
        
        # Position in top-right corner
        bg_x = w - bg_width - 5
        bg_y = 5
        
        # Shot color (different color for each shot)
        shot_colors = [
            (0, 255, 0),    # Green for shot 1
            (255, 0, 0),    # Blue for shot 2
            (0, 0, 255),    # Red for shot 3
            (255, 255, 0),  # Cyan for shot 4
            (255, 0, 255),  # Magenta for shot 5
        ]
        shot_color = shot_colors[(shot_id - 1) % len(shot_colors)]
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                    (bg_x, bg_y), 
                    (bg_x + bg_width, bg_y + bg_height), 
                    (0, 0, 0), -1)
        
        # Draw colored indicator at left edge
        indicator_width = 3
        cv2.rectangle(frame, 
                    (bg_x, bg_y), 
                    (bg_x + indicator_width, bg_y + bg_height), 
                    shot_color, -1)
        
        # Draw shot text (first line)
        text_x = bg_x + padding
        text_y = bg_y + shot_text_height + padding - 1
        cv2.putText(frame, shot_text, (text_x, text_y), 
                font, font_scale, (255, 255, 255), font_thickness)
        
        # Draw torso text (second line)
        text_y2 = text_y + torso_text_height + 3
        cv2.putText(frame, torso_text, (text_x, text_y2), 
                font, font_scale, (200, 200, 200), font_thickness)
        
        return frame

    def create_normalized_analysis_video(self, video_path: str, output_path: str) -> bool:
        """Generate normalized data visualization video (shot-based normalized data only)"""
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
                
                # Draw normalized data on frame
                if frame_count < len(self.normalized_data):
                    normalized_frame_data = self.normalized_data[frame_count]
                    
                    # Draw normalized pose skeleton
                    if 'normalized_pose' in normalized_frame_data:
                        frame = self._draw_normalized_pose_skeleton(frame, frame_count, self.normalized_data)
                    
                    # Draw normalized ball
                    if 'normalized_ball' in normalized_frame_data:
                        frame = self._draw_normalized_ball(frame, frame_count, self.normalized_data)
                
                # Draw phase label
                if self.phases and frame_count < len(self.phases):
                    frame = self._draw_phase_label(frame, frame_count, "Normalized", self.phases)
                
                # Add selected hand label
                frame = self._draw_selected_hand_label(frame, self.selected_hand, self.selected_hand_confidence, frame_count)
                
                # Add scaling factor information
                if frame_count < len(self.normalized_data):
                    frame = self._draw_scaling_info_label(frame, frame_count)
                
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            print(f"✅ Normalized data visualization video generated: {output_path}")
            print(f"📊 Total processed frames: {frame_count}")
            print("Shot-based normalized data with torso-relative coordinates")
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate normalized visualization: {e}")
            return False

    def _draw_normalized_pose_skeleton(self, frame: np.ndarray, frame_idx: int, normalized_data: List[Dict]) -> np.ndarray:
        """Draw normalized pose skeleton"""
        if frame_idx >= len(normalized_data):
            return frame
        
        frame_data = normalized_data[frame_idx]
        normalized_pose = frame_data.get('normalized_pose', {})
        
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
        
        # Draw connections
        for connection in connections:
            start_key, end_key = connection
            
            if start_key in normalized_pose and end_key in normalized_pose:
                start_point = normalized_pose[start_key]
                end_point = normalized_pose[end_key]
                
                if (start_point.get('confidence', 0) > 0.3 and 
                    end_point.get('confidence', 0) > 0.3):
                    
                    # Convert normalized coordinates to pixel coordinates
                    start_x = int(start_point['x'] * w)
                    start_y = int(start_point['y'] * h)
                    end_x = int(end_point['x'] * w)
                    end_y = int(end_point['y'] * h)
                    
                    # Draw line
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Draw keypoints
        for key, point in normalized_pose.items():
            if point.get('confidence', 0) > 0.3:
                x = int(point['x'] * w)
                y = int(point['y'] * h)
                
                # Different colors for different keypoint types
                if 'shoulder' in key:
                    color = (255, 0, 0)  # Blue
                elif 'elbow' in key:
                    color = (0, 255, 0)  # Green
                elif 'wrist' in key:
                    color = (0, 0, 255)  # Red
                elif 'hip' in key:
                    color = (255, 255, 0)  # Cyan
                elif 'knee' in key:
                    color = (255, 0, 255)  # Magenta
                elif 'ankle' in key:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (255, 255, 255)  # White
                
                cv2.circle(frame, (x, y), 4, color, -1)
        
        return frame

    def _draw_normalized_ball(self, frame: np.ndarray, frame_idx: int, normalized_data: List[Dict]) -> np.ndarray:
        """Draw normalized ball"""
        if frame_idx >= len(normalized_data):
            return frame
        
        frame_data = normalized_data[frame_idx]
        normalized_ball = frame_data.get('normalized_ball', {})
        
        if not normalized_ball or not normalized_ball.get('detected', False):
            return frame
        
        h, w = frame.shape[:2]
        
        # Get ball coordinates
        ball_x = normalized_ball.get('x', 0)
        ball_y = normalized_ball.get('y', 0)
        confidence = normalized_ball.get('confidence', 0)
        
        if confidence > 0.3:
            # Convert normalized coordinates to pixel coordinates
            pixel_x = int(ball_x * w)
            pixel_y = int(ball_y * h)
            
            # Draw ball circle
            cv2.circle(frame, (pixel_x, pixel_y), 8, (0, 255, 255), -1)  # Yellow ball
            cv2.circle(frame, (pixel_x, pixel_y), 8, (0, 0, 0), 2)  # Black border
            
            # Draw confidence text
            conf_text = f"{confidence:.1f}"
            cv2.putText(frame, conf_text, (pixel_x + 10, pixel_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def _draw_scaling_info_label(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw scaling factor information in bottom-right corner"""
        if frame_idx >= len(self.normalized_data):
            return frame
        
        frame_data = self.normalized_data[frame_idx]
        scaling_factor = frame_data.get('scaling_factor', 0)
        shot_id = frame_data.get('shot', None)
        shot_applied = frame_data.get('shot_normalization_applied', False)
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Create scaling info text
        if shot_applied:
            scaling_text = f"Shot {shot_id} Torso: {scaling_factor:.4f}"
        else:
            scaling_text = f"Fallback Torso: {scaling_factor:.4f}"
        
        # Font settings
        font_scale = 0.6
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(scaling_text, font, font_scale, font_thickness)
        
        # Add padding to the background rectangle
        padding = 3
        bg_width = text_width + padding * 2
        bg_height = text_height + padding * 2
        
        # Position in bottom-right corner
        bg_x = w - bg_width - 5
        bg_y = h - bg_height - 5
        
        # Color based on shot application
        bg_color = (0, 255, 0) if shot_applied else (128, 128, 128)  # Green for shot, Gray for fallback
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                    (bg_x, bg_y), 
                    (bg_x + bg_width, bg_y + bg_height), 
                    (0, 0, 0), -1)
        
        # Draw colored indicator at right edge
        indicator_width = 3
        cv2.rectangle(frame, 
                    (bg_x + bg_width - indicator_width, bg_y), 
                    (bg_x + bg_width, bg_y + bg_height), 
                    bg_color, -1)
        
        # Draw scaling text
        text_x = bg_x + padding
        text_y = bg_y + text_height + padding - 1
        cv2.putText(frame, scaling_text, (text_x, text_y), 
                font, font_scale, (255, 255, 255), font_thickness)
        
        return frame

    def _calculate_torso_from_all_frames(self) -> float:
        """
        Calculate torso length from all frames in the video (3rd fallback for torso)
        """
        confidence_threshold = 0.3  # Same as other torso calculations
        print(f"   📊 Calculating torso from all frames in video...")
        
        torso_values = []
        valid_frames = 0
        
        for i, frame_data in enumerate(self.pose_data):
            pose = frame_data.get('pose', {})
            
            # Get keypoints
            left_shoulder = pose.get('left_shoulder', {})
            right_shoulder = pose.get('right_shoulder', {})
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            
            valid_torso_lengths = []
            
            # Check left side torso (신뢰도 기반)
            if (isinstance(left_shoulder, dict) and isinstance(left_hip, dict) and
                'x' in left_shoulder and 'y' in left_shoulder and
                'x' in left_hip and 'y' in left_hip):
                
                left_shoulder_conf = left_shoulder.get('confidence', 1.0)
                left_hip_conf = left_hip.get('confidence', 1.0)
                left_avg_conf = (left_shoulder_conf + left_hip_conf) / 2
                
                if left_avg_conf >= confidence_threshold:
                    left_torso_length = ((left_shoulder['x'] - left_hip['x'])**2 + 
                                       (left_shoulder['y'] - left_hip['y'])**2)**0.5
                    if left_torso_length > 0:
                        valid_torso_lengths.append(left_torso_length)
            
            # Check right side torso (신뢰도 기반)
            if (isinstance(right_shoulder, dict) and isinstance(right_hip, dict) and
                'x' in right_shoulder and 'y' in right_shoulder and
                'x' in right_hip and 'y' in right_hip):
                
                right_shoulder_conf = right_shoulder.get('confidence', 1.0)
                right_hip_conf = right_hip.get('confidence', 1.0)
                right_avg_conf = (right_shoulder_conf + right_hip_conf) / 2
                
                if right_avg_conf >= confidence_threshold:
                    right_torso_length = ((right_shoulder['x'] - right_hip['x'])**2 + 
                                        (right_shoulder['y'] - right_hip['y'])**2)**0.5
                    if right_torso_length > 0:
                        valid_torso_lengths.append(right_torso_length)
            
            # Calculate frame torso (average of valid measurements)
            if len(valid_torso_lengths) > 0:
                frame_torso = np.mean(valid_torso_lengths)
                torso_values.append(frame_torso)
                valid_frames += 1
        
        if len(torso_values) >= 10:  # 최소 10프레임 이상 필요
            overall_torso = np.mean(torso_values)
            print(f"   ✅ Overall torso: {overall_torso:.4f} (from {valid_frames}/{len(self.pose_data)} frames)")
            return overall_torso
        else:
            print(f"   ⚠️ Not enough valid torso measurements ({len(torso_values)}/10)")
            return None  # No fallback value

    def _get_torso_from_phase_detection(self) -> float:
        """Get stable torso length from phase detection result (deprecated - use shot-based torso)"""
        if hasattr(self, 'phase_detector') and self.phase_detector is not None:
            if hasattr(self.phase_detector, 'transition_reference_torso') and self.phase_detector.transition_reference_torso is not None:
                torso_length = self.phase_detector.transition_reference_torso
                print(f"   ✅ Using phase detection torso: {torso_length:.4f}")
                return torso_length
            else:
                print(f"   ⚠️ Phase detection torso not available, calculating from all frames")
                return self._calculate_torso_from_all_frames()
        else:
            print(f"   ⚠️ Phase detector not available, calculating from all frames")
            return self._calculate_torso_from_all_frames()

    def _determine_facing_direction_from_all_frames(self) -> tuple:
        """Determine facing direction from all frames in video using voting system (3rd fallback for direction)"""
        print(f"   📊 Determining facing direction from all frames using voting...")
        
        direction_votes = []
        valid_frames = 0
        
        for i, frame_data in enumerate(self.pose_data):
            pose = frame_data.get('pose', {})
            
            # Get hip and arm positions
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            left_wrist = pose.get('left_wrist', {})
            right_wrist = pose.get('right_wrist', {})
            left_shoulder = pose.get('left_shoulder', {})
            right_shoulder = pose.get('right_shoulder', {})
            
            # Determine center reference point (prefer hip, fallback to shoulder)
            center_x = None
            reference_type = None
            
            # Try hip center first
            if (isinstance(left_hip, dict) and isinstance(right_hip, dict) and
                'x' in left_hip and 'x' in right_hip):
                center_x = (left_hip['x'] + right_hip['x']) / 2
                reference_type = "Hip center"
            # Fallback to shoulder center
            elif (isinstance(left_shoulder, dict) and isinstance(right_shoulder, dict) and
                  'x' in left_shoulder and 'x' in right_shoulder):
                center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                reference_type = "Shoulder center"
            
            if center_x is None:
                continue
            
            # Check arm positions relative to center
            arm_direction = None
            if isinstance(left_wrist, dict) and 'x' in left_wrist:
                if left_wrist['x'] > center_x:  # 왼팔이 중심보다 오른쪽에 있음 → 오른쪽을 보고 있음
                    arm_direction = 'right'
                else:
                    arm_direction = 'left'
            elif isinstance(right_wrist, dict) and 'x' in right_wrist:
                if right_wrist['x'] < center_x:  # 오른팔이 중심보다 왼쪽에 있음 → 왼쪽을 보고 있음
                    arm_direction = 'left'
                else:
                    arm_direction = 'right'
            
            if arm_direction:
                direction_votes.append(arm_direction)
                valid_frames += 1
        
        # Determine final direction
        if len(direction_votes) >= 10:  # 최소 10프레임 이상 필요
            from collections import Counter
            direction_count = Counter(direction_votes)
            facing_direction = direction_count.most_common(1)[0][0]
            
            # Determine reference hip
            reference_hip_side = 'right' if facing_direction == 'right' else 'left'
            
            print(f"   ✅ Overall direction: {facing_direction} (from {valid_frames}/{len(self.pose_data)} frames)")
            print(f"   📊 Direction votes: {dict(direction_count)}")
            print(f"   📏 Reference hip: {reference_hip_side}")
            
            return facing_direction, reference_hip_side
        else:
            print(f"   ⚠️ Not enough valid direction measurements ({len(direction_votes)}/10)")
            return None, None

    def _determine_facing_direction_from_start(self) -> tuple:
        """Fallback method: determine facing direction from first 4 frames"""
        fps = getattr(self, 'video_fps', 30.0)
        required_frames = max(3, int(4 * (fps / 30.0)))
        
        print(f"   📊 Determining direction from first {required_frames} frames...")
        
        direction_votes = []
        
        for i in range(min(required_frames, len(self.pose_data))):
            frame_data = self.pose_data[i]
            pose = frame_data.get('pose', {})
            
            # Get hip and arm positions
            left_hip = pose.get('left_hip', {})
            right_hip = pose.get('right_hip', {})
            left_wrist = pose.get('left_wrist', {})
            right_wrist = pose.get('right_wrist', {})
            left_shoulder = pose.get('left_shoulder', {})
            right_shoulder = pose.get('right_shoulder', {})
            
            # Determine center reference point (prefer hip, fallback to shoulder)
            center_x = None
            reference_type = None
            
            # Try hip center first
            if (isinstance(left_hip, dict) and isinstance(right_hip, dict) and
                'x' in left_hip and 'x' in right_hip):
                center_x = (left_hip['x'] + right_hip['x']) / 2
                reference_type = "Hip center"
            # Fallback to shoulder center
            elif (isinstance(left_shoulder, dict) and isinstance(right_shoulder, dict) and
                  'x' in left_shoulder and 'x' in right_shoulder):
                center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
                reference_type = "Shoulder center"
            
            if center_x is None:
                continue
            
            # Check arm positions relative to center
            arm_direction = None
            if isinstance(left_wrist, dict) and 'x' in left_wrist:
                if left_wrist['x'] > center_x:  # 왼팔이 중심보다 오른쪽에 있음 → 오른쪽을 보고 있음
                    arm_direction = 'right'
                else:
                    arm_direction = 'left'
            elif isinstance(right_wrist, dict) and 'x' in right_wrist:
                if right_wrist['x'] < center_x:  # 오른팔이 중심보다 왼쪽에 있음 → 왼쪽을 보고 있음
                    arm_direction = 'left'
                else:
                    arm_direction = 'right'
            
            if arm_direction:
                direction_votes.append(arm_direction)
        
        # Determine final direction
        if direction_votes:
            from collections import Counter
            direction_count = Counter(direction_votes)
            facing_direction = direction_count.most_common(1)[0][0]
            
            # Determine reference hip
            reference_hip_side = 'right' if facing_direction == 'right' else 'left'
            
            print(f"   ✅ First frames direction: {facing_direction}")
            print(f"   📊 Direction votes: {dict(direction_count)}")
            print(f"   📏 Reference hip: {reference_hip_side}")
            
            return facing_direction, reference_hip_side
        else:
            print(f"   ⚠️ Could not determine direction from first frames, using overall frames")
            return self._determine_facing_direction_from_all_frames()

def main():
    """Main execution function"""
    analyzer = BasketballShootingAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 