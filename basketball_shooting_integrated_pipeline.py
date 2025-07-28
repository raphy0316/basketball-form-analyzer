# -*- coding: utf-8 -*-
"""
Integrated pipeline for extraction, normalization, and visualization from video
"""

import os
import glob
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import cv2 # Added for FPS extraction
import traceback
import threading
import concurrent.futures

# Import existing analysis pipeline
from basketball_shooting_analyzer import BasketballShootingAnalyzer
# Import extraction pipeline
from pose_extraction.pose_extraction_pipeline import PoseExtractionPipeline
from ball_extraction.ball_extraction_pipeline import BallExtractionPipeline

class BasketballShootingIntegratedPipeline:
    def __init__(self):
        self.references_dir = "data"
        self.video_dir = os.path.join(self.references_dir, "video")
        self.extracted_data_dir = os.path.join(self.references_dir, "extracted_data")
        self.pose_pipeline = PoseExtractionPipeline(output_dir=self.extracted_data_dir)
        self.ball_pipeline = BallExtractionPipeline(output_dir=self.extracted_data_dir)
        
        # Create analyzer instance
        self.analyzer = BasketballShootingAnalyzer()
        
        print("🏀 Basketball Shooting Integrated Pipeline Initialized")
        print("=" * 50)

    def prompt_extraction_mode(self) -> Optional[bool]:
        """Choose whether to use existing extraction data or extract new data"""
        print("\n⚙️ Extraction data mode selection")
        print("-" * 30)
        print("[1] Use existing extraction data (fast, for experiment repetition)")
        print("[2] New extraction (model re-execution)")
        print("[3] Cancel")
        while True:
            choice = input("\nSelection (1/2/3): ").strip()
            if choice == "1":
                return True  # Use existing data
            elif choice == "2":
                return False  # New extraction
            elif choice == "3":
                return None
            else:
                print("❌ Please enter a valid number (1, 2, 3)")

    def run_full_pipeline(self, video_path: str, overwrite_mode: bool = False, use_existing_extraction: bool = True) -> bool:
        """
        Run the full pipeline: extraction → normalization → visualization
        Args:
            video_path: Path to the video file
            overwrite_mode: Overwrite mode
            use_existing_extraction: Whether to use existing extraction data
        Returns:
            Success status
        """
        print(f"🎬 Starting Full Pipeline: {os.path.basename(video_path)}")
        print("=" * 50)
        try:
            # STEP 1: Extract original data
            print("\n🔍 STEP 1: Extract original data")
            print("-" * 30)
            if not self._extract_original_data(video_path, overwrite_mode, use_existing_extraction):
                print("❌ Failed to extract original data")
                return False
            
            # STEP 2: Load original data
            print("\n📂 STEP 2: Load original data")
            print("-" * 30)
            
            if not self.analyzer.load_associated_data(video_path, overwrite_mode):
                print("❌ Failed to load original data")
                return False
            
            # STEP 3: Normalize and save data
            print("\n🔄 STEP 3: Normalize and save data")
            print("-" * 30)
            print("  - Torso-based scaling normalization")
            print("  - Consistent scaling across different video resolutions")
            
            self.analyzer.normalize_pose_data(video_path)
            
            # STEP 4: Segment shooting phases
            print("\n🎯 STEP 4: Segment shooting phases")
            print("-" * 30)
            
            self.analyzer.segment_shooting_phases("hybrid_fps")
            print("✅ Phase segmentation completed (using hybrid FPS-based detector)")
            
            # STEP 5: Save analysis results
            print("\n💾 STEP 5: Save analysis results")
            print("-" * 30)
            
            self.analyzer.save_results(video_path, overwrite_mode)
            print("✅ Analysis results saved successfully")
            
            # STEP 6: Generate visualization
            print("\n🎨 STEP 6: Generate visualization")
            print("-" * 30)
            
            success = self.analyzer.generate_visualization(video_path, overwrite_mode)
            if success:
                print("✅ Visualization generated successfully")
            else:
                print("❌ Failed to generate visualization")
            
            print("\n🎉 Full pipeline completed!")
            print("=" * 50)
            return True
            
        except Exception as e:
            print(f"❌ Error occurred during pipeline execution: {e}")
            traceback.print_exc()  # Print full error stack trace
            return False

     # Define extraction functions for threading
    def _extract_pose(self, video_path: str) -> str:
        print("🔍 Extracting pose data with coordinate transformation...")
        print("  - MoveNet crop coordinates → Full frame coordinates")
        print("  - Aspect ratio correction applied to x-axis")
        pose_file = self.pose_pipeline.extract_poses(video_path, confidence_threshold=0.3)
        print(f"✅ Pose extraction completed: {os.path.basename(pose_file)}")
        return pose_file
    
    def _extract_ball(self, video_path: str) -> str:
        print("🔍 Extracting ball data with normalized coordinates...")
        print("  - YOLO using 0~1 normalized coordinates")
        print("  - Aspect ratio correction applied to x-axis")
        ball_file = self.ball_pipeline.extract_ball_trajectory(
            video_path, conf_threshold=0.15, min_confidence=0.3, min_ball_size=0.01
        )
        print(f"✅ Ball extraction completed: {os.path.basename(ball_file)}")
        return ball_file
    
    def _extract_original_data(self, video_path: str, overwrite_mode: bool = False, use_existing_extraction: bool = True) -> bool:
        """Extract original data with coordinate transformation (pose + ball)"""
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        pose_original_file = os.path.join(self.extracted_data_dir, f"{base_name}_pose_original.json")
        ball_original_file = os.path.join(self.extracted_data_dir, f"{base_name}_ball_original.json")
        
        if use_existing_extraction and (os.path.exists(pose_original_file) or os.path.exists(ball_original_file)):
            print(f"⚠️ Using existing extraction data:")
            if os.path.exists(pose_original_file):
                print(f"  - Pose data: {os.path.basename(pose_original_file)}")
            if os.path.exists(ball_original_file):
                print(f"  - Ball data: {os.path.basename(ball_original_file)}")
            return True
        
        # If overwrite_mode is True, always extract new data
        if not use_existing_extraction and not overwrite_mode and (os.path.exists(pose_original_file) or os.path.exists(ball_original_file)):
            choice = input("Overwrite and extract new data? (y/n): ").strip().lower()
            if choice != 'y':
                print("Using existing data.")
                return True
        
        try:
                # Use ThreadPoolExecutor for parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both extraction tasks
                pose_future = executor.submit(self._extract_pose, video_path)
                ball_future = executor.submit(self._extract_ball, video_path)
                
                # Wait for both to complete and get results
                pose_file = pose_future.result()
                ball_file = ball_future.result()
                
                print("🔄 Both pose and ball extraction completed in parallel")
            
            return True
    
        except Exception as e:
            print(f"❌ Failed to extract data: {e}")
            return False

    def get_folder_name_from_path(self, video_path: str) -> str:
        """비디오 경로에서 폴더 이름을 추출합니다."""
        # Extract folder name from video_path
        # Example: data/video/Standard/video1.mp4 -> Standard
        # Example: data/video/test/clips/video1.mov -> test
        path_parts = video_path.replace('\\', '/').split('/')
        if 'video' in path_parts:
            video_index = path_parts.index('video')
            if video_index + 1 < len(path_parts):
                folder_name = path_parts[video_index + 1]
                # For test/clips folder, return test
                if folder_name == 'test' and video_index + 2 < len(path_parts) and path_parts[video_index + 2] == 'clips':
                    return 'test'
                return folder_name
        return "unknown"

    def prompt_video_selection(self) -> Optional[List[str]]:
        """Prompt user to select processing mode with multiple selection support"""
        # Get available videos from analyzer
        self.available_videos = self.analyzer.list_available_videos()
        standard_videos = [v for v in self.available_videos if 'Standard' in v]
        edgecase_videos = [v for v in self.available_videos if 'EdgeCase' in v]
        bakke_videos = [v for v in self.available_videos if 'Bakke' in v]
        test_videos = [v for v in self.available_videos if 'test' in v.lower()]
        
        print("\n🎬 STEP 0: Select processing mode")
        print("=" * 50)
        print("Available processing options:")
        print(f"[1] Single video selection ({len(self.available_videos)} total videos)")
        print(f"[2] Process all Standard videos ({len(standard_videos)} videos)")
        print(f"[3] Process all EdgeCase videos ({len(edgecase_videos)} videos)")
        print(f"[4] Process all Bakke videos ({len(bakke_videos)} videos)")
        print(f"[5] Process all Test videos ({len(test_videos)} videos)")
        print(f"[6] Process all videos ({len(self.available_videos)} videos)")
        print("[7] Cancel")
        print("\n💡 Tip: You can select multiple options (e.g., '2,1,3' to process Standard → Single → EdgeCase)")
        
        while True:
            try:
                choice = input("\nEnter your choice(s) (e.g., 1 or 2,1,3): ").strip()
                
                if choice == "7":
                    print("❌ Analysis canceled.")
                    return None
                
                # Parse multiple selections
                selections = [s.strip() for s in choice.split(',')]
                selected_modes = []
                
                for selection in selections:
                    if selection == "1":
                        print(f"\n📹 Single video selection:")
                        print("Available videos:")
                        for i, video in enumerate(self.available_videos, 1):
                            print(f"  [{i}] {os.path.basename(video)}")
                        
                        video_choice = input("Enter the number or file name: ").strip()
                        if video_choice.isdigit():
                            idx = int(video_choice) - 1
                            if 0 <= idx < len(self.available_videos):
                                selected_modes.append(self.available_videos[idx])
                            else:
                                print("❌ Invalid number.")
                                continue
                        else:
                            for video in self.available_videos:
                                if os.path.basename(video) == video_choice:
                                    selected_modes.append(video)
                                    break
                            else:
                                print("❌ Invalid selection.")
                                continue
                    elif selection == "2":
                        if standard_videos:
                            selected_modes.append("standard_all")
                            print(f"✅ Added: Process all Standard videos ({len(standard_videos)} videos)")
                        else:
                            print("❌ No videos found in Standard folder.")
                            continue
                    elif selection == "3":
                        if edgecase_videos:
                            selected_modes.append("edgecase_all")
                            print(f"✅ Added: Process all EdgeCase videos ({len(edgecase_videos)} videos)")
                        else:
                            print("❌ No videos found in EdgeCase folder.")
                            continue
                    elif selection == "4":
                        if bakke_videos:
                            selected_modes.append("bakke_all")
                            print(f"✅ Added: Process all Bakke videos ({len(bakke_videos)} videos)")
                        else:
                            print("❌ No videos found in Bakke folder.")
                            continue
                    elif selection == "5":
                        # When test folder is selected, directly call test_video_selection
                        test_selection = self.prompt_test_video_selection()
                        if test_selection:
                            if test_selection.startswith("test_clips_"):
                                selected_modes.append("test_all")
                                print(f"✅ Added: Process all clips in test folder")
                            else:
                                selected_modes.append(test_selection)
                                print(f"✅ Added: Process combined_output.mov")
                        else:
                            print("❌ Test selection canceled.")
                            continue
                    elif selection == "6":
                        if self.available_videos:
                            selected_modes.append("all_videos")
                            print(f"✅ Added: Process all videos ({len(self.available_videos)} videos)")
                        else:
                            print("❌ No videos found.")
                            continue
                    else:
                        print(f"❌ Invalid choice: {selection}")
                        continue
                
                if selected_modes:
                    print(f"\n🎯 Selected processing order:")
                    for i, mode in enumerate(selected_modes, 1):
                        if isinstance(mode, str) and mode.endswith("_all"):
                            print(f"  {i}. {mode}")
                        else:
                            print(f"  {i}. {os.path.basename(mode) if isinstance(mode, str) else mode}")
                    return selected_modes
                else:
                    print("❌ No valid selections made.")
                    continue
                    
            except KeyboardInterrupt:
                print("\n❌ Analysis canceled.")
                return None

    def prompt_test_video_selection(self) -> Optional[str]:
        """test 폴더에서 combined_output.mov 또는 clips 폴더 선택"""
        test_dir = os.path.join(self.video_dir, "test")
        
        if not os.path.exists(test_dir):
            print("❌ Test folder not found.")
            return None
        
        combined_video = os.path.join(test_dir, "combined_output.mov")
        clips_dir = os.path.join(test_dir, "clips")
        
        print("\n🎬 Test folder selection")
        print("=" * 30)
        print("What would you like to process?")
        print("[1] combined_output.mov (single file)")
        print("[2] clips folder (all files in folder)")
        print("[3] Cancel")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == "1":
                    if os.path.exists(combined_video):
                        print(f"✅ Selected: combined_output.mov")
                        return combined_video
                    else:
                        print("❌ combined_output.mov not found in test folder.")
                        continue
                elif choice == "2":
                    if os.path.exists(clips_dir):
                        # Find all video files in clips folder
                        video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
                        clips_videos = []
                        for ext in video_extensions:
                            clips_videos.extend(glob.glob(os.path.join(clips_dir, f"*{ext}")))
                            clips_videos.extend(glob.glob(os.path.join(clips_dir, f"*{ext.upper()}")))
                        
                        if clips_videos:
                            print(f"✅ Selected: clips folder ({len(clips_videos)} videos)")
                            return f"test_clips_{len(clips_videos)}"
                        else:
                            print("❌ No video files found in clips folder.")
                            continue
                    else:
                        print("❌ clips folder not found in test folder.")
                        continue
                elif choice == "3":
                    print("❌ Test selection canceled.")
                    return None
                else:
                    print("❌ Invalid choice. Please enter 1-3.")
                    continue
            except KeyboardInterrupt:
                print("\n❌ Test selection canceled.")
                return None

    def prompt_overwrite_mode(self) -> Optional[bool]:
        """Prompt for overwrite mode selection"""
        print("\n⚙️ Select overwrite mode")
        print("-" * 30)
        print("How do you want to handle existing files?")
        print("[1] Overwrite (delete existing files and create new)")
        print("[2] Skip (skip if existing files exist)")
        print("[3] Cancel")
        
        while True:
            choice = input("\nSelect (1/2/3): ").strip()
            if choice == "1":
                return True
            elif choice == "2":
                return False
            elif choice == "3":
                return None
            else:
                print("❌ Invalid selection. Please choose 1, 2, or 3.")

def main():
    """Main execution function"""
    print("🏀 Basketball Shooting Integrated Pipeline")
    print("=" * 50)
    
    pipeline = BasketballShootingIntegratedPipeline()
    
    # Get video selection
    video_selections = pipeline.prompt_video_selection()
    if not video_selections:
        print("❌ No videos selected. Exiting.")
        return
    
    # Get overwrite mode
    overwrite_mode = pipeline.prompt_overwrite_mode()
    if overwrite_mode is None:
        print("❌ Overwrite mode selection canceled. Exiting.")
        return
    
    # Get extraction mode
    extraction_mode = pipeline.prompt_extraction_mode()
    if extraction_mode is None:
        print("❌ Extraction mode selection canceled. Exiting.")
        return
    
    # Process videos
    success_count = 0
    total_count = len(video_selections)
    
    for i, selection in enumerate(video_selections, 1):
        print(f"\n🎬 Processing {i}/{total_count}: {selection}")
        print("=" * 50)
        
        if isinstance(selection, str) and selection.endswith("_all"):
            # Process all videos in category
            if selection == "standard_all":
                videos = [v for v in pipeline.available_videos if 'Standard' in v]
            elif selection == "edgecase_all":
                videos = [v for v in pipeline.available_videos if 'EdgeCase' in v]
            elif selection == "bakke_all":
                videos = [v for v in pipeline.available_videos if 'Bakke' in v]
            elif selection == "test_all":
                test_dir = os.path.join(pipeline.video_dir, "test", "clips")
                video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
                videos = []
                for ext in video_extensions:
                    videos.extend(glob.glob(os.path.join(test_dir, f"*{ext}")))
                    videos.extend(glob.glob(os.path.join(test_dir, f"*{ext.upper()}")))
            elif selection == "all_videos":
                videos = pipeline.available_videos
            else:
                print(f"❌ Unknown selection: {selection}")
                continue
            
            print(f"📁 Processing {len(videos)} videos in {selection}")
            
            for video in videos:
                print(f"\n🎬 Processing: {os.path.basename(video)}")
                if pipeline.run_full_pipeline(video, overwrite_mode, extraction_mode):
                    success_count += 1
                    print(f"✅ Successfully processed: {os.path.basename(video)}")
                else:
                    print(f"❌ Failed to process: {os.path.basename(video)}")
        else:
            # Process single video
            if pipeline.run_full_pipeline(selection, overwrite_mode, extraction_mode):
                success_count += 1
                print(f"✅ Successfully processed: {os.path.basename(selection)}")
            else:
                print(f"❌ Failed to process: {os.path.basename(selection)}")
        
        print(f"\n🎉 Batch processing completed!")
    print(f"Successfully processed: {success_count}/{total_count} videos")
    
    if success_count < total_count:
        print(f"❌ Errors occurred during processing:")
        for selection in video_selections:
            if isinstance(selection, str) and selection.endswith("_all"):
                # Check if any videos in this category failed
                pass
            else:
                # Check if this single video failed
                pass
    
    print(f"\n🎉 All selected processing completed!")

if __name__ == "__main__":
    main() 