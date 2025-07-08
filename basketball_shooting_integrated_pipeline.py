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

# 기존 분석 파이프라인 import
from basketball_shooting_analyzer import BasketballShootingAnalyzer
# 추출 파이프라인 import
from pose_extraction.pose_extraction_pipeline import PoseExtractionPipeline
from ball_extraction.ball_extraction_pipeline import BallExtractionPipeline

class BasketballShootingIntegratedPipeline(BasketballShootingAnalyzer):
    def __init__(self):
        super().__init__()
        self.references_dir = "data"
        self.video_dir = os.path.join(self.references_dir, "video")
        self.extracted_data_dir = os.path.join(self.references_dir, "extracted_data")
        self.pose_pipeline = PoseExtractionPipeline(output_dir=self.extracted_data_dir)
        self.ball_pipeline = BallExtractionPipeline(output_dir=self.extracted_data_dir)
        
        print("🏀 Basketball Shooting Integrated Pipeline Initialized")
        print("=" * 50)

    def run_full_pipeline(self, video_path: str, overwrite_mode: bool = False) -> bool:
        """
        Run the full pipeline: extraction → normalization → visualization
        
        Args:
            video_path: Path to the video file
            overwrite_mode: Overwrite mode
        
        Returns:
            Success status
        """
        print(f"🎬 Starting Full Pipeline: {os.path.basename(video_path)}")
        print("=" * 50)
        
        try:
            # STEP 1: Extract original data
            print("\n🔍 STEP 1: Extract original data")
            print("-" * 30)
            
            if not self._extract_original_data(video_path, overwrite_mode):
                print("❌ Failed to extract original data")
                return False
            
            # STEP 2: Load original data
            print("\n📂 STEP 2: Load original data")
            print("-" * 30)
            
            if not self.load_associated_data(video_path, overwrite_mode):
                print("❌ Failed to load original data")
                return False
            
            # STEP 3: Normalize and save data
            print("\n🔄 STEP 3: Normalize and save data")
            print("-" * 30)
            
            self.normalize_pose_data(video_path)
            
            # STEP 4: Segment shooting phases
            print("\n🎯 STEP 4: Segment shooting phases")
            print("-" * 30)
            
            self.segment_shooting_phases()
            
            # STEP 5: Save analysis results
            print("\n💾 STEP 5: Save analysis results")
            print("-" * 30)
            
            self.save_results(video_path, overwrite_mode)
            
            # STEP 6: Generate visualization
            print("\n🎨 STEP 6: Generate visualization")
            print("-" * 30)
            
            self.generate_visualization(video_path, overwrite_mode)
            
            print("\n🎉 Full pipeline completed!")
            print("=" * 50)
            return True
            
        except Exception as e:
            print(f"❌ Error occurred during pipeline execution: {e}")
            return False

    def _extract_original_data(self, video_path: str, overwrite_mode: bool = False) -> bool:
        """Extract original data (pose + ball)"""
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Check for existing original data files
        pose_original_file = os.path.join(self.extracted_data_dir, f"{base_name}_pose_original.json")
        ball_original_file = os.path.join(self.extracted_data_dir, f"{base_name}_ball_original.json")
        
        if not overwrite_mode and (os.path.exists(pose_original_file) or os.path.exists(ball_original_file)):
            print(f"⚠️ Existing original extraction data found:")
            if os.path.exists(pose_original_file):
                print(f"  - Pose data: {os.path.basename(pose_original_file)}")
            if os.path.exists(ball_original_file):
                print(f"  - Ball data: {os.path.basename(ball_original_file)}")
            choice = input("Overwrite and extract new data? (y/n): ").strip().lower()
            if choice != 'y':
                print("Using existing original extraction data.")
                return True
        
        try:
            # Extract pose data
            print("🔍 Extracting pose data...")
            pose_file = self.pose_pipeline.extract_poses(video_path, confidence_threshold=0.3)
            print(f"✅ Pose extraction completed: {os.path.basename(pose_file)}")
            
            # Extract ball data
            print("🔍 Extracting ball data...")
            ball_file = self.ball_pipeline.extract_ball_trajectory(
                video_path, conf_threshold=0.15, min_confidence=0.3, min_ball_size=10.0
            )
            print(f"✅ Ball extraction completed: {os.path.basename(ball_file)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to extract data: {e}")
            return False

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
                else:
                    print("❌ File not found.")
                    continue
                    
            except ValueError:
                print("❌ Invalid input.")
                continue

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
    
    # Initialize pipeline
    pipeline = BasketballShootingIntegratedPipeline()
    
    # Video selection
    selected_video = pipeline.prompt_video_selection()
    if not selected_video:
        print("❌ Video selection canceled.")
        return
    
    # Overwrite mode selection
    overwrite_mode = pipeline.prompt_overwrite_mode()
    if overwrite_mode is None:
        print("❌ Analysis canceled.")
        return
    
    # Run full pipeline
    success = pipeline.run_full_pipeline(selected_video, overwrite_mode)
    
    if success:
        print("\n🎉 Pipeline execution completed!")
        print("Generated files:")
        print(f"  • Original data: data/extracted_data/{os.path.splitext(os.path.basename(selected_video))[0]}_*_original.json")
        print(f"  • Normalized data: data/extracted_data/{os.path.splitext(os.path.basename(selected_video))[0]}_*_normalized.json")
        print(f"  • Analysis result: data/results/{os.path.splitext(os.path.basename(selected_video))[0]}_analysis.json")
        print(f"  • Visualization video: data/visualized_video/{os.path.splitext(os.path.basename(selected_video))[0]}_analyzed.mp4")
    else:
        print("\n❌ Pipeline execution failed!")

if __name__ == "__main__":
    main() 