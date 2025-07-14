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

    def get_folder_name_from_path(self, video_path: str) -> str:
        """비디오 경로에서 폴더 이름을 추출합니다."""
        # video_path에서 폴더 이름 추출
        # 예: data/video/Standard/video1.mp4 -> Standard
        # 예: data/video/test/clips/video1.mov -> test
        path_parts = video_path.replace('\\', '/').split('/')
        if 'video' in path_parts:
            video_index = path_parts.index('video')
            if video_index + 1 < len(path_parts):
                folder_name = path_parts[video_index + 1]
                # test/clips 폴더의 경우 test로 반환
                if folder_name == 'test' and video_index + 2 < len(path_parts) and path_parts[video_index + 2] == 'clips':
                    return 'test'
                return folder_name
        return "unknown"

    def generate_visualization(self, video_path: str, overwrite_mode: bool = False) -> bool:
        """Generate visualization with folder-specific output directory"""
        try:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # 폴더별 출력 디렉토리 생성
            folder_name = self.get_folder_name_from_path(video_path)
            output_dir = os.path.join("data", "visualized_video", folder_name)
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f"{base_name}_analyzed.mp4")
            
            if not overwrite_mode and os.path.exists(output_path):
                print(f"⚠️ Visualization already exists: {os.path.basename(output_path)}")
                choice = input("Overwrite? (y/n): ").strip().lower()
                if choice != 'y':
                    print("Skipping visualization generation.")
                    return True
            
            print(f"🎨 Generating visualization: {os.path.basename(output_path)}")
            print(f"📁 Output directory: {output_dir}")
            
            # 기존 generate_visualization 메서드 호출
            result = super().generate_visualization(video_path, overwrite_mode)
            
            if result:
                # 파일을 올바른 위치로 이동
                old_output_path = os.path.join("data", "visualized_video", f"{base_name}_analyzed.mp4")
                if os.path.exists(old_output_path):
                    import shutil
                    shutil.move(old_output_path, output_path)
                    print(f"✅ Visualization saved to: {output_path}")
                    return True
                elif os.path.exists(output_path):
                    print(f"✅ Visualization saved to: {output_path}")
                    return True
                else:
                    print(f"❌ Visualization file was not created. Please check codec settings.")
                    return False
            else:
                print(f"❌ Visualization failed.")
                return False
            
        except Exception as e:
            print(f"❌ Failed to generate visualization: {e}")
            return False

    def prompt_video_selection(self) -> Optional[str]:
        """Prompt user to select processing mode (test 폴더 추가)"""
        # 비디오 목록 갱신
        self.available_videos = self.list_available_videos()
        
        # 폴더별 분류
        standard_videos = [v for v in self.available_videos if 'Standard' in v]
        edgecase_videos = [v for v in self.available_videos if 'EdgeCase' in v]
        test_videos = [v for v in self.available_videos if 'test' in v.lower()]
        
        print("\n🎬 STEP 0: Select processing mode")
        print("=" * 50)
        print("Available processing options:")
        print(f"[1] Single video selection ({len(self.available_videos)} total videos)")
        print(f"[2] Process all Standard videos ({len(standard_videos)} videos)")
        print(f"[3] Process all EdgeCase videos ({len(edgecase_videos)} videos)")
        print(f"[4] Process all Test videos ({len(test_videos)} videos)")
        print(f"[5] Process all videos ({len(self.available_videos)} videos)")
        print("[6] Cancel")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-6): ").strip()
                
                if choice == "1":
                    # Single video selection
                    print("\nAvailable videos:")
                    for i, video in enumerate(self.available_videos, 1):
                        print(f"  [{i}] {os.path.basename(video)}")
                    video_choice = input("Enter the number or file name: ").strip()
                    if video_choice.isdigit():
                        idx = int(video_choice) - 1
                    if 0 <= idx < len(self.available_videos):
                        return self.available_videos[idx]
                    else:
                        print("❌ Invalid number.")
                        continue
                    for video in self.available_videos:
                        if os.path.basename(video) == video_choice:
                            return video
                    print("❌ Invalid selection. Please try again.")
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
                    # test 폴더 선택 시 바로 test_video_selection 호출
                    test_selection = self.prompt_test_video_selection()
                    if test_selection:
                        if test_selection.startswith("test_clips_"):
                            print(f"✅ Selected: Process all clips in test folder")
                            return "test_all"
                        else:
                            print(f"✅ Selected: Process combined_output.mov")
                            return test_selection
                    else:
                        print("❌ Test selection canceled.")
                        continue
                elif choice == "5":
                    if self.available_videos:
                        print(f"✅ Selected: Process all videos ({len(self.available_videos)} videos)")
                        return "all_videos"
                    else:
                        print("❌ No videos found.")
                        continue
                elif choice == "6":
                    print("❌ Analysis canceled.")
                    return None
                else:
                    print("❌ Invalid choice. Please enter 1-6.")
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
                        # clips 폴더의 모든 비디오 파일 찾기
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
    
    # Handle special keywords for batch processing
    if selected_video in ["standard_all", "edgecase_all", "test_all", "all_videos"]:
        # Get video list based on selection
        available_videos = pipeline.list_available_videos()
        
        if selected_video == "standard_all":
            videos_to_process = [v for v in available_videos if 'Standard' in v]
        elif selected_video == "edgecase_all":
            videos_to_process = [v for v in available_videos if 'EdgeCase' in v]
        elif selected_video == "test_all":
            # clips 폴더의 모든 비디오를 개별적으로 처리
            clips_dir = os.path.join("data", "video", "test", "clips")
            video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
            videos_to_process = []
            for ext in video_extensions:
                videos_to_process.extend(glob.glob(os.path.join(clips_dir, f"*{ext}")))
                videos_to_process.extend(glob.glob(os.path.join(clips_dir, f"*{ext.upper()}")))
            
            print(f"\n🔄 Processing {len(videos_to_process)} clips individually...")
            print("=" * 50)
            
            success_count = 0
            error_summary = []
            
            for i, video_path in enumerate(videos_to_process, 1):
                print(f"\n📹 Processing clip {i}/{len(videos_to_process)}: {os.path.basename(video_path)}")
                print("-" * 40)
                
                try:
                    success = pipeline.run_full_pipeline(video_path, overwrite_mode)
                    if success:
                        success_count += 1
                        print(f"✅ Successfully processed: {os.path.basename(video_path)}")
                    else:
                        error_msg = f"Failed to process: {os.path.basename(video_path)}"
                        print(f"❌ {error_msg}")
                        error_summary.append(error_msg)
                except Exception as e:
                    error_msg = f"Error processing {os.path.basename(video_path)}: {e}"
                    print(f"❌ {error_msg}")
                    error_summary.append(error_msg)
            
            print(f"\n🎉 Clips processing completed!")
            print(f"Successfully processed: {success_count}/{len(videos_to_process)} clips")
            
            if error_summary:
                print(f"\n❌ Errors occurred during processing:")
                for error in error_summary:
                    print(f"  - {error}")
            
            return  # clips 처리는 여기서 완료되므로 main 함수 종료
        else:  # all_videos
            videos_to_process = available_videos
        
        # 배치 처리 (clips 제외)
        print(f"\n🔄 Processing {len(videos_to_process)} videos in batch mode...")
        print("=" * 50)
        
        success_count = 0
        error_summary = []
        
        for i, video_path in enumerate(videos_to_process, 1):
            print(f"\n📹 Processing video {i}/{len(videos_to_process)}: {os.path.basename(video_path)}")
            print("-" * 40)
            
            try:
                success = pipeline.run_full_pipeline(video_path, overwrite_mode)
                if success:
                    success_count += 1
                    print(f"✅ Successfully processed: {os.path.basename(video_path)}")
                else:
                    error_msg = f"Failed to process: {os.path.basename(video_path)}"
                    print(f"❌ {error_msg}")
                    error_summary.append(error_msg)
            except Exception as e:
                error_msg = f"Error processing {os.path.basename(video_path)}: {e}"
                print(f"❌ {error_msg}")
                error_summary.append(error_msg)
        
        print(f"\n🎉 Batch processing completed!")
        print(f"Successfully processed: {success_count}/{len(videos_to_process)} videos")
        
        if error_summary:
            print(f"\n❌ Errors occurred during processing:")
            for error in error_summary:
                print(f"  - {error}")
        
    else:
        # Single video processing (including test mov files)
        success = pipeline.run_full_pipeline(selected_video, overwrite_mode)
        
        if success:
            folder_name = pipeline.get_folder_name_from_path(selected_video)
            print("\n🎉 Pipeline execution completed!")
            print("Generated files:")
            print(f"  • Original data: data/extracted_data/{os.path.splitext(os.path.basename(selected_video))[0]}_*_original.json")
            print(f"  • Normalized data: data/extracted_data/{os.path.splitext(os.path.basename(selected_video))[0]}_*_normalized.json")
            print(f"  • Analysis result: data/results/{os.path.splitext(os.path.basename(selected_video))[0]}_analysis.json")
            print(f"  • Visualization video: data/visualized_video/{folder_name}/{os.path.splitext(os.path.basename(selected_video))[0]}_analyzed.mp4")
        else:
            print("\n❌ Pipeline execution failed!")

if __name__ == "__main__":
    main() 