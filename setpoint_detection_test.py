#!/usr/bin/env python3
"""
Setpoint Detection Test

This script provides an interactive interface for testing setpoint detection
on basketball shooting videos. It allows users to select videos and visualize
detected setpoints.
"""

import os
import sys
import json
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Dict, Optional, Tuple
import numpy as np
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shooting_comparison.rising_analyzer import SetpointDetector


class SetpointVisualizer:
    """Visualizer for setpoint detection results."""
    
    def __init__(self):
        self.colors = {
            'setpoint': (0, 255, 0),      # Green
            'ball_trajectory': (255, 0, 0), # Red
            'wrist_trajectory': (0, 0, 255), # Blue
            'text': (255, 255, 255)        # White
        }
    
    def visualize_setpoints(self, video_path: str, pose_data: List[Dict], 
                           ball_data: List[Dict], setpoints: List[int],
                           output_path: str = None) -> str:
        """
        Visualize setpoints on video.
        
        Args:
            video_path: Path to input video
            pose_data: Pose data for each frame
            ball_data: Ball data for each frame
            setpoints: List of setpoint frame indices
            output_path: Output video path (optional)
            
        Returns:
            Path to output video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Generate output path if not provided
        if not output_path:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"setpoint_visualization_{base_name}.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        setpoint_idx = 0
        
        print(f"🎬 Creating setpoint visualization...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if current frame is a setpoint
            is_setpoint = frame_count in setpoints
            
            # Draw setpoint indicator
            if is_setpoint:
                # Draw green circle around frame
                cv2.circle(frame, (width//2, height//2), min(width, height)//4, 
                          self.colors['setpoint'], 5)
                
                # Draw setpoint text
                cv2.putText(frame, f"SETPOINT {setpoint_idx + 1}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                           self.colors['setpoint'], 3)
                
                setpoint_idx += 1
            
            # Draw ball trajectory if available
            if frame_count < len(ball_data):
                try:
                    ball_frame = ball_data[frame_count]
                    if ball_frame and 'center_x' in ball_frame and 'center_y' in ball_frame:
                        # Convert normalized coordinates to pixel coordinates
                        ball_x = int(ball_frame['center_x'] * width)
                        ball_y = int(ball_frame['center_y'] * height)
                        
                        # Draw ball position
                        cv2.circle(frame, (ball_x, ball_y), 10, self.colors['ball_trajectory'], -1)
                        
                        # Draw trajectory line (connect to previous frame)
                        if frame_count > 0 and frame_count - 1 < len(ball_data):
                            prev_ball = ball_data[frame_count - 1]
                            if prev_ball and 'center_x' in prev_ball and 'center_y' in prev_ball:
                                prev_x = int(prev_ball['center_x'] * width)
                                prev_y = int(prev_ball['center_y'] * height)
                                cv2.line(frame, (prev_x, prev_y), (ball_x, ball_y), 
                                       self.colors['ball_trajectory'], 2)
                except (KeyError, IndexError):
                    pass
            
            # Draw wrist trajectory if available
            if frame_count < len(pose_data):
                try:
                    pose_frame = pose_data[frame_count]
                    if pose_frame and 'normalized_pose' in pose_frame:
                        wrist = pose_frame['normalized_pose'].get('right_wrist', {})
                        if wrist and 'x' in wrist and 'y' in wrist:
                            # Convert normalized coordinates to pixel coordinates
                            wrist_x = int(wrist['x'] * width)
                            wrist_y = int(wrist['y'] * height)
                            
                            # Draw wrist position
                            cv2.circle(frame, (wrist_x, wrist_y), 8, self.colors['wrist_trajectory'], -1)
                            
                            # Draw trajectory line
                            if frame_count > 0 and frame_count - 1 < len(pose_data):
                                prev_pose = pose_data[frame_count - 1]
                                if prev_pose and 'normalized_pose' in prev_pose:
                                    prev_wrist = prev_pose['normalized_pose'].get('right_wrist', {})
                                    if prev_wrist and 'x' in prev_wrist and 'y' in prev_wrist:
                                        prev_x = int(prev_wrist['x'] * width)
                                        prev_y = int(prev_wrist['y'] * height)
                                        cv2.line(frame, (prev_x, prev_y), (wrist_x, wrist_y), 
                                               self.colors['wrist_trajectory'], 2)
                except (KeyError, IndexError):
                    pass
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
            
            # Write frame
            out.write(frame)
            frame_count += 1
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"✅ Setpoint visualization saved: {output_path}")
        return output_path


class SetpointDetectionPipeline:
    """Interactive pipeline for setpoint detection and visualization."""
    
    def __init__(self):
        # Get the current working directory and construct paths relative to it
        current_dir = os.getcwd()
        self.video_dir = os.path.join(current_dir, "data", "video")
        self.results_dir = os.path.join(current_dir, "data", "results")
        self.output_dir = os.path.join(current_dir, "data", "visualized_video", "setup")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.detector = SetpointDetector()
        self.visualizer = SetpointVisualizer()
        
        # Data storage
        self.video_path = None
        self.video_data = None
        self.pose_data = None
        self.ball_data = None
        self.setpoints = None
    
    def select_video(self) -> Optional[str]:
        """
        Select video processing mode via console.
        
        Returns:
            Selected video path, "ALL_VIDEOS" for batch processing, or None if cancelled
        """
        print("\n🎯 Setpoint Detection - Video Selection")
        print("=" * 50)
        print("1. 📁 모든 영상 처리 (Standard + Edge)")
        print("2. 🎬 특정 영상 선택")
        print("3. ❌ 취소")
        print("-" * 50)
        
        while True:
            try:
                choice = input("선택하세요 (1-3): ").strip()
                
                if choice == "1":
                    return "ALL_VIDEOS"
                elif choice == "2":
                    return self._select_specific_video()
                elif choice == "3":
                    print("❌ 취소되었습니다.")
                    return None
                else:
                    print("❌ 잘못된 선택입니다. 1, 2, 또는 3을 입력하세요.")
            except KeyboardInterrupt:
                print("\n❌ 취소되었습니다.")
                return None
    
    def _select_specific_video(self) -> Optional[str]:
        """Select a specific video file using file dialog."""
        import tkinter as tk
        from tkinter import filedialog
        
        # Hide main tkinter window
        root = tk.Tk()
        root.withdraw()
        
        try:
            video_path = filedialog.askopenfilename(
                title="Select Video File for Setpoint Detection",
                initialdir=self.video_dir,
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ]
            )
            
            if video_path:
                print(f"✅ Video selected: {os.path.basename(video_path)}")
                return video_path
            else:
                print("❌ No video selected.")
                return None
                
        except Exception as e:
            print(f"❌ Error selecting video: {e}")
            return None
        finally:
            root.destroy()
    
    def get_all_video_files(self) -> List[str]:
        """Get all video files from Standard and Edge directories."""
        video_files = []
        
        # Standard directory
        standard_dir = os.path.join(self.video_dir, "Standard")
        if os.path.exists(standard_dir):
            for file in os.listdir(standard_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join("Standard", file))
        
        # Edge directory
        edge_dir = os.path.join(self.video_dir, "EdgeCase")
        if os.path.exists(edge_dir):
            for file in os.listdir(edge_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join("EdgeCase", file))
        
        return video_files
    
    def process_all_videos(self) -> bool:
        """Process all videos in Standard and Edge directories."""
        video_files = self.get_all_video_files()
        
        if not video_files:
            print("❌ No video files found in Standard or Edge directories.")
            return False
        
        print(f"📁 Found {len(video_files)} video files to process:")
        for i, video_file in enumerate(video_files, 1):
            print(f"   {i}. {video_file}")
        
        print("\n" + "="*60)
        print("🎯 Starting batch setpoint detection...")
        print("="*60)
        
        success_count = 0
        total_count = len(video_files)
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n📹 Processing {i}/{total_count}: {video_file}")
            print("-" * 50)
            
            # Set the video path for current processing
            video_path = os.path.join(self.video_dir, video_file)
            self.video_path = video_path
            
            # Check if corresponding JSON file exists
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            json_file = os.path.join(self.results_dir, f"{video_name}_normalized_output.json")
            
            if not os.path.exists(json_file):
                print(f"❌ No processed data found for {video_name}")
                print(f"   Looking for: {json_file}")
                continue
            
            # Load data
            if not self.load_data_files(video_file):
                print(f"❌ Failed to load data for {video_file}")
                continue
            
            # Detect setpoints
            if not self.detect_setpoints():
                print(f"❌ Failed to detect setpoints for {video_file}")
                continue
            
            # Create visualization
            if not self.create_visualization():
                print(f"❌ Failed to create visualization for {video_file}")
                continue
            
            success_count += 1
            print(f"✅ Successfully processed {video_file}")
        
        print("\n" + "="*60)
        print(f"🎯 Batch processing completed!")
        print(f"✅ Successfully processed: {success_count}/{total_count} videos")
        print("="*60)
        
        return success_count > 0
    
    def load_data_files(self, video_path: str) -> bool:
        """
        Load processed data files for the selected video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if successful, False otherwise
        """
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        result_file = os.path.join(self.results_dir, f"{base_name}_normalized_output.json")
        
        print(f"\n🔍 Loading data for: {os.path.basename(video_path)}")
        print(f"🔍 Looking for file: {result_file}")
        
        if not os.path.exists(result_file):
            print(f"❌ No processed data found: {os.path.basename(result_file)}")
            print("Please run the data processing pipeline first.")
            return False
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract pose and ball data
            frames = data.get('frames', [])
            self.pose_data = []
            self.ball_data = []
            
            for frame in frames:
                # Extract pose data
                pose_info = {
                    'normalized_pose': frame.get('normalized_pose', {}),
                    'phase': frame.get('phase', 'General')  # Add phase information
                }
                self.pose_data.append(pose_info)
                
                # Extract ball data
                ball_info = {
                    'center_x': frame.get('normalized_ball', {}).get('center_x', 0.0),
                    'center_y': frame.get('normalized_ball', {}).get('center_y', 0.0)
                }
                self.ball_data.append(ball_info)
            
            print(f"✅ Loaded {len(self.pose_data)} frames")
            print(f"📊 Pose data: {len(self.pose_data)} frames")
            print(f"🏀 Ball data: {len(self.ball_data)} frames")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def detect_setpoints(self) -> bool:
        """
        Detect setpoints in the loaded data.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.pose_data or not self.ball_data:
            print("❌ No data loaded. Please load video data first.")
            return False
        
        print("\n🔍 Detecting setpoints...")
        
        try:
            self.setpoints = self.detector.detect_setpoint(self.pose_data, self.ball_data)
            
            print(f"✅ Detected {len(self.setpoints)} setpoints:")
            for i, frame_idx in enumerate(self.setpoints):
                print(f"   Setpoint {i+1}: Frame {frame_idx}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error detecting setpoints: {e}")
            return False
    
    def create_visualization(self) -> bool:
        """
        Create visualization of setpoint detection results.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.setpoints:
            print("❌ No setpoints detected. Please run detection first.")
            return False
        
        if not self.video_path:
            print("❌ No video path set.")
            return False
        
        print("\n🎬 Creating setpoint visualization...")
        
        try:
            # Generate output path
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}_setpoint_visualization.mp4")
            
            # Create visualization
            output_video = self.visualizer.visualize_setpoints(
                self.video_path, self.pose_data, self.ball_data, 
                self.setpoints, output_path
            )
            
            print(f"✅ Visualization created: {output_video}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            return False
    
    def run_pipeline(self) -> bool:
        """
        Run the complete setpoint detection pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        print("🎯 Setpoint Detection Pipeline")
        print("=" * 50)
        
        # Step 1: Select video or processing mode
        video_selection = self.select_video()
        
        if video_selection is None:
            print("❌ No selection made. Exiting.")
            return False
        
        if video_selection == "ALL_VIDEOS":
            # Process all videos
            return self.process_all_videos()
        else:
            # Process single video
            self.video_path = video_selection
            
            # Step 2: Load data
            if not self.load_data_files(video_selection):
                return False
            
            # Step 3: Detect setpoints
            if not self.detect_setpoints():
                return False
            
            # Step 4: Create visualization
            if not self.create_visualization():
                return False
            
            print("\n✅ Setpoint detection pipeline completed successfully!")
            return True


def main():
    """Main function to run the setpoint detection pipeline."""
    try:
        pipeline = SetpointDetectionPipeline()
        success = pipeline.run_pipeline()
        
        if success:
            print("\n✅ Setpoint detection completed successfully!")
        else:
            print("\n❌ Setpoint detection failed!")
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main() 