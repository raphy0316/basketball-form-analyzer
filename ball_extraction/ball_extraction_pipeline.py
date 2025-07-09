# -*- coding: utf-8 -*-
"""
Basketball Ball Extraction Integrated Pipeline
Extracts only original absolute coordinates and saves as JSON
"""

import os
import sys
import cv2
from typing import Dict, List, Optional

# Import layer modules
from .ball_detection_layer import BallDetectionLayer
from .ball_storage_layer import BallStorageLayer

class BallExtractionPipeline:
    def __init__(self, model_path: str = "ball_extraction/yolov8n736-customContinue.pt", output_dir: str = "data"):
        """Initialize ball extraction pipeline"""
        self.detection_layer = BallDetectionLayer(model_path)
        self.storage_layer = BallStorageLayer(output_dir)
        
        print("Ball extraction pipeline initialized")
        print("=" * 50)

    def extract_ball_trajectory(self, video_path: str, conf_threshold: float = 0.15,
                               classes: List[int] = [0, 1, 2], iou_threshold: float = 0.1,
                               min_confidence: float = 0.3, min_ball_size: float = 10.0) -> str:
        """
        Run ball extraction pipeline for original absolute coordinates
        
        Args:
            video_path: Path to video file
            conf_threshold: YOLO confidence threshold
            classes: Classes to detect
            iou_threshold: IoU threshold
            min_confidence: Minimum confidence (for filtering)
            min_ball_size: Minimum ball size (pixels)
        
        Returns:
            Path to saved file
        """
        print(f"🏀 Video file: {video_path}")
        print(f"🎯 Confidence threshold: {conf_threshold}")
        print(f"📊 Filtering threshold: {min_confidence}")
        print("-" * 50)
        
        try:
            # Step 1: Detection layer - extract original ball trajectory
            print("🔍 Step 1: Extracting original basketball trajectory...")
            raw_ball_trajectory, rim_info = self.detection_layer.extract_ball_trajectory_and_rim_info_from_video(
                video_path, conf_threshold, classes, iou_threshold
            )
            print(f"✅ Extraction complete: {len(raw_ball_trajectory)} frames")
            
            # Step 2: Confidence and size filtering
            print("\n🔄 Step 2: Filtering by confidence...")
            filtered_trajectory = self.detection_layer.filter_ball_detections(
                raw_ball_trajectory, min_confidence, min_ball_size
            )
            print(f"✅ Filtering complete: {len(filtered_trajectory)} frames")
            
            # Step 3: Save original absolute coordinates as JSON
            print("\n💾 Step 3: Saving original data...")
            base_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_ball_original"
            saved_file = self.storage_layer.save_original_as_json(filtered_trajectory, f"{base_filename}.json")
            
            print("✅ Save complete")
            print("=" * 50)
            
            # Print summary
            self._print_summary(filtered_trajectory, saved_file)
            
            return saved_file
            
        except Exception as e:
            print(f"❌ Error occurred: {e}")
            raise

    def _print_summary(self, ball_trajectory: List[Dict], saved_file: str):
        """Print extraction summary"""
        # Statistics
        stats = self.detection_layer.get_ball_statistics(ball_trajectory)
        
        print("\n📋 Ball trajectory extraction summary:")
        print(f"   • Total frames: {stats['total_frames']}")
        print(f"   • Frames with ball: {stats['frames_with_ball']}")
        print(f"   • Detection rate: {stats['detection_rate']:.2%}")
        print(f"   • Total balls detected: {stats['total_balls_detected']}")
        print(f"   • Average confidence: {stats['avg_confidence']:.3f}")
        print(f"   • Saved file: {os.path.basename(saved_file)}")
        print(f"   • Coordinate system: Original absolute coordinates (pixel units)")

    def get_pipeline_info(self) -> Dict:
        """Return pipeline info"""
        storage_info = self.storage_layer.get_storage_info()
        
        return {
            "model_info": {
                "model_path": self.detection_layer.model_path
            },
            "storage_info": storage_info
        }

def main():
    """Main execution function"""
    print("🏀 Basketball Ball Trajectory Extraction Pipeline (Original Absolute Coordinates)")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = BallExtractionPipeline()
    
    # Set video file path
    video_path = "../data/video/curry_freethrow1.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    try:
        # Run ball trajectory extraction
        saved_file = pipeline.extract_ball_trajectory(
            video_path=video_path,
            conf_threshold=0.15,
            classes=[0, 1, 2],
            iou_threshold=0.1,
            min_confidence=0.3,
            min_ball_size=10.0
        )
        
        print("\n🎉 Ball trajectory extraction pipeline complete!")
        
        # Print pipeline info
        info = pipeline.get_pipeline_info()
        print(f"\n📊 Pipeline info:")
        print(f"   • Model: {info['model_info']['model_path']}")
        print(f"   • Storage: {info['storage_info']['output_dir']}")
        
    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 