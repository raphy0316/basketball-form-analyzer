# -*- coding: utf-8 -*-
"""
Pose Extraction Integrated Pipeline
Extracts only original absolute coordinates and saves as JSON
"""

import os
import sys
from typing import Dict, List, Optional

# Import layer modules
from .pose_model_layer import PoseModelLayer
from .pose_storage_layer import PoseStorageLayer

class PoseExtractionPipeline:
    def __init__(self, output_dir: str = "data", load_model: bool = True):
        """Initialize pose extraction pipeline"""
        self.model_layer = PoseModelLayer(load_model=load_model)
        self.storage_layer = PoseStorageLayer(output_dir)
        
        print("Pose extraction pipeline initialized")
        print("=" * 50)

    def extract_poses(self, video_path: str, confidence_threshold: float = 0.3) -> str:
        """
        Run pose extraction pipeline for original absolute coordinates
        
        Args:
            video_path: Path to video file
            confidence_threshold: Confidence threshold
        
        Returns:
            Path to saved file
        """
        print(f"🎬 Video file: {video_path}")
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        print("-" * 50)
        
        try:
            # Step 1: Model layer - extract original pose data
            print("🔍 Step 1: Extracting original pose data...")
            raw_pose_data = self.model_layer.extract_poses_from_video(video_path)
            print(f"✅ Extraction complete: {len(raw_pose_data)} frames")
            
            # Step 2: Confidence filtering (remove low-confidence keypoints)
            print("\n🔄 Step 2: Filtering by confidence...")
            filtered_data = self._filter_low_confidence_poses(raw_pose_data, confidence_threshold)
            print(f"✅ Filtering complete: {len(filtered_data)} frames")
            
            # Step 3: Save original absolute coordinates as JSON
            print("\n💾 Step 3: Saving original data...")
            base_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_pose_original"
            saved_file = self.storage_layer.save_original_as_json(filtered_data, f"{base_filename}.json")
            
            print("✅ Save complete")
            print("=" * 50)
            
            # Print summary
            self._print_summary(filtered_data, saved_file)
            
            return saved_file
            
        except Exception as e:
            print(f"❌ Error occurred: {e}")
            raise
    
    def filter_low_confidence_poses_api_helper(self, pose_data: List[Dict], confidence_threshold: float = 0.3) -> List[Dict]:
        """
        API helper to filter low-confidence poses
        
        Args:
            pose_data: List of pose data dictionaries
            confidence_threshold: Minimum confidence threshold
        
        Returns:
            Filtered pose data
        """
        return self._filter_low_confidence_poses(pose_data, confidence_threshold)
    
    def _filter_low_confidence_poses(self, pose_data: List[Dict], confidence_threshold: float) -> List[Dict]:
        """Filter out low-confidence keypoints"""
        filtered_data = []
        
        for frame_data in pose_data:
            filtered_frame = {
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp'],
                'pose': {}
            }
            
            for kp_name, kp_data in frame_data['pose'].items():
                if kp_data['confidence'] >= confidence_threshold:
                    # Save only original absolute coordinates
                    filtered_frame['pose'][kp_name] = {
                        'x': kp_data['x'],  # Original pixel coordinate
                        'y': kp_data['y'],  # Original pixel coordinate
                        'confidence': kp_data['confidence']
                    }
            
            filtered_data.append(filtered_frame)
        
        return filtered_data

    def print_summary_api_helper(self, pose_data: List[Dict], saved_file: str):
        """
        API helper to print extraction summary
        
        Args:
            pose_data: List of pose data dictionaries
            saved_file: Path to saved file
        """
        self._print_summary(pose_data, saved_file)

    def _print_summary(self, pose_data: List[Dict], saved_file: str):
        """Print extraction summary"""
        print("\n📋 Extraction summary:")
        print(f"   • Total frames: {len(pose_data)}")
        print(f"   • Keypoints per frame: {len(pose_data[0]['pose']) if pose_data else 0}")
        print(f"   • Saved file: {os.path.basename(saved_file)}")
        print(f"   • Coordinate system: Original absolute coordinates (pixel units)")

    def get_pipeline_info(self) -> Dict:
        """Return pipeline info"""
        storage_info = self.storage_layer.get_storage_info()
        
        return {
            "model_info": {
                "model_name": self.model_layer.model_name,
                "keypoint_count": len(self.model_layer.keypoint_names)
            },
            "storage_info": storage_info
        }

def main():
    """Main execution function"""
    print("🏀 Basketball Pose Extraction Pipeline (Original Absolute Coordinates)")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = PoseExtractionPipeline()
    
    # Set video file path
    video_path = "../References/stephen_curry_multy_person_part.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    try:
        # Run pose extraction
        saved_file = pipeline.extract_poses(
            video_path=video_path,
            confidence_threshold=0.3
        )
        
        print("\n🎉 Pose extraction pipeline complete!")
        
        # Print pipeline info
        info = pipeline.get_pipeline_info()
        print(f"\n📊 Pipeline info:")
        print(f"   • Model: {info['model_info']['model_name']}")
        print(f"   • Keypoint count: {info['model_info']['keypoint_count']}")
        print(f"   • Storage: {info['storage_info']['output_dir']}")
        
    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 