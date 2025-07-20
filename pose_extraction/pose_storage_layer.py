# -*- coding: utf-8 -*-
"""
Pose Data Storage Layer
Saves normalized pose data in various formats
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

def convert_numpy_types(obj):
    """Convert numpy types to Python basic types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class PoseStorageLayer:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_original_as_json(self, pose_data: List[Dict], filename: Optional[str] = None) -> str:
        """Save original absolute coordinate pose data as JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_original_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Only include original absolute coordinates
        original_pose_data = []
        for frame_data in pose_data:
            original_frame = {
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp']
            }
            
            # Save only original absolute coordinates
            pose = frame_data['pose']
            original_pose = {}
            
            for kp_name, kp_data in pose.items():
                original_pose[kp_name] = {
                    'x': kp_data['x'],  # Original pixel coordinate
                    'y': kp_data['y'],  # Original pixel coordinate
                    'confidence': kp_data['confidence']
                }
            
            original_frame['pose'] = original_pose
            original_pose_data.append(original_frame)
        
        data_to_save = {
            "metadata": {
                "total_frames": len(original_pose_data),
                "extraction_time": datetime.now().isoformat(),
                "keypoint_count": len(original_pose_data[0]['pose']) if original_pose_data else 0,
                "coordinate_system": "relative_coordinates_with_aspect_ratio_correction"
            },
            "pose_data": original_pose_data
        }
        
        # Convert numpy types to Python basic types
        data_to_save = convert_numpy_types(data_to_save)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"Original JSON save complete: {filepath}")
        return filepath

    def save_as_json(self, pose_data: List[Dict], filename: Optional[str] = None) -> str:
        """Save pose data as JSON (includes both original and normalized coordinates)"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_data_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Include both original pixel coordinates and normalized coordinates
        enhanced_pose_data = []
        for frame_data in pose_data:
            enhanced_frame = {
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp']
            }
            
            # Save both original pixel coordinates and normalized coordinates
            pose = frame_data['pose']
            enhanced_pose = {}
            
            for kp_name, kp_data in pose.items():
                enhanced_pose[kp_name] = {
                    'x': kp_data['x'],  # Normalized coordinate
                    'y': kp_data['y'],  # Normalized coordinate
                    'confidence': kp_data['confidence'],
                    'original_x': kp_data.get('original_x', kp_data['x']),  # Original pixel coordinate
                    'original_y': kp_data.get('original_y', kp_data['y'])   # Original pixel coordinate
                }
            
            enhanced_frame['pose'] = enhanced_pose
            enhanced_pose_data.append(enhanced_frame)
        
        data_to_save = {
            "metadata": {
                "total_frames": len(enhanced_pose_data),
                "extraction_time": datetime.now().isoformat(),
                "keypoint_count": len(enhanced_pose_data[0]['pose']) if enhanced_pose_data else 0,
                "coordinate_system": "normalized_with_original_pixel_coordinates"
            },
            "pose_data": enhanced_pose_data
        }
        
        # Convert numpy types to Python basic types
        data_to_save = convert_numpy_types(data_to_save)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"JSON save complete: {filepath}")
        return filepath

    def save_as_csv(self, pose_data: List[Dict], filename: Optional[str] = None) -> str:
        """Save pose data as CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_data_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to CSV format
        csv_data = []
        for frame_data in pose_data:
            row = {
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp']
            }
            
            # Add each keypoint as a column
            for kp_name, kp_data in frame_data['pose'].items():
                row[f'{kp_name}_x'] = kp_data['x']
                row[f'{kp_name}_y'] = kp_data['y']
                row[f'{kp_name}_confidence'] = kp_data['confidence']
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"CSV save complete: {filepath}")
        return filepath

    def save_metadata(self, pose_data: List[Dict], filename: Optional[str] = None) -> str:
        """Save only metadata separately"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metadata_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Calculate statistics
        total_frames = len(pose_data)
        keypoint_names = list(pose_data[0]['pose'].keys()) if pose_data else []
        
        # Confidence statistics
        confidence_stats = {}
        for kp_name in keypoint_names:
            confidences = [frame['pose'][kp_name]['confidence'] for frame in pose_data if kp_name in frame['pose']]
            if confidences:
                confidence_stats[kp_name] = {
                    'mean': sum(confidences) / len(confidences),
                    'min': min(confidences),
                    'max': max(confidences)
                }
        
        metadata = {
            "extraction_info": {
                "total_frames": total_frames,
                "keypoint_count": len(keypoint_names),
                "extraction_time": datetime.now().isoformat(),
                "duration_seconds": pose_data[-1]['timestamp'] if pose_data else 0
            },
            "keypoints": keypoint_names,
            "confidence_statistics": confidence_stats
        }
        
        # Convert numpy types to Python basic types
        metadata = convert_numpy_types(metadata)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Metadata save complete: {filepath}")
        return filepath

    def save_all_formats(self, pose_data: List[Dict], base_filename: Optional[str] = None) -> Dict[str, str]:
        """Save in all formats"""
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"pose_data_{timestamp}"
        
        saved_files = {}
        
        # JSON save
        json_file = self.save_as_json(pose_data, f"{base_filename}.json")
        saved_files['json'] = json_file
        
        # CSV save
        csv_file = self.save_as_csv(pose_data, f"{base_filename}.csv")
        saved_files['csv'] = csv_file
        
        # Metadata save
        metadata_file = self.save_metadata(pose_data, f"{base_filename}_metadata.json")
        saved_files['metadata'] = metadata_file
        
        print(f"All formats saved: {len(saved_files)} files")
        return saved_files

    def load_pose_data(self, filepath: str) -> List[Dict]:
        """Load saved pose data"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "pose_data" in data:
            return data["pose_data"]
        else:
            return data  # If directly an array

    def get_storage_info(self) -> Dict:
        """Return storage info"""
        if not os.path.exists(self.output_dir):
            return {"output_dir": self.output_dir, "exists": False, "file_count": 0}
        
        files = os.listdir(self.output_dir)
        json_files = [f for f in files if f.endswith('.json')]
        csv_files = [f for f in files if f.endswith('.csv')]
        
        return {
            "output_dir": self.output_dir,
            "exists": True,
            "total_files": len(files),
            "json_files": len(json_files),
            "csv_files": len(csv_files),
            "file_list": files
        } 