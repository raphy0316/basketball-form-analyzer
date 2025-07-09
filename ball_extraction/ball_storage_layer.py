# -*- coding: utf-8 -*-
"""
Ball Trajectory Storage Layer
Saves normalized ball trajectory data in various formats
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

class BallStorageLayer:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_original_as_json(self, ball_trajectory: List[Dict], filename: Optional[str] = None) -> str:
        """Save original absolute coordinate ball trajectory data as JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ball_original_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        data_to_save = {
            "metadata": {
                "total_frames": len(ball_trajectory),
                "extraction_time": datetime.now().isoformat(),
                "frames_with_ball": sum(1 for frame in ball_trajectory if frame['ball_count'] > 0),
                "total_balls_detected": sum(frame['ball_count'] for frame in ball_trajectory),
                "coordinate_system": "original_pixel_coordinates"
            },
            "ball_trajectory": ball_trajectory
        }
        
        # Convert numpy types to Python basic types
        data_to_save = convert_numpy_types(data_to_save)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"Original ball trajectory JSON save complete: {filepath}")
        return filepath
    
    def save_rim_original_as_json(self, rim_info: List[Dict], filename: Optional[str] = None) -> str:
        """Save original absolute coordinate rim info data as JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rim_original_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        data_to_save = {
            "metadata": {
                "total_frames": len(rim_info),
                "extraction_time": datetime.now().isoformat(),
                "frames_with_rim": sum(1 for frame in rim_info if frame['rim_count'] > 0),
                "total_rims_detected": sum(frame['rim_count'] for frame in rim_info),
                "coordinate_system": "original_pixel_coordinates"
            },
            "rim_info": rim_info
        }
        
        # Convert numpy types to Python basic types
        data_to_save = convert_numpy_types(data_to_save)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"Original rim info JSON save complete: {filepath}")
        return filepath

    def save_as_json(self, ball_trajectory: List[Dict], filename: Optional[str] = None) -> str:
        """Save ball trajectory data as JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ball_trajectory_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        data_to_save = {
            "metadata": {
                "total_frames": len(ball_trajectory),
                "extraction_time": datetime.now().isoformat(),
                "frames_with_ball": sum(1 for frame in ball_trajectory if frame['ball_count'] > 0),
                "total_balls_detected": sum(frame['ball_count'] for frame in ball_trajectory)
            },
            "ball_trajectory": ball_trajectory
        }
        
        # Convert numpy types to Python basic types
        data_to_save = convert_numpy_types(data_to_save)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"Ball trajectory JSON save complete: {filepath}")
        return filepath

    def save_as_csv(self, ball_trajectory: List[Dict], filename: Optional[str] = None) -> str:
        """Save ball trajectory data as CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ball_trajectory_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to CSV format
        csv_data = []
        for frame_data in ball_trajectory:
            if frame_data['ball_count'] > 0:
                for ball_idx, ball in enumerate(frame_data['ball_detections']):
                    row = {
                        'frame_number': frame_data['frame_number'],
                        'timestamp': frame_data['timestamp'],
                        'ball_id': ball_idx,
                        'center_x': ball['center_x'],
                        'center_y': ball['center_y'],
                        'width': ball['width'],
                        'height': ball['height'],
                        'confidence': ball['confidence'],
                        'class_id': ball['class_id']
                    }
                    
                    # Add velocity info if present
                    if 'velocity_x' in ball:
                        row['velocity_x'] = ball['velocity_x']
                        row['velocity_y'] = ball['velocity_y']
                        row['velocity_magnitude'] = ball['velocity_magnitude']
                    
                    csv_data.append(row)
            else:
                # Frame with no detected ball
                row = {
                    'frame_number': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                    'ball_id': -1,
                    'center_x': np.nan,
                    'center_y': np.nan,
                    'width': np.nan,
                    'height': np.nan,
                    'confidence': np.nan,
                    'class_id': np.nan
                }
                csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"Ball trajectory CSV save complete: {filepath}")
        return filepath

    def save_ball_events(self, ball_events: Dict, filename: Optional[str] = None) -> str:
        """Save ball event data"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ball_events_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        events_data = {
            "metadata": {
                "extraction_time": datetime.now().isoformat(),
                "total_shots": len(ball_events['shots']),
                "total_passes": len(ball_events['passes']),
                "total_dribbles": len(ball_events['dribbles']),
                "total_rebounds": len(ball_events['rebounds'])
            },
            "events": ball_events
        }
        
        # Convert numpy types to Python basic types
        events_data = convert_numpy_types(events_data)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(events_data, f, indent=2, ensure_ascii=False)
        
        print(f"Ball event data save complete: {filepath}")
        return filepath

    def save_trajectory_analysis(self, ball_trajectory: List[Dict], filename: Optional[str] = None) -> str:
        """Save ball trajectory analysis results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_analysis_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Trajectory analysis
        frames_with_ball = [frame for frame in ball_trajectory if frame['ball_count'] > 0]
        
        if frames_with_ball:
            # Ball position statistics
            center_xs = []
            center_ys = []
            velocities = []
            confidences = []
            
            for frame in frames_with_ball:
                for ball in frame['ball_detections']:
                    center_xs.append(ball['center_x'])
                    center_ys.append(ball['center_y'])
                    confidences.append(ball['confidence'])
                    
                    if 'velocity_magnitude' in ball:
                        velocities.append(ball['velocity_magnitude'])
            
            analysis = {
                "metadata": {
                    "extraction_time": datetime.now().isoformat(),
                    "total_frames": len(ball_trajectory),
                    "frames_with_ball": len(frames_with_ball),
                    "detection_rate": len(frames_with_ball) / len(ball_trajectory)
                },
                "position_statistics": {
                    "center_x_mean": np.mean(center_xs),
                    "center_x_std": np.std(center_xs),
                    "center_x_min": np.min(center_xs),
                    "center_x_max": np.max(center_xs),
                    "center_y_mean": np.mean(center_ys),
                    "center_y_std": np.std(center_ys),
                    "center_y_min": np.min(center_ys),
                    "center_y_max": np.max(center_ys)
                },
                "velocity_statistics": {
                    "velocity_mean": np.mean(velocities) if velocities else 0,
                    "velocity_std": np.std(velocities) if velocities else 0,
                    "velocity_max": np.max(velocities) if velocities else 0
                },
                "confidence_statistics": {
                    "confidence_mean": np.mean(confidences),
                    "confidence_std": np.std(confidences),
                    "confidence_min": np.min(confidences),
                    "confidence_max": np.max(confidences)
                }
            }
        else:
            analysis = {
                "metadata": {
                    "extraction_time": datetime.now().isoformat(),
                    "total_frames": len(ball_trajectory),
                    "frames_with_ball": 0,
                    "detection_rate": 0
                },
                "error": "No ball detected."
            }
        
        # Convert numpy types to Python basic types
        analysis = convert_numpy_types(analysis)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"Trajectory analysis save complete: {filepath}")
        return filepath

    def save_all_formats(self, ball_trajectory: List[Dict], ball_events: Optional[Dict] = None,
                        base_filename: Optional[str] = None) -> Dict[str, str]:
        """Save in all formats"""
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"ball_trajectory_{timestamp}"
        
        saved_files = {}
        
        # JSON save
        json_file = self.save_as_json(ball_trajectory, f"{base_filename}.json")
        saved_files['json'] = json_file
        
        # CSV save
        csv_file = self.save_as_csv(ball_trajectory, f"{base_filename}.csv")
        saved_files['csv'] = csv_file
        
        # Metadata save
        metadata_file = self.save_metadata(ball_trajectory, f"{base_filename}_metadata.json")
        saved_files['metadata'] = metadata_file
        
        print(f"All formats saved: {len(saved_files)} files")
        return saved_files

    def load_ball_trajectory(self, filepath: str) -> List[Dict]:
        """Load saved ball trajectory data"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "ball_trajectory" in data:
            return data["ball_trajectory"]
        else:
            return data  # If directly an array
    
    def load_rim_info(self, filepath: str) -> List[Dict]:
        """Load saved ball trajectory data"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "rim_info" in data:
            return data["rim_info"]
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