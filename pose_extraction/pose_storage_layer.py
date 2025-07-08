# -*- coding: utf-8 -*-
"""
포즈 데이터 저장 레이어
정규화된 포즈 데이터를 다양한 형식으로 저장
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

def convert_numpy_types(obj):
    """numpy 타입을 Python 기본 타입으로 변환"""
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
        """원본 절대좌표 포즈 데이터를 JSON 형식으로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_original_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 원본 절대좌표만 포함하도록 데이터 구조 수정
        original_pose_data = []
        for frame_data in pose_data:
            original_frame = {
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp']
            }
            
            # 원본 절대좌표만 저장
            pose = frame_data['pose']
            original_pose = {}
            
            for kp_name, kp_data in pose.items():
                original_pose[kp_name] = {
                    'x': kp_data['x'],  # 원본 픽셀 좌표
                    'y': kp_data['y'],  # 원본 픽셀 좌표
                    'confidence': kp_data['confidence']
                }
            
            original_frame['pose'] = original_pose
            original_pose_data.append(original_frame)
        
        data_to_save = {
            "metadata": {
                "total_frames": len(original_pose_data),
                "extraction_time": datetime.now().isoformat(),
                "keypoint_count": len(original_pose_data[0]['pose']) if original_pose_data else 0,
                "coordinate_system": "original_pixel_coordinates"
            },
            "pose_data": original_pose_data
        }
        
        # numpy 타입을 Python 기본 타입으로 변환
        data_to_save = convert_numpy_types(data_to_save)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"원본 JSON 저장 완료: {filepath}")
        return filepath

    def save_as_json(self, pose_data: List[Dict], filename: Optional[str] = None) -> str:
        """포즈 데이터를 JSON 형식으로 저장 (원본 + 정규화된 좌표 모두 포함)"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_data_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 원본 픽셀 좌표와 정규화된 좌표를 모두 포함하도록 데이터 구조 수정
        enhanced_pose_data = []
        for frame_data in pose_data:
            enhanced_frame = {
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp']
            }
            
            # 원본 픽셀 좌표와 정규화된 좌표를 모두 저장
            pose = frame_data['pose']
            enhanced_pose = {}
            
            for kp_name, kp_data in pose.items():
                enhanced_pose[kp_name] = {
                    'x': kp_data['x'],  # 정규화된 좌표
                    'y': kp_data['y'],  # 정규화된 좌표
                    'confidence': kp_data['confidence'],
                    'original_x': kp_data.get('original_x', kp_data['x']),  # 원본 픽셀 좌표
                    'original_y': kp_data.get('original_y', kp_data['y'])   # 원본 픽셀 좌표
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
        
        # numpy 타입을 Python 기본 타입으로 변환
        data_to_save = convert_numpy_types(data_to_save)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"JSON 저장 완료: {filepath}")
        return filepath

    def save_as_csv(self, pose_data: List[Dict], filename: Optional[str] = None) -> str:
        """포즈 데이터를 CSV 형식으로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_data_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # CSV 형식으로 변환
        csv_data = []
        for frame_data in pose_data:
            row = {
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp']
            }
            
            # 각 키포인트를 컬럼으로 추가
            for kp_name, kp_data in frame_data['pose'].items():
                row[f'{kp_name}_x'] = kp_data['x']
                row[f'{kp_name}_y'] = kp_data['y']
                row[f'{kp_name}_confidence'] = kp_data['confidence']
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"CSV 저장 완료: {filepath}")
        return filepath

    def save_metadata(self, pose_data: List[Dict], filename: Optional[str] = None) -> str:
        """메타데이터만 별도로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metadata_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 통계 정보 계산
        total_frames = len(pose_data)
        keypoint_names = list(pose_data[0]['pose'].keys()) if pose_data else []
        
        # 신뢰도 통계
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
        
        # numpy 타입을 Python 기본 타입으로 변환
        metadata = convert_numpy_types(metadata)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"메타데이터 저장 완료: {filepath}")
        return filepath

    def save_all_formats(self, pose_data: List[Dict], base_filename: Optional[str] = None) -> Dict[str, str]:
        """모든 형식으로 저장"""
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"pose_data_{timestamp}"
        
        saved_files = {}
        
        # JSON 저장
        json_file = self.save_as_json(pose_data, f"{base_filename}.json")
        saved_files['json'] = json_file
        
        # CSV 저장
        csv_file = self.save_as_csv(pose_data, f"{base_filename}.csv")
        saved_files['csv'] = csv_file
        
        # 메타데이터 저장
        metadata_file = self.save_metadata(pose_data, f"{base_filename}_metadata.json")
        saved_files['metadata'] = metadata_file
        
        print(f"모든 형식 저장 완료: {len(saved_files)}개 파일")
        return saved_files

    def load_pose_data(self, filepath: str) -> List[Dict]:
        """저장된 포즈 데이터 로드"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "pose_data" in data:
            return data["pose_data"]
        else:
            return data  # 직접 pose_data 배열인 경우

    def get_storage_info(self) -> Dict:
        """저장소 정보 반환"""
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