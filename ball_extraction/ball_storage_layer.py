# -*- coding: utf-8 -*-
"""
농구공 궤적 저장 레이어
정규화된 공 궤적 데이터를 다양한 형식으로 저장
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

class BallStorageLayer:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_original_as_json(self, ball_trajectory: List[Dict], filename: Optional[str] = None) -> str:
        """원본 절대좌표 공 궤적 데이터를 JSON 형식으로 저장"""
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
        
        # numpy 타입을 Python 기본 타입으로 변환
        data_to_save = convert_numpy_types(data_to_save)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"원본 공 궤적 JSON 저장 완료: {filepath}")
        return filepath

    def save_as_json(self, ball_trajectory: List[Dict], filename: Optional[str] = None) -> str:
        """공 궤적 데이터를 JSON 형식으로 저장"""
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
        
        # numpy 타입을 Python 기본 타입으로 변환
        data_to_save = convert_numpy_types(data_to_save)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"공 궤적 JSON 저장 완료: {filepath}")
        return filepath

    def save_as_csv(self, ball_trajectory: List[Dict], filename: Optional[str] = None) -> str:
        """공 궤적 데이터를 CSV 형식으로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ball_trajectory_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # CSV 형식으로 변환
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
                    
                    # 속도 정보가 있으면 추가
                    if 'velocity_x' in ball:
                        row['velocity_x'] = ball['velocity_x']
                        row['velocity_y'] = ball['velocity_y']
                        row['velocity_magnitude'] = ball['velocity_magnitude']
                    
                    csv_data.append(row)
            else:
                # 공이 감지되지 않은 프레임
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
        
        print(f"공 궤적 CSV 저장 완료: {filepath}")
        return filepath

    def save_ball_events(self, ball_events: Dict, filename: Optional[str] = None) -> str:
        """공 이벤트 데이터 저장"""
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
        
        # numpy 타입을 Python 기본 타입으로 변환
        events_data = convert_numpy_types(events_data)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(events_data, f, indent=2, ensure_ascii=False)
        
        print(f"공 이벤트 저장 완료: {filepath}")
        return filepath

    def save_trajectory_analysis(self, ball_trajectory: List[Dict], filename: Optional[str] = None) -> str:
        """공 궤적 분석 결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_analysis_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 궤적 분석
        frames_with_ball = [frame for frame in ball_trajectory if frame['ball_count'] > 0]
        
        if frames_with_ball:
            # 공 위치 통계
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
                "error": "공이 감지되지 않았습니다."
            }
        
        # numpy 타입을 Python 기본 타입으로 변환
        analysis = convert_numpy_types(analysis)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"궤적 분석 저장 완료: {filepath}")
        return filepath

    def save_all_formats(self, ball_trajectory: List[Dict], ball_events: Optional[Dict] = None,
                        base_filename: Optional[str] = None) -> Dict[str, str]:
        """모든 형식으로 저장"""
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"ball_trajectory_{timestamp}"
        
        saved_files = {}
        
        # JSON 저장
        json_file = self.save_as_json(ball_trajectory, f"{base_filename}.json")
        saved_files['json'] = json_file
        
        # CSV 저장
        csv_file = self.save_as_csv(ball_trajectory, f"{base_filename}.csv")
        saved_files['csv'] = csv_file
        
        # 궤적 분석 저장
        analysis_file = self.save_trajectory_analysis(ball_trajectory, f"{base_filename}_analysis.json")
        saved_files['analysis'] = analysis_file
        
        # 이벤트 저장 (있는 경우)
        if ball_events:
            events_file = self.save_ball_events(ball_events, f"{base_filename}_events.json")
            saved_files['events'] = events_file
        
        print(f"공 궤적 모든 형식 저장 완료: {len(saved_files)}개 파일")
        return saved_files

    def load_ball_trajectory(self, filepath: str) -> List[Dict]:
        """저장된 공 궤적 데이터 로드"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "ball_trajectory" in data:
            return data["ball_trajectory"]
        else:
            return data  # 직접 trajectory 배열인 경우

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