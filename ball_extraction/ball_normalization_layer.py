# -*- coding: utf-8 -*-
"""
농구공 궤적 정규화 레이어
추출된 공 궤적 데이터를 분석에 적합한 형태로 정규화
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

class BallNormalizationLayer:
    def __init__(self):
        self.frame_width = 1920  # 기본 프레임 너비
        self.frame_height = 1080  # 기본 프레임 높이

    def set_frame_dimensions(self, width: int, height: int):
        """프레임 크기 설정"""
        self.frame_width = width
        self.frame_height = height

    def normalize_ball_coordinates(self, ball_trajectory: List[Dict]) -> List[Dict]:
        """공 좌표를 0-1 범위로 정규화"""
        normalized_trajectory = []
        
        for frame_data in ball_trajectory:
            normalized_detections = []
            
            for detection in frame_data['ball_detections']:
                # 좌표 정규화 (0-1 범위)
                norm_center_x = detection['center_x'] / self.frame_width
                norm_center_y = detection['center_y'] / self.frame_height
                norm_width = detection['width'] / self.frame_width
                norm_height = detection['height'] / self.frame_height
                
                # bbox 정규화
                bbox = detection['bbox']
                norm_bbox = [
                    bbox[0] / self.frame_width,   # x1
                    bbox[1] / self.frame_height,  # y1
                    bbox[2] / self.frame_width,   # x2
                    bbox[3] / self.frame_height   # y2
                ]
                
                normalized_detection = {
                    'bbox': norm_bbox,
                    'confidence': detection['confidence'],
                    'class_id': detection['class_id'],
                    'center_x': norm_center_x,
                    'center_y': norm_center_y,
                    'width': norm_width,
                    'height': norm_height,
                    'original_center_x': detection['center_x'],
                    'original_center_y': detection['center_y']
                }
                normalized_detections.append(normalized_detection)
            
            normalized_frame = {
                "frame_number": frame_data['frame_number'],
                "timestamp": frame_data['timestamp'],
                "ball_detections": normalized_detections,
                "ball_count": len(normalized_detections)
            }
            normalized_trajectory.append(normalized_frame)
        
        return normalized_trajectory

    def calculate_ball_velocity(self, ball_trajectory: List[Dict]) -> List[Dict]:
        """공의 속도 계산"""
        trajectory_with_velocity = []
        
        for i, frame_data in enumerate(ball_trajectory):
            frame_with_velocity = frame_data.copy()
            
            if i > 0 and frame_data['ball_count'] > 0:
                prev_frame = ball_trajectory[i-1]
                
                for detection in frame_data['ball_detections']:
                    # 이전 프레임에서 가장 가까운 공 찾기
                    min_distance = float('inf')
                    closest_prev_detection = None
                    
                    for prev_detection in prev_frame['ball_detections']:
                        distance = np.sqrt(
                            (detection['center_x'] - prev_detection['center_x'])**2 +
                            (detection['center_y'] - prev_detection['center_y'])**2
                        )
                        if distance < min_distance:
                            min_distance = distance
                            closest_prev_detection = prev_detection
                    
                    # 속도 계산 (픽셀/프레임)
                    if closest_prev_detection and min_distance < 100:  # 최대 100픽셀 이동
                        velocity_x = detection['center_x'] - closest_prev_detection['center_x']
                        velocity_y = detection['center_y'] - closest_prev_detection['center_y']
                        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
                        
                        detection['velocity_x'] = velocity_x
                        detection['velocity_y'] = velocity_y
                        detection['velocity_magnitude'] = velocity_magnitude
                    else:
                        detection['velocity_x'] = 0
                        detection['velocity_y'] = 0
                        detection['velocity_magnitude'] = 0
                else:
                    # 첫 프레임이거나 공이 없는 경우
                    for detection in frame_data['ball_detections']:
                        detection['velocity_x'] = 0
                        detection['velocity_y'] = 0
                        detection['velocity_magnitude'] = 0
            
            trajectory_with_velocity.append(frame_with_velocity)
        
        return trajectory_with_velocity

    def smooth_ball_trajectory(self, ball_trajectory: List[Dict], window_size: int = 5) -> List[Dict]:
        """공 궤적 스무딩 (이동 평균)"""
        if len(ball_trajectory) < window_size:
            return ball_trajectory
        
        smoothed_trajectory = []
        
        for i in range(len(ball_trajectory)):
            frame_data = ball_trajectory[i].copy()
            
            # 윈도우 범위 계산
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(ball_trajectory), i + window_size // 2 + 1)
            
            smoothed_detections = []
            
            for detection in frame_data['ball_detections']:
                # 주변 프레임들의 같은 위치 공들의 평균 계산
                center_xs = []
                center_ys = []
                confidences = []
                
                for j in range(start_idx, end_idx):
                    for prev_detection in ball_trajectory[j]['ball_detections']:
                        # 같은 위치의 공으로 간주 (거리 임계값)
                        distance = np.sqrt(
                            (detection['center_x'] - prev_detection['center_x'])**2 +
                            (detection['center_y'] - prev_detection['center_y'])**2
                        )
                        if distance < 50:  # 50픽셀 이내
                            center_xs.append(prev_detection['center_x'])
                            center_ys.append(prev_detection['center_y'])
                            confidences.append(prev_detection['confidence'])
                
                if center_xs:
                    smoothed_detection = detection.copy()
                    smoothed_detection['center_x'] = np.mean(center_xs)
                    smoothed_detection['center_y'] = np.mean(center_ys)
                    smoothed_detection['confidence'] = np.mean(confidences)
                    smoothed_detections.append(smoothed_detection)
                else:
                    smoothed_detections.append(detection)
            
            frame_data['ball_detections'] = smoothed_detections
            frame_data['ball_count'] = len(smoothed_detections)
            smoothed_trajectory.append(frame_data)
        
        return smoothed_trajectory

    def extract_ball_events(self, ball_trajectory: List[Dict]) -> Dict:
        """공 이벤트 추출 (슛, 패스, 드리블 등)"""
        events = {
            'shots': [],
            'passes': [],
            'dribbles': [],
            'rebounds': []
        }
        
        # 속도 기반 이벤트 분류
        for i, frame_data in enumerate(ball_trajectory):
            for detection in frame_data['ball_detections']:
                velocity = detection.get('velocity_magnitude', 0)
                
                # 슛 감지 (높은 속도 + 위쪽 방향)
                if velocity > 20 and detection.get('velocity_y', 0) < -10:
                    events['shots'].append({
                        'frame': frame_data['frame_number'],
                        'timestamp': frame_data['timestamp'],
                        'position': (detection['center_x'], detection['center_y']),
                        'velocity': velocity
                    })
                
                # 패스 감지 (중간 속도)
                elif 10 < velocity < 30:
                    events['passes'].append({
                        'frame': frame_data['frame_number'],
                        'timestamp': frame_data['timestamp'],
                        'position': (detection['center_x'], detection['center_y']),
                        'velocity': velocity
                    })
                
                # 드리블 감지 (낮은 속도 + 반복)
                elif velocity < 10:
                    events['dribbles'].append({
                        'frame': frame_data['frame_number'],
                        'timestamp': frame_data['timestamp'],
                        'position': (detection['center_x'], detection['center_y']),
                        'velocity': velocity
                    })
        
        return events

    def validate_ball_data(self, ball_trajectory: List[Dict]) -> bool:
        """공 데이터 유효성 검증"""
        if not ball_trajectory:
            return False
        
        # 최소한 몇 프레임에서 공이 감지되어야 함
        frames_with_ball = sum(1 for frame in ball_trajectory if frame['ball_count'] > 0)
        min_detection_rate = 0.1  # 최소 10% 프레임에서 공 감지
        
        if frames_with_ball / len(ball_trajectory) < min_detection_rate:
            return False
        
        return True 