# -*- coding: utf-8 -*-
"""
농구공 인식 모델 레이어
YOLOv8을 사용해서 비디오에서 농구공을 감지하고 추적
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import torch

class BallDetectionLayer:
    def __init__(self, model_path: str = "ball_extraction/yolov8n736-customContinue.pt"):
        """
        농구공 인식 모델 초기화
        
        Args:
            model_path: YOLOv8 모델 파일 경로
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """YOLOv8 모델 로드"""
        try:
            self.model = YOLO(self.model_path)
            print(f"YOLOv8 모델 로드 완료: {self.model_path}")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            # 기본 YOLOv8 모델로 대체
            self.model = YOLO("yolov8n.pt")
            print("기본 YOLOv8 모델로 대체")

    def detect_ball_in_frame(self, frame: np.ndarray, conf_threshold: float = 0.15, 
                           classes: List[int] = [0, 1, 2], iou_threshold: float = 0.1) -> List[Dict]:
        """
        단일 프레임에서 농구공 감지
        
        Args:
            frame: 입력 프레임
            conf_threshold: 신뢰도 임계값
            classes: 감지할 클래스 (0: 농구공, 1: 선수, 2: 기타)
            iou_threshold: IoU 임계값
            
        Returns:
            감지된 공들의 정보 리스트
        """
        results = self.model(frame, conf=conf_threshold, classes=classes, 
                           iou=iou_threshold, imgsz=736, verbose=False)
        
        ball_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 박스 좌표 추출
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # 농구공 클래스 (0)인 경우만 처리
                    if class_id == 0:
                        ball_info = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'center_x': float((x1 + x2) / 2),
                            'center_y': float((y1 + y2) / 2),
                            'width': float(x2 - x1),
                            'height': float(y2 - y1)
                        }
                        ball_detections.append(ball_info)
        
        return ball_detections

    def extract_ball_trajectory_from_video(self, video_path: str, conf_threshold: float = 0.15,
                                         classes: List[int] = [0, 1, 2], iou_threshold: float = 0.1) -> List[Dict]:
        """
        비디오에서 농구공 궤적 추출
        
        Args:
            video_path: 비디오 파일 경로
            conf_threshold: 신뢰도 임계값
            classes: 감지할 클래스
            iou_threshold: IoU 임계값
            
        Returns:
            프레임별 공 감지 정보 리스트
        """
        if not cv2.VideoCapture(video_path).isOpened():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"농구공 궤적 추출 시작: {total_frames}프레임, {fps}fps")
        
        ball_trajectory = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            print(f"공 감지 처리: {frame_count}/{total_frames}", end="\r")
            
            # 현재 프레임에서 공 감지
            ball_detections = self.detect_ball_in_frame(
                frame, conf_threshold, classes, iou_threshold
            )
            
            frame_data = {
                "frame_number": frame_count,
                "timestamp": frame_count / fps,
                "ball_detections": ball_detections,
                "ball_count": len(ball_detections)
            }
            ball_trajectory.append(frame_data)
        
        cap.release()
        print(f"\n농구공 궤적 추출 완료: {len(ball_trajectory)} 프레임")
        
        return ball_trajectory

    def filter_ball_detections(self, ball_trajectory: List[Dict], 
                             min_confidence: float = 0.3, 
                             min_ball_size: float = 10.0) -> List[Dict]:
        """
        공 감지 결과 필터링
        
        Args:
            ball_trajectory: 공 궤적 데이터
            min_confidence: 최소 신뢰도
            min_ball_size: 최소 공 크기 (픽셀)
            
        Returns:
            필터링된 공 궤적 데이터
        """
        filtered_trajectory = []
        
        for frame_data in ball_trajectory:
            filtered_detections = []
            
            for detection in frame_data['ball_detections']:
                if (detection['confidence'] >= min_confidence and 
                    detection['width'] >= min_ball_size and 
                    detection['height'] >= min_ball_size):
                    filtered_detections.append(detection)
            
            filtered_frame = {
                "frame_number": frame_data['frame_number'],
                "timestamp": frame_data['timestamp'],
                "ball_detections": filtered_detections,
                "ball_count": len(filtered_detections)
            }
            filtered_trajectory.append(filtered_frame)
        
        print(f"공 감지 필터링: {len(ball_trajectory)} -> {len(filtered_trajectory)} 프레임")
        return filtered_trajectory

    def get_ball_statistics(self, ball_trajectory: List[Dict]) -> Dict:
        """공 감지 통계 정보 반환"""
        total_frames = len(ball_trajectory)
        frames_with_ball = sum(1 for frame in ball_trajectory if frame['ball_count'] > 0)
        total_balls_detected = sum(frame['ball_count'] for frame in ball_trajectory)
        
        # 신뢰도 통계
        confidences = []
        for frame in ball_trajectory:
            for detection in frame['ball_detections']:
                confidences.append(detection['confidence'])
        
        stats = {
            "total_frames": total_frames,
            "frames_with_ball": frames_with_ball,
            "total_balls_detected": total_balls_detected,
            "detection_rate": frames_with_ball / total_frames if total_frames > 0 else 0,
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "min_confidence": np.min(confidences) if confidences else 0,
            "max_confidence": np.max(confidences) if confidences else 0
        }
        
        return stats 