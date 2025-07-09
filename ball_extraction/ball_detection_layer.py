# -*- coding: utf-8 -*-
"""
Basketball Detection Model Layer
Detects and tracks basketballs in video using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import torch

class BallDetectionLayer:
    def __init__(self, model_path: str = "ball_extraction/models/yolov8n736-customContinue.pt"):
        """
        Initialize basketball detection model
        
        Args:
            model_path: YOLOv8 model file path
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"YOLOv8 model loaded: {self.model_path}")
        except Exception as e:
            print(f"Model load failed: {e}")
            # Fallback to default YOLOv8 model
            self.model = YOLO("yolov8n.pt")
            print("Fallback to default YOLOv8 model")

    def _detect_ball_and_rim_in_frame(self, frame: np.ndarray, conf_threshold: float = 0.15, 
                           classes: List[int] = [0, 1, 2], iou_threshold: float = 0.1) -> List[Dict]:
        """
        Detect basketball and rim in a single frame
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold
            classes: Classes to detect (0: basketball, 1: player, 2: other)
            iou_threshold: IoU threshold
            
        Returns:
            List of detected balls and rims
        """
        results = self.model(frame, conf=conf_threshold, classes=classes, 
                           iou=iou_threshold, imgsz=736, verbose=False)
        
        ball_detections = []
        rim_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Process only basketball class (0)
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
                    elif class_id == 2:
                        rim_info = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'center_x': float((x1 + x2) / 2),
                            'center_y': float((y1 + y2) / 2),
                            'width': float(x2 - x1),
                            'height': float(y2 - y1)
                        }  
                        rim_detections.append(rim_info)
        return ball_detections, rim_detections

    def extract_ball_trajectory_and_rim_info_from_video(self, video_path: str, conf_threshold: float = 0.15,
                                         classes: List[int] = [0, 1, 2], iou_threshold: float = 0.1) -> List[Dict]:
        """
        Extract basketball trajectory from video
        
        Args:
            video_path: Path to video file
            conf_threshold: Confidence threshold
            classes: Classes to detect
            iou_threshold: IoU threshold
            
        Returns:
            List of per-frame ball detection info
        """
        if not cv2.VideoCapture(video_path).isOpened():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Basketball trajectory extraction started: {total_frames} frames, {fps}fps")
        
        ball_trajectory = []
        rim_info = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            print(f"Ball detection processing: {frame_count}/{total_frames}", end="\r")
            
            # Detect ball in current frame
            ball_detections, rim_detections = self._detect_ball_and_rim_in_frame(
                frame, conf_threshold, classes, iou_threshold
            )
            
            frame_data = {
                "frame_number": frame_count,
                "timestamp": frame_count / fps,
                "ball_detections": ball_detections,
                "ball_count": len(ball_detections)
            }
            rim_data = {
                "frame_number": frame_count,
                "timestamp": frame_count / fps,
                "rim_detections": rim_detections,
                "rim_count": len(rim_detections)
            }
            ball_trajectory.append(frame_data)
            rim_info.append(rim_data)
    
        cap.release()
        print(f"\nBasketball trajectory extraction complete: {len(ball_trajectory)} frames")
        
        return ball_trajectory, rim_info

    def filter_rim_detections(self, rim_info: List[Dict],
                              min_confidence: float = 0.4) -> List[Dict]:
        """
        Filter rim detection results
        
        Args:
            rim_info: Rim detection data
            min_confidence: Minimum confidence
        Returns
        """
        filtered_rim = []
        widths = [detection['width'] for frame in rim_info for detection in frame['rim_detections']]
        median_width = np.median(widths)
        std = np.std(widths)
        for frame_data in rim_info:
            filtered_detections = []
            
            for detection in frame_data['rim_detections']:
                if (detection['confidence'] >= min_confidence and 
                    abs (detection['width'] - median_width) <= std):
                    filtered_detections.append(detection)
            
            filtered_frame = {
                "frame_number": frame_data['frame_number'],
                "timestamp": frame_data['timestamp'],
                "rim_detections": filtered_detections,
                "rim_count": len(filtered_detections)
            }
            filtered_rim.append(filtered_frame)
        
        print(f"Ball detection filtering: {len(rim_info)} -> {len(filtered_rim)} frames")
        return filtered_rim
            

    def filter_ball_detections(self, ball_trajectory: List[Dict], 
                             min_confidence: float = 0.3, 
                             min_ball_size: float = 10.0) -> List[Dict]:
        """
        Filter ball detection results
        
        Args:
            ball_trajectory: Ball trajectory data
            min_confidence: Minimum confidence
            min_ball_size: Minimum ball size (pixels)
            
        Returns:
            Filtered ball trajectory data
        """
        filtered_trajectory = []
        widths = [detection['width'] for frame in ball_trajectory for detection in frame['ball_detections']]
        median_width = np.median(widths)
        std = np.std(widths)
        for frame_data in ball_trajectory:
            filtered_detections = []
            
            for detection in frame_data['ball_detections']:
                if (detection['confidence'] >= min_confidence and 
                    detection['width']  >= min_ball_size and 
                    # abs(detection['width'] - median_width) <= std and
                    detection['height'] >= min_ball_size):
                    filtered_detections.append(detection)
            
            filtered_frame = {
                "frame_number": frame_data['frame_number'],
                "timestamp": frame_data['timestamp'],
                "ball_detections": filtered_detections,
                "ball_count": len(filtered_detections)
            }
            filtered_trajectory.append(filtered_frame)
        
        print(f"Ball detection filtering: {len(ball_trajectory)} -> {len(filtered_trajectory)} frames")
        return filtered_trajectory

    def get_ball_statistics(self, ball_trajectory: List[Dict]) -> Dict:
        """Return ball detection statistics"""
        total_frames = len(ball_trajectory)
        frames_with_ball = sum(1 for frame in ball_trajectory if frame['ball_count'] > 0)
        total_balls_detected = sum(frame['ball_count'] for frame in ball_trajectory)
        
        # Confidence statistics
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

    def get_rim_statistics(self, rim_info: List[Dict]) -> Dict:
        """Return rim detection statistics"""
        total_frames = len(rim_info)
        frames_with_rim = sum(1 for frame in rim_info if frame['rim_count'] > 0)
        total_rims_detected = sum(frame['rim_count'] for frame in rim_info)
        
        # Confidence statistics
        confidences = []
        for frame in rim_info:
            for detection in frame['rim_detections']:
                confidences.append(detection['confidence'])
            
        stats = {
                "total_frames": total_frames,
                "frames_with_rim": frames_with_rim,
                "total_rims_detected": total_rims_detected,
                "detection_rate": frames_with_rim / total_frames if total_frames > 0 else 0,
                "avg_confidence": np.mean(confidences) if confidences else 0,
                "min_confidence": np.min(confidences) if confidences else 0,
                "max_confidence": np.max(confidences) if confidences else 0
            }

        return stats
    