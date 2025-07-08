# -*- coding: utf-8 -*-
"""
pose recognition model layer
Use MoveNet model to extract pose keypoints from video
"""

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import Dict, List, Tuple

class PoseModelLayer:
    def __init__(self, model_name="lightning"):
        self.model_name = model_name
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load MoveNet model"""
        model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
        self.movenet = hub.load(model_url)
        self.model = self.movenet.signatures["serving_default"]
        print("MoveNet model loading completed")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (192, 192))
        input_frame = np.expand_dims(resized_frame, axis=0).astype(np.int32)
        return input_frame

    def detect_pose(self, frame: np.ndarray) -> Dict:
        """Detect pose from single frame (convert to pixel coordinates)"""
        input_frame = self.preprocess_frame(frame)
        results = self.model(input=input_frame)
        keypoints = results["output_0"].numpy()
        keypoints = keypoints[0, 0]
        
        # Original frame size
        h, w = frame.shape[:2]
        
        pose_data = {}
        for i, name in enumerate(self.keypoint_names):
            y, x, confidence = keypoints[i]
            
            # Convert normalized coordinates (0~1) to pixel coordinates
            pixel_x = int(x * w)
            pixel_y = int(y * h)
            
            pose_data[name] = {
                "x": pixel_x,  # Pixel coordinates
                "y": pixel_y,  # Pixel coordinates
                "confidence": float(confidence)
            }
        return pose_data

    def extract_poses_from_video(self, video_path: str) -> List[Dict]:
        """Extract poses from all frames in video"""
        if not cv2.VideoCapture(video_path).isOpened():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video information: {total_frames} frames, {fps} fps")
        
        pose_data = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            print(f"Processing frame: {frame_count}/{total_frames}", end="\r")
            
            pose = self.detect_pose(frame)
            frame_data = {
                "frame_number": frame_count,
                "timestamp": frame_count / fps,
                "pose": pose
            }
            pose_data.append(frame_data)
        
        cap.release()
        print(f"\nTotal {len(pose_data)} frames extracted")
        
        return pose_data 