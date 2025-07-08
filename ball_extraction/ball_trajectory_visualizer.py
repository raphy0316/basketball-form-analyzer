# -*- coding: utf-8 -*-
"""
Basketball trajectory visualization tool
Overlay the trajectory of a basketball detected by YOLOv8 on a video
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class BallTrajectoryVisualizer:
    def __init__(self, model_path: str = "ball_extraction/yolov8n736-customContinue.pt"):
        """
        Initialize basketball trajectory visualization
        
        Args:
            model_path: Path to the YOLOv8 model file
        """
        self.model_path = model_path
        self.trajectory_history = []  # trajectory history
        self.max_history = 30  # maximum trajectory length
        
    def detect_and_visualize_frame(self, frame: np.ndarray, conf_threshold: float = 0.15,
                                 classes: List[int] = [0, 1, 2], iou_threshold: float = 0.1) -> np.ndarray:
        """
        Detect and visualize a basketball in a single frame
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold
            classes: Classes to detect
            iou_threshold: IoU threshold
            
        Returns:
            Visualized frame
        """
        # Load YOLOv8 model and detect basketball
        from ultralytics import YOLO
        model = YOLO(self.model_path)
        
        results = model(frame, conf=conf_threshold, classes=classes, 
                       iou=iou_threshold, imgsz=736, verbose=False)
        
        # Store current basketball position
        current_balls = []
        
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
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        current_balls.append({
                            'center': (center_x, center_y),
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence
                        })
        
        # Add current basketball position to trajectory history
        if current_balls:
            self.trajectory_history.append(current_balls)
        else:
            self.trajectory_history.append([])
        
        # Limit trajectory length
        if len(self.trajectory_history) > self.max_history:
            self.trajectory_history.pop(0)
        
        # Create visualized frame
        visualized_frame = self._draw_trajectory(frame)
        
        return visualized_frame
    
    def _draw_trajectory(self, frame: np.ndarray) -> np.ndarray:
        """Draw trajectory on frame"""
        result_frame = frame.copy()
        
        # Draw trajectory
        for i, frame_balls in enumerate(self.trajectory_history):
            if not frame_balls:
                continue
                
            # Calculate transparency based on recentness (more recent frames are darker)
            alpha = (i + 1) / len(self.trajectory_history)
            
            for ball in frame_balls:
                center = ball['center']
                bbox = ball['bbox']
                confidence = ball['confidence']
                
                # Draw circle at basketball position
                color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)  # Green or orange
                cv2.circle(result_frame, center, 8, color, -1)
                cv2.circle(result_frame, center, 10, (255, 255, 255), 2)
                
                # Draw bounding box
                cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Confidence text
                cv2.putText(result_frame, f"{confidence:.2f}", 
                           (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
        
        # Draw trajectory line
        if len(self.trajectory_history) > 1:
            for ball_idx in range(len(self.trajectory_history[0])):
                trajectory_points = []
                
                for frame_balls in self.trajectory_history:
                    if len(frame_balls) > ball_idx:
                        trajectory_points.append(frame_balls[ball_idx]['center'])
                
                if len(trajectory_points) > 1:
                    # Draw trajectory line
                    for i in range(len(trajectory_points) - 1):
                        pt1 = trajectory_points[i]
                        pt2 = trajectory_points[i + 1]
                        
                        # Color change based on time
                        color_intensity = int(255 * (i + 1) / len(trajectory_points))
                        color = (0, color_intensity, 255 - color_intensity)
                        
                        cv2.line(result_frame, pt1, pt2, color, 2)
        
        return result_frame
    
    def create_trajectory_video(self, input_video_path: str, output_video_path: Optional[str] = None,
                               conf_threshold: float = 0.15, classes: List[int] = [0, 1, 2],
                               iou_threshold: float = 0.1) -> str:
        """
        Create a new video with visualized basketball trajectory from a video
        
        Args:
            input_video_path: Path to the input video
            output_video_path: Path to the output video (None for automatic generation)
            conf_threshold: Confidence threshold
            classes: Classes to detect
            iou_threshold: IoU threshold
            
        Returns:
            Path to the generated video file
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Could not find input video file: {input_video_path}")
        
        # Automatic output file name generation
        if output_video_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(input_video_path))[0]
            output_video_path = f"ball_trajectory_{base_name}_{timestamp}.mp4"
        
        # Video capture
        cap = cv2.VideoCapture(input_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Starting to create basketball trajectory video:")
        print(f"   • Input: {input_video_path}")
        print(f"   • Output: {output_video_path}")
        print(f"   • Frames: {total_frames} frames, FPS: {fps}")
        print(f"   • Resolution: {width}x{height}")
        
        # Video writer setup (H264 codec used)
        fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print("VideoWriter could not be opened!")
            return ""
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame: {frame_count}/{total_frames}", end="\r")
            
            # Detect and visualize
            visualized_frame = self.detect_and_visualize_frame(
                frame, conf_threshold, classes, iou_threshold
            )
            
            # Add frame to video
            out.write(visualized_frame)
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"\n✅ Basketball trajectory video creation completed: {output_video_path}")
        
        return output_video_path
    
    def visualize_single_frame(self, frame: np.ndarray, conf_threshold: float = 0.15,
                             classes: List[int] = [0, 1, 2], iou_threshold: float = 0.1) -> np.ndarray:
        """
        Visualize a single frame (initialize trajectory history)
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold
            classes: Classes to detect
            iou_threshold: IoU threshold
            
        Returns:
            Visualized frame
        """
        # Initialize trajectory history
        self.trajectory_history = []
        
        return self.detect_and_visualize_frame(frame, conf_threshold, classes, iou_threshold)
    
    def get_trajectory_statistics(self) -> Dict:
        """Return trajectory statistics information"""
        total_frames = len(self.trajectory_history)
        frames_with_ball = sum(1 for frame_balls in self.trajectory_history if frame_balls)
        
        # Collect confidence of all balls
        confidences = []
        for frame_balls in self.trajectory_history:
            for ball in frame_balls:
                confidences.append(ball['confidence'])
        
        stats = {
            "total_frames": total_frames,
            "frames_with_ball": frames_with_ball,
            "detection_rate": frames_with_ball / total_frames if total_frames > 0 else 0,
            "total_balls_detected": len(confidences),
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "min_confidence": np.min(confidences) if confidences else 0,
            "max_confidence": np.max(confidences) if confidences else 0
        }
        
        return stats

def main():
    """Main execution function"""
    print("🏀 Basketball trajectory visualization tool")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = BallTrajectoryVisualizer()
    
    # Set video file path
    input_video = "../References/stephen_curry_multy_person_part.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Could not find video file: {input_video}")
        return
    
    try:
        # Create basketball trajectory video
        output_video = visualizer.create_trajectory_video(
            input_video_path=input_video,
            conf_threshold=0.15,
            classes=[0, 1, 2],
            iou_threshold=0.1
        )
        
        # Print statistics information
        stats = visualizer.get_trajectory_statistics()
        print(f"\n📊 Trajectory statistics:")
        print(f"   • Total frames: {stats['total_frames']}")
        print(f"   • Frames with ball: {stats['frames_with_ball']}")
        print(f"   • Detection rate: {stats['detection_rate']:.2%}")
        print(f"   • Total detected balls: {stats['total_balls_detected']} balls")
        print(f"   • Average confidence: {stats['avg_confidence']:.3f}")
        
        print(f"\n🎉 Basketball trajectory visualization completed!")
        print(f"   • Output file: {output_video}")
        
    except Exception as e:
        print(f"❌ Error occurred during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 