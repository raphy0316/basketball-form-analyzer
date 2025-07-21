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
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
   
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
         # Confidence score to determine whether a keypoint prediction is reliable.
        self.MIN_CROP_KEYPOINT_SCORE = 0.3
        self._load_model()
        self.crop_region = None
         # Dictionary that maps from joint names to keypoint indices.
        self.KEYPOINT_DICT = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }

    # Maps bones to a matplotlib color name.
        self.KEYPOINT_EDGE_INDS_TO_COLOR = {
            (0, 1): 'm',
            (0, 2): 'c',
            (1, 3): 'm',
            (2, 4): 'c',
            (0, 5): 'm',
            (0, 6): 'c',
            (5, 7): 'm',
            (7, 9): 'm',
            (6, 8): 'c',
            (8, 10): 'c',
            (5, 6): 'y',
            (5, 11): 'm',
            (6, 12): 'c',
            (11, 12): 'y',
            (11, 13): 'm',
            (13, 15): 'm',
            (12, 14): 'c',
            (14, 16): 'c'
        }
    def _load_model(self):
        """Load MoveNet model"""
        model_url = "https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder/4"
        # model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
        self.movenet = hub.load(model_url)
        self.model = self.movenet.signatures["serving_default"]
        print("MoveNet model loading completed")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (256, 256))
        input_frame = np.expand_dims(resized_frame, axis=0).astype(np.int32)
        return input_frame

    def _transform_crop_to_full_frame(self, crop_x: float, crop_y: float, crop_region: Dict, 
                                    original_h: int, original_w: int) -> Tuple[float, float]:
        """
        Transform crop coordinates to full frame relative coordinates (0~1)
        
        Args:
            crop_x: x coordinate in crop (0~1)
            crop_y: y coordinate in crop (0~1)
            crop_region: crop region info (relative coordinates 0~1)
            original_h: original frame height
            original_w: original frame width
            
        Returns:
            Tuple of (full_frame_relative_x, full_frame_relative_y) in relative coordinates (0~1)
        """
        # Transform from crop coordinates (0~1) to full frame relative coordinates (0~1)
        full_frame_relative_x = crop_region['x_min'] + (crop_x * (crop_region['x_max'] - crop_region['x_min']))
        full_frame_relative_y = crop_region['y_min'] + (crop_y * (crop_region['y_max'] - crop_region['y_min']))
        
        return full_frame_relative_x, full_frame_relative_y

    def _apply_aspect_ratio_correction(self, x: float, y: float, frame_h: int, frame_w: int) -> Tuple[float, float]:
        """
        Apply aspect ratio correction to ensure same relative coordinate change
        corresponds to same pixel distance change in both x and y axes.
        
        Args:
            x: x coordinate (0~1 relative)
            y: y coordinate (0~1 relative)
            frame_h: frame height
            frame_w: frame width
            
        Returns:
            Tuple of (corrected_x, y) where x is normalized for aspect ratio, y remains 0~1
        """
        # Calculate aspect ratio
        aspect_ratio = frame_w / frame_h
        
        # Normalize x coordinate so that same relative change corresponds to same pixel change
        # We want: delta_x * frame_w = delta_y * frame_h
        # So: corrected_x * frame_w = y * frame_h
        # Therefore: corrected_x = y * frame_h / frame_w = y / aspect_ratio
        corrected_x = x / aspect_ratio
        corrected_y = y
        
        return corrected_x, corrected_y

    def detect_pose(self, frame: np.ndarray, output_images: List[np.ndarray]) -> Dict:
        """Detect pose from single frame (return normalized coordinates with aspect ratio correction on x-axis)"""
        input_frame = self.preprocess_frame(frame)
        # results = self.model(input=input_frame)
        # keypoints = results["output_0"].numpy()
        # keypoints = keypoints[0, 0]

        # Original frame size
        h, w = frame.shape[:2]

        self.crop_region = self.init_crop_region(h, w) if self.crop_region is None else self.crop_region
        keypoints_with_scores = self.run_inference(self.model, frame, self.crop_region, crop_size=[256, 256])
        
        self.crop_region = self.determine_crop_region(keypoints_with_scores, h, w)
        # print("After determine crop region")
        pose_data = {}
        for i, name in enumerate(self.keypoint_names):
            # Ensure proper indexing of keypoints
            y, x, confidence = keypoints_with_scores[0, 0][i]
            
            corrected_x, y  = self._apply_aspect_ratio_correction(x, y, h, w)
            
            pose_data[name] = {
                "x": float(corrected_x),  # Aspect ratio corrected 0~1 relative coordinates
                "y": float(y),            # 0~1 relative coordinates
                "confidence": float(confidence)
            }
        return pose_data

    def detect_and_draw_rim(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect basketball rim in the frame and draw a bounding box around it
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with bounding box around detected rim
        """
        # Create a copy of the frame to draw on
        output_frame = frame.copy()
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape to find potential rims
        potential_rims = []
        for contour in contours:
            # Calculate area and perimeter
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filter by area
            if area > 100:
                # Calculate circularity
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Basketball rims are circular, so check for high circularity
                if 0.5 < circularity < 1.2:
                    # Get bounding rect
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (rim should be roughly circular)
                    aspect_ratio = float(w) / h
                    if 0.8 < aspect_ratio < 1.3:
                        potential_rims.append((x, y, w, h))
        
        # Draw boxes around detected rims
        for x, y, w, h in potential_rims:
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_frame, 'Rim', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return output_frame

    def extract_poses_from_video(self, video_path: str, detect_rim: bool = False) -> List[Dict]:
        """Extract poses from all frames in video"""
        if not cv2.VideoCapture(video_path).isOpened():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video information: {total_frames} frames, {fps} fps")
        
        pose_data = []
        output_images = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            print(f"Processing frame: {frame_count}/{total_frames}", end="\r")

            # Detect rim if requested
            if detect_rim:
                frame = self.detect_and_draw_rim(frame)
                
            pose = self.detect_pose(frame, output_images)
            frame_data = {
                "frame_number": frame_count,
                "timestamp": frame_count / fps,
                "pose": pose
            }
            pose_data.append(frame_data)
        
        cap.release()
        print(f"\nTotal {len(pose_data)} frames extracted")
        for image in output_images:
            cv2.imshow("crop algo image: ", image)
            cv2.waitKey(100)

        return pose_data 

    def init_crop_region(self, image_height, image_width):
        """Defines the default crop region.

        The function provides the initial crop region (pads the full image from both
        sides to make it a square image) when the algorithm cannot reliably determine
        the crop region from the previous frame.
        """
        if image_width > image_height:
            # Width > Height: pad height to make square
            box_height = image_width / image_height  # > 1.0
            box_width = 1.0
            # Center the crop region vertically
            y_min = max(0.0, (1.0 - box_height) / 2)  # Ensure non-negative
            x_min = 0.0
        else:
            # Height >= Width: pad width to make square
            box_height = 1.0
            box_width = image_height / image_width  # >= 1.0
            y_min = 0.0
            # Center the crop region horizontally
            x_min = max(0.0, (1.0 - box_width) / 2)  # Ensure non-negative

        return {
            'y_min': y_min,
            'x_min': x_min,
            'y_max': y_min + box_height,
            'x_max': x_min + box_width,
            'height': box_height,
            'width': box_width
        }

    def torso_visible(self, keypoints):
        """Checks whether there are enough torso keypoints.

        This function checks whether the model is confident at predicting one of the
        shoulders/hips which is required to determine a good crop region.
        """
        return ((keypoints[0, 0, self.KEYPOINT_DICT['left_hip'], 2] >
                self.MIN_CROP_KEYPOINT_SCORE or
                keypoints[0, 0, self.KEYPOINT_DICT['right_hip'], 2] >
                self.MIN_CROP_KEYPOINT_SCORE) and
                (keypoints[0, 0, self.KEYPOINT_DICT['left_shoulder'], 2] >
                self.MIN_CROP_KEYPOINT_SCORE or
                keypoints[0, 0, self.KEYPOINT_DICT['right_shoulder'], 2] >
                self.MIN_CROP_KEYPOINT_SCORE))

    def determine_torso_and_body_range(self,
            keypoints, target_keypoints, center_y, center_x):
        """Calculates the maximum distance from each keypoints to the center location.

        The function returns the maximum distances from the two sets of keypoints:
        full 17 keypoints and 4 torso keypoints. The returned information will be
        used to determine the crop size. See determineCropRegion for more detail.
        """
        torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for joint in self.KEYPOINT_DICT.keys():
            if keypoints[0, 0, self.KEYPOINT_DICT[joint], 2] < self.MIN_CROP_KEYPOINT_SCORE:
                continue
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y

            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

    def determine_crop_region(self,
        keypoints, image_height,
        image_width):
        """Determines the region to crop the image for the model to run inference on.

        The algorithm uses the detected joints from the previous frame to estimate
        the square region that encloses the full body of the target person and
        centers at the midpoint of two hip joints. The crop size is determined by
        the distances between each joints and the center point.
        When the model is not confident with the four torso joint predictions, the
        function returns a default crop which is the full image padded to square.
        """
        target_keypoints = {}
        for joint in self.KEYPOINT_DICT.keys():
            # MoveNet output: [y, x, confidence] order
            y, x, confidence = keypoints[0, 0, self.KEYPOINT_DICT[joint]]
            target_keypoints[joint] = [
                y * image_height,  # Y coordinate
                x * image_width    # X coordinate
            ]

        if self.torso_visible(keypoints):
            center_y = (target_keypoints['left_hip'][0] +
                        target_keypoints['right_hip'][0]) / 2
            center_x = (target_keypoints['left_hip'][1] +
                        target_keypoints['right_hip'][1]) / 2

            (max_torso_yrange, max_torso_xrange,
            max_body_yrange, max_body_xrange) = self.determine_torso_and_body_range(
                keypoints, target_keypoints, center_y, center_x)

            crop_length_half = np.amax(
                [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
                max_body_yrange * 1.2, max_body_xrange * 1.2])

            tmp = np.array(
                [center_x, image_width - center_x, center_y, image_height - center_y])
            crop_length_half = np.amin(
                [crop_length_half, np.amax(tmp)])

            crop_corner = [center_y - crop_length_half, center_x - crop_length_half]

            if crop_length_half > max(image_width, image_height) / 2:
                return self.init_crop_region(image_height, image_width)
            else:
                crop_length = crop_length_half * 2
                return {
                    'y_min': crop_corner[0] / image_height,
                    'x_min': crop_corner[1] / image_width,
                    'y_max': (crop_corner[0] + crop_length) / image_height,
                    'x_max': (crop_corner[1] + crop_length) / image_width,
                    'height': (crop_corner[0] + crop_length) / image_height -
                        crop_corner[0] / image_height,
                    'width': (crop_corner[1] + crop_length) / image_width -
                        crop_corner[1] / image_width
                }
        else:
            return self.init_crop_region(image_height, image_width)

    def crop_and_resize(self, image, crop_region, crop_size):
        """Crops and resize the image to prepare for the model input."""
        boxes=[[crop_region['y_min'], crop_region['x_min'],
                crop_region['y_max'], crop_region['x_max']]]
        # Make sure the image is in the correct format (int32)
        output_image = tf.image.crop_and_resize(
            image, box_indices=[0], boxes=boxes, crop_size=crop_size)
        return output_image

    def run_inference(self, model, image, crop_region, crop_size):
        """Runs model inference on the cropped region.

        The function runs the model inference on the cropped region and updates the
        model output to the original image coordinate system.
        """
        image_height, image_width, _ = image.shape
        
        # Convert the image to int32 before expanding dimensions
        image_int32 = tf.cast(image, tf.int32)
        input_image = self.crop_and_resize(
            tf.expand_dims(image_int32, axis=0), crop_region, crop_size=crop_size)
        
        # Run model inference.
        input_image = tf.cast(input_image, tf.int32)
        outputs = model(input=input_image)
        keypoints_with_scores = outputs["output_0"].numpy()
        
        # Update the coordinates.
        for idx in range(17):
            keypoints_with_scores[0, 0, idx, 0] = (
                crop_region['y_min'] * image_height +
                crop_region['height'] * image_height *
                keypoints_with_scores[0, 0, idx, 0]) / image_height
            keypoints_with_scores[0, 0, idx, 1] = (
                crop_region['x_min'] * image_width +
                crop_region['width'] * image_width *
                keypoints_with_scores[0, 0, idx, 1]) / image_width
        return keypoints_with_scores
