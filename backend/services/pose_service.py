from pose_extraction.pose_extraction_pipeline import PoseExtractionPipeline
from backend.config import POSE_CONFIDENCE_THRESHOLD, BASE_FILENAME, OUTPUT_DIR, DEFAULT_FPS

class PoseService:
    def __init__(self):
        self.pose_extraction_pipeline = PoseExtractionPipeline(output_dir = OUTPUT_DIR, load_model=False)

    def process_pose_data(self, data):
        """
        Process pose data from the received frames.
        """
        if not data:
            return {}

        print(f"Processing {len(data)} frames for pose data")

        # Preprocess pose data
        pose_data = self._preprocess_pose_data(data)

        # Filter low-confidence poses
        filtered_data = self.pose_extraction_pipeline.filter_low_confidence_poses_api_helper(
            pose_data, confidence_threshold=POSE_CONFIDENCE_THRESHOLD
        )
        # print(filtered_data)
        # Save processed data
        saved_file = self.pose_extraction_pipeline.storage_layer.save_original_as_json(
            filtered_data, f"{BASE_FILENAME}_pose_original.json"
        )

        # Print summary and return stats
        self.pose_extraction_pipeline.print_summary_api_helper(filtered_data, saved_file)
        return saved_file

    def _preprocess_pose_data(self, data):
        """
        Preprocess pose data from frames.
        """
        pose_data = []
        frame_count = 0
        fps = data[0].fps if data else DEFAULT_FPS
        aspect_ratio = data[0].frameWidth / data[0].frameHeight if data else 1.0

        for frame in data:
            frame_count += 1 
            pose = {kp.name: {"x": kp.x * aspect_ratio, "y": kp.y, "confidence": kp.confidence} for kp in frame.keypoints}
            frame_data = {
                "frame_number": frame_count,
                "timestamp": frame_count / fps,
                "pose": pose
            }
            pose_data.append(frame_data)
        return pose_data