from pose_extraction.pose_extraction_pipeline import PoseExtractionPipeline
from backend.config import POSE_CONFIDENCE_THRESHOLD, BASE_FILENAME

class PoseService:
    def __init__(self):
        self.pose_extraction_pipeline = PoseExtractionPipeline(load_model=False)

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

        # Save processed data
        saved_file = self.pose_extraction_pipeline.storage_layer.save_original_as_json(
            filtered_data, f"demo_pose_{BASE_FILENAME}.json"
        )

        # Print summary and return stats
        self.pose_extraction_pipeline.print_summary_api_helper(filtered_data, saved_file)
        return {"filtered_frames": len(filtered_data), "saved_file": saved_file}

    def _preprocess_pose_data(self, data):
        """
        Preprocess pose data from frames.
        """
        pose_data = []
        for frame in data:
            pose = {kp.name: {"x": kp.x, "y": kp.y, "confidence": kp.confidence} for kp in frame.keypoints}
            pose_data.append({"frameId": frame.frameId, "pose": pose})
        return pose_data