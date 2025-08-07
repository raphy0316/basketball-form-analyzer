from ball_extraction.ball_detection_layer import BallDetectionLayer
from ball_extraction.ball_storage_layer import BallStorageLayer
from backend.config import MIN_BALL_CONFIDENCE, MIN_BALL_SIZE, MIN_RIM_CONFIDENCE, OUTPUT_DIR

class BallService:
    def __init__(self):
        self.ball_detection_layer = BallDetectionLayer(load_model=False)
        self.ball_storage_layer = BallStorageLayer(OUTPUT_DIR)

    def process_ball_data(self, data):
        """
        Process ball data from the received frames.
        """
        if not data:
            return {}

        print(f"Processing {len(data)} frames for ball data")

        # Extract ball trajectory and rim information
        ball_info, rim_info = self.ball_detection_layer.extract_ball_trajectory_and_rim_info_api_helper(data)

        # Filter detections based on confidence and size thresholds
        filtered_ball_info = self.ball_detection_layer.filter_ball_detections(
            ball_info, MIN_BALL_CONFIDENCE, MIN_BALL_SIZE
        )
        filtered_rim_info = self.ball_detection_layer.filter_rim_detections(
            rim_info, MIN_RIM_CONFIDENCE
        )
    
        # Save processed data
        saved_file = self.ball_storage_layer.save_original_as_json(filtered_ball_info, "demo_ball_original.json")
        self.ball_storage_layer.save_rim_original_as_json(filtered_rim_info, "demo_rim_original.json")

        # Get and return statistics
        stats = self.ball_detection_layer.get_ball_statistics(filtered_ball_info)
        return saved_file