import glob
from ball_extraction.ball_detection_layer import BallDetectionLayer
from ball_extraction.ball_storage_layer import BallStorageLayer
from backend.config import DEFAULT_FPS, BASE_FILENAME, OUTPUT_DIR
from basketball_shooting_analyzer import BasketballShootingAnalyzer
from pose_extraction.pose_extraction_pipeline import PoseExtractionPipeline
from backend.services.pose_service import PoseService
from backend.services.ball_service import BallService
from basketball_shooting_integrated_pipeline import BasketballShootingIntegratedPipeline
from shooting_comparison.shooting_comparison_pipeline import ShootingComparisonPipeline
import concurrent.futures
import os
import json

class AnalysisService:
    # def __init__(self):
        # self.analyzer = BasketballShootingAnalyzer(interactive=False,fps=DEFAULT_FPS)
        # self.ball_detection_layer = BallDetectionLayer(load_model=False)
        # self.ball_storage_layer = BallStorageLayer(OUTPUT_DIR)
        # self.pose_extraction_pipeline = PoseExtractionPipeline(output_dir = OUTPUT_DIR, load_model=False)
        # self.pose_service = PoseService()
        # self.ball_service = BallService()
        # self.extracted_data_dir = "data/extracted_data"
        # self.results_dir = "data/results"
        # self.visualized_video_dir = "data/visualized_video"
        # self.video_fps = DEFAULT_FPS
        
    # def _extract_pose(self, data):
    #     """
    #     Extract pose data from the received frames.
    #     """
    #     if not data:
    #         return {}

    #     print(f"Processing {len(data)} frames for pose data")

    #     # Use the PoseService to process pose data
    #     saved_file = self.pose_service.process_pose_data(data)
    #     return saved_file

    # def _extract_ball(self, data):
    #     """
    #     Extract ball data from the received frames.
    #     """
    #     if not data:
    #         return {}

    #     print(f"Processing {len(data)} frames for ball data")

    #     # Use the BallService to process ball data
    #     saved_file = self.ball_service.process_ball_data(data)
    #     return saved_file

    # def _extract_original_data(self, data):
    #     """
    #     Extract original data from the received frames.
    #     """
    #     if not data:
    #         return {}

    #     print(f"Processing {len(data)} frames for original data")

    #     try:
    #             # Use ThreadPoolExecutor for parallel execution
    #         with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    #             # Submit both extraction tasks
    #             pose_future = executor.submit(self._extract_pose, data)
    #             ball_future = executor.submit(self._extract_ball, data)
                
    #             # Wait for both to complete and get results
    #             pose_file = pose_future.result()
    #             ball_file = ball_future.result()
                
    #             print("Both pose and ball extraction completed in parallel")

    #         return True

    #     except Exception as e:
    #         print(f"Failed to extract data: {e}")
    #         return False

    # def _normalize_pose_data(self):
    #     """
    #     Normalize pose data to a standard format.
    #     """
    #     self.analyzer.normalize_pose_data(video_path="demo")  

    # def _segment_phases(self):
    #     """
    #     Segment the phases of the basketball shooting process.
    #     """
    #     self.analyzer.segment_shooting_phases("hybrid_fps")
    #     return True

    # def _save_results(self):
    #     self.analyzer.save_results(video_path="demo", overwrite_mode=True)

    # def _load_associated_data(self, base_name):
    #     return self.analyzer.load_associated_data_api_helper(base_name)

    # def run_analysis(self, data):
    #     """
    #     Run the complete analysis pipeline on the received frames.
    #     """
    #     if not data:
    #         print("No data received for analysis.")
    #         return False

    #     # Step 1: Extract original data
    #     if not self._extract_original_data(data):
    #         return False

    #     # Step 2: Load associated data
    #     if not self._load_associated_data(base_name=BASE_FILENAME):
    #         print("Failed to load associated data.")
    #         return False

    #     # Step 3: Normalize pose data
    #     self._normalize_pose_data()

    #     # Step 4: Segment phases
    #     if not self._segment_phases():
    #         return False

    #     # Step 5: Save results
    #     self._save_results()

    #     print("Analysis completed successfully!")
    #     return True
    
    def run_pipeline(self, saved_path):
        pipeline = BasketballShootingIntegratedPipeline()
        pipeline.run_full_pipeline(saved_path, overwrite_mode=True, use_existing_extraction=False)
        comparison_pipeline = ShootingComparisonPipeline()
        comparison_pipeline.process_video_data(saved_path)