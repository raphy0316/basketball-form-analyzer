# from fastapi import FastAPI
# import sys
# import os
# from typing import List

# # Add the parent directory to sys.path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from ball_extraction.ball_detection_layer import BallDetectionLayer
# from ball_extraction.ball_storage_layer import BallStorageLayer
# from pose_extraction.pose_extraction_pipeline import PoseExtractionPipeline
# from pose_extraction.pose_model_layer import PoseModelLayer

# from backend.models.fram_data import FrameData
# from backend.config import (
#     MIN_BALL_SIZE, MIN_BALL_CONFIDENCE, MIN_RIM_CONFIDENCE,
#     DEFAULT_FPS, BASE_FILENAME, OUTPUT_DIR, POSE_CONFIDENCE_THRESHOLD
# )

# app = FastAPI()

# # Store received data
# frames_data = []

# def process_ball_data(data):
#     if not data:
#         return
    
#     print(f"Processing {len(data)} frames")
#     ball_detection_layer = BallDetectionLayer(load_model=False)
#     ball_storage_layer = BallStorageLayer(OUTPUT_DIR)
    
#     # Extract ball trajectory and rim information
#     ball_info, rim_info = ball_detection_layer.extract_ball_trajectory_and_rim_info_api_helper(data)
    
#     # Filter detections based on confidence and size thresholds
#     filtered_ball_info = ball_detection_layer.filter_ball_detections(
#         ball_info, MIN_BALL_CONFIDENCE, MIN_BALL_SIZE
#     )
#     filtered_rim_info = ball_detection_layer.filter_rim_detections(
#         rim_info, MIN_RIM_CONFIDENCE
#     )
    
#     # Save processed data
#     ball_storage_layer.save_original_as_json(filtered_ball_info, f"ball_{BASE_FILENAME}.json")
#     ball_storage_layer.save_rim_original_as_json(filtered_rim_info, f"rim_{BASE_FILENAME}.json")
    
#     # Get and print statistics
#     stats = ball_detection_layer.get_ball_statistics(filtered_ball_info)
    
#     print("\n📋 Ball trajectory extraction summary:")
#     print(f"   • Total frames: {stats['total_frames']}")
#     print(f"   • Frames with ball: {stats['frames_with_ball']}")
#     print(f"   • Detection rate: {stats['detection_rate']:.2%}")
#     print(f"   • Total balls detected: {stats['total_balls_detected']}")
#     print(f"   • Average confidence: {stats['avg_confidence']:.3f}")
#     print(f"   • Saved file: ball_{BASE_FILENAME}.json")
    
#     return stats

# def _preprocess_pose_data(data):
#     if not data:
#         return []
        
#     pose_data = []
#     print(f"Processing {len(data)} frames")
    
#     fps = data[0].fps if data else DEFAULT_FPS
#     aspect_ratio = data[0].frameWidth / data[0].frameHeight
#     keypoints = [frame.keypoints for frame in data]
#     frame_count = 0

#     for keypoint in keypoints:
#         pose = {}
#         frame_count += 1
        
#         for kp in keypoint:
#             kp_name = kp.name
#             pose[kp_name] = {
#                 'x': kp.x * aspect_ratio,
#                 'y': kp.y,
#                 'confidence': kp.confidence
#             }
        
#         frame_data = {
#             "frame_number": frame_count,
#             "timestamp": frame_count / fps,
#             "pose": pose
#         }
#         pose_data.append(frame_data)
        
#     return pose_data

# def process_pose_data(data):
#     if not data:
#         return
    
#     pose_data = _preprocess_pose_data(data)
    
#     pose_extraction_pipeline = PoseExtractionPipeline(load_model=False)
#     filtered_data = pose_extraction_pipeline.filter_low_confidence_poses_api_helper(
#         pose_data, confidence_threshold=POSE_CONFIDENCE_THRESHOLD
#     )
    
#     saved_file = pose_extraction_pipeline.storage_layer.save_original_as_json(
#         filtered_data, f"demo_pose_{BASE_FILENAME}.json"
#     )
    
#     pose_extraction_pipeline.print_summary_api_helper(filtered_data, saved_file)
    
#     return {"filtered_frames": len(filtered_data), "saved_file": saved_file}

# @app.post("/model-output")
# async def receive_batch(batch: List[FrameData]):
#     print(f"✅ Received {len(batch)} frames")
    
#     # Store data for processing
#     frames_data.extend(batch)
    
#     # Process ball data
#     process_ball_data(frames_data)
    
#     # Log sample data
#     if batch:
#         first_frame = batch[0]
#         print('keypoints: ', first_frame.keypoints)
#         print('detections: ', first_frame.detections)
#         print('fps: ', first_frame.fps)
#         print('frameId: ', first_frame.frameId)
#         print('frameWidth: ', first_frame.frameWidth)
#         print('frameHeight: ', first_frame.frameHeight)
    
#     return {"received": len(batch), "total_frames": len(frames_data)}
    
# @app.post("/model-output/processed")
# async def receive_processed_batch():
#     print("Processing received data...")
    
#     ball_stats = process_ball_data(frames_data)
#     pose_stats = process_pose_data(frames_data)
    
#     return {
#         "status": "processed", 
#         "frames": len(frames_data),
#         "ball_stats": ball_stats,
#         "pose_stats": pose_stats
#     }

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "frames_collected": len(frames_data)}


from fastapi import FastAPI, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from backend.routes.model_routes import model_router
import backend.routes.llm_routes as llm_routes
import sys
import os
app = FastAPI()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Include routes
app.include_router(model_router, prefix="/model-output", tags=["Model Output"])
app.include_router(model_router, prefix="/api", tags=["Model Routes"])
app.include_router(llm_routes.llm_router, prefix="/llm", tags=["LLM Routes"])

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request: Request, exc: RequestValidationError):
#     return JSONResponse(
#         status_code=400,
#         content={"error": str(exc)},
#     )