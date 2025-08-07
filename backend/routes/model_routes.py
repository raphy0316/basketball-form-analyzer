from fastapi import APIRouter, Depends
from backend.models.frame_data import FrameData
from backend.services.ball_service import BallService
from backend.services.pose_service import PoseService
from backend.services.analysis_service import AnalysisService

from typing import List
import os
from fastapi import File, UploadFile

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
import os


model_router = APIRouter()

# In-memory storage for received frames
frames_data = []
@model_router.post("/")
async def receive_batch(file: UploadFile = File(...)):
    """
    Receive a batch of frames and store them for later processing.
    """
    # print(f"✅ Received {len(batch)} frames")
    # frames_data.extend(batch)
    # return {"received": len(batch), "total_frames": len(frames_data)}

@model_router.post("/processed")
async def process_aggregated_data(
    analysis_service: AnalysisService = Depends(AnalysisService),
):
    """
    Process the aggregated data (both ball and pose data).
    """
    print("Processing aggregated data...")

    # # Process ball data
    # ball_stats = ball_service.process_ball_data(frames_data)

    # # Process pose data
    # pose_stats = pose_service.process_pose_data(frames_data)
    analysis_service.run_analysis(frames_data)
    data_size = len(frames_data)
    frames_data.clear()  # Clear the frames after processing
    return {
        "status": "processed",
        "frames": data_size,
        # "ball_stats": ball_stats,
        # "pose_stats": pose_stats,
    }

UPLOAD_DIRECTORY = "uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# In-memory storage for received frames
frames_data = []

@model_router.post("/upload-video")
async def upload_video(file: UploadFile = File(...),
    analysis_service: AnalysisService = Depends(AnalysisService)):
    try:
        os.makedirs("videos", exist_ok=True)
        save_path = f"videos/{file.filename}"
        print("Saving file to:", save_path)
        print(save_path)
        with open(save_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        analysis_service.run_pipeline(save_path)
        
        return {"filename": file.filename}
    
    except Exception as e:
        # Return only safe string message, avoid returning exception object directly
        return JSONResponse(status_code=500, content={"error": str(e)})