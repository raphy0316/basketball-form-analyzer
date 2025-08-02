from fastapi import APIRouter, Depends
from backend.models.frame_data import FrameData
from backend.services.ball_service import BallService
from backend.services.pose_service import PoseService
from typing import List

model_output_router = APIRouter()

# In-memory storage for received frames
frames_data = []

@model_output_router.post("/")
async def receive_batch(batch: List[FrameData]):
    """
    Receive a batch of frames and store them for later processing.
    """
    print(f"✅ Received {len(batch)} frames")
    frames_data.extend(batch)
    return {"received": len(batch), "total_frames": len(frames_data)}

@model_output_router.post("/processed")
async def process_aggregated_data(
    ball_service: BallService = Depends(),
    pose_service: PoseService = Depends()
):
    """
    Process the aggregated data (both ball and pose data).
    """
    print("Processing aggregated data...")

    # Process ball data
    ball_stats = ball_service.process_ball_data(frames_data)

    # Process pose data
    pose_stats = pose_service.process_pose_data(frames_data)

    return {
        "status": "processed",
        "frames": len(frames_data),
        "ball_stats": ball_stats,
        "pose_stats": pose_stats,
    }