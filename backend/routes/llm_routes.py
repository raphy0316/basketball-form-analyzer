from fastapi import APIRouter, Depends
from backend.models.frame_data import FrameData
from backend.services.llm_service import LLMService
from typing import List

llm_router = APIRouter()

# In-memory storage for received frames
@llm_router.get("/")
async def receive_batch(
    llm_service: LLMService = Depends(LLMService)
):
    """
    Receive a batch of frames and store them for later processing.
    """
    print("This endpoint is not implemented yet.")
    return {"status": "not_implemented"}