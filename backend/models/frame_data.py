from pydantic import BaseModel
from typing import List

class Keypoint(BaseModel):
    id: int
    name: str
    x: float
    y: float
    confidence: float

class Detection(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    classId: int

class FrameData(BaseModel):
    frameId: int
    frameWidth: int
    frameHeight: int
    keypoints: List[Keypoint]
    detections: List[Detection]
    fps: float
