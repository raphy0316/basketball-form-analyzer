# analyze_routes.py

import os
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from basketball_shooting_integrated_pipeline import run_pipeline
from datetime import datetime
import shutil

router = APIRouter()

@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    try:
        # Save uploaded file to video directory
        os.makedirs("data/video/from_mobile", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mobile_upload_{timestamp}_{file.filename}"
        save_path = os.path.join("data/video/from_mobile", filename)

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📥 Received video: {filename}")

        # Run the full analysis pipeline
        result = run_pipeline(save_path)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e)
        })

