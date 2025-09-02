#!/usr/bin/env python3
"""
Basketball Form Analyzer Backend with Synthetic Motion Profiles Integration

This FastAPI backend provides endpoints for analyzing basketball shots and comparing
them with synthetic NBA player motion profiles.
"""

import os
import sys
# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from fastapi.staticfiles import StaticFiles

from backend.services.analysis_service import analyze_video_service, compare_with_player_service

app = FastAPI(
    title="Basketball Form Analyzer - Synthetic Profiles Integration",
    description="API for analyzing basketball shooting form with synthetic NBA player comparisons",
    version="1.0.0"
)

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../shooting_comparison/results"))
app.mount("/results", StaticFiles(directory=results_dir), name="results")

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include existing routes
# # app.include_router(model_router, prefix="/api", tags=["Model Routes"])
# app.include_router(llm_routes.llm_router, prefix="/llm", tags=["LLM Routes"])

@app.post("/analysis/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    return JSONResponse(content=analyze_video_service(video))

@app.post("/analysis/compare-with-player")
async def compare_with_player(
    video: UploadFile = File(...),
    player_id: str = Form(...),
    player_style: str = Form(...)
):
    return JSONResponse(content=compare_with_player_service(video, player_id, player_style))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
