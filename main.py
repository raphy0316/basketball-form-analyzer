from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the integrated pipeline
from basketball_shooting_integrated_pipeline import BasketballShootingIntegratedPipeline

app = FastAPI(
    title="Basketball Form Analyzer API",
    description="API for analyzing basketball shooting form using computer vision",
    version="1.0.0"
)

# Add CORS middleware to allow requests from mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your mobile app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the pipeline
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize the basketball analysis pipeline on startup"""
    global pipeline
    try:
        print("Initializing Basketball Analysis Pipeline...")
        pipeline = BasketballShootingIntegratedPipeline()
        print("Pipeline initialized successfully!")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        pipeline = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Basketball Form Analyzer API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "version": "1.0.0"
    }

@app.post("/analysis/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    """
    Analyze a basketball shooting video and return form analysis results
    
    Args:
        video: Uploaded video file (MP4 format recommended)
    
    Returns:
        JSON response with analysis results including:
        - player_match: NBA player with most similar form
        - similarity_score: Percentage similarity (0.0 to 1.0)
        - phase_scores: Breakdown of different shooting phases
        - feedback: List of improvement suggestions
    """
    try:
        # Validate file type
        if not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Check if pipeline is ready
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Analysis pipeline is not ready")
        
        print(f"Received video: {video.filename}, size: {video.size} bytes")
        
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            shutil.copyfileobj(video.file, temp_file)
            temp_video_path = temp_file.name
        
        try:
            # Run the analysis pipeline
            print("Starting video analysis...")
            results = pipeline.analyze_video(temp_video_path)
            
            # Format the response
            response = {
                "player_match": results.get("best_match_player", "Unknown Player"),
                "similarity_score": results.get("similarity_score", 0.0),
                "phase_scores": {
                    "dip": results.get("dip_score", 0.0),
                    "setpoint": results.get("setpoint_score", 0.0),
                    "release": results.get("release_score", 0.0),
                    "follow_through": results.get("follow_through_score", 0.0)
                },
                "feedback": results.get("feedback", ["Analysis completed successfully."])
            }
            
            print(f"Analysis completed. Best match: {response['player_match']} ({response['similarity_score']:.2f})")
            return JSONResponse(content=response)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
                
    except Exception as e:
        print(f"Error analyzing video: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analysis/analyze-video-test")
async def analyze_video_test():
    """
    Test endpoint that returns mock analysis results
    Useful for testing the mobile app without running the full pipeline
    """
    return {
        "player_match": "Kevin Durant",
        "similarity_score": 0.82,
        "phase_scores": {
            "dip": 0.78,
            "setpoint": 0.85,
            "release": 0.81,
            "follow_through": 0.84
        },
        "feedback": [
            "Try to lower your dip slightly for better consistency.",
            "Your release angle is very similar to Kevin Durant's form.",
            "Consider extending your follow-through a bit more."
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
