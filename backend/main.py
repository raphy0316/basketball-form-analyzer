#!/usr/bin/env python3
"""
Basketball Form Analyzer Backend with Synthetic Motion Profiles Integration

This FastAPI backend provides endpoints for analyzing basketball shots and comparing
them with synthetic NBA player motion profiles.
"""

import os
import sys
import json
import tempfile
import shutil
from typing import Dict, List, Optional
from pathlib import Path
from backend.utils.save_file import save_json_to_directory

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import existing routes
from backend.routes.model_routes import model_router
import backend.routes.llm_routes as llm_routes
from shooting_comparison.shooting_comparison_pipeline import ShootingComparisonPipeline

# Import synthetic profiles
try:
    from synthetic_profiles.motion_profile_generator import (
        SyntheticProfileGenerator,
        create_lebron_style,
        create_curry_style,
        create_durant_style,
        create_kawhi_style,
        create_harden_style
    )
    SYNTHETIC_AVAILABLE = True
except ImportError:
    # Try relative import
    try:
        from ..synthetic_profiles.motion_profile_generator import (
            SyntheticProfileGenerator,
            create_lebron_style,
            create_curry_style,
            create_durant_style,
            create_kawhi_style,
            create_harden_style
        )
        SYNTHETIC_AVAILABLE = True
    except ImportError:
        SYNTHETIC_AVAILABLE = False
        print("Warning: Synthetic profiles not available. Using mock data.")

# Import analysis components (if available)
try:
    from basketball_shooting_integrated_pipeline import BasketballShootingIntegratedPipeline
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    print("Warning: Basketball analysis pipeline not available. Using mock analysis.")

app = FastAPI(
    title="Basketball Form Analyzer - Synthetic Profiles Integration",
    description="API for analyzing basketball shooting form with synthetic NBA player comparisons",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include existing routes
app.include_router(model_router, prefix="/api", tags=["Model Routes"])
app.include_router(llm_routes.llm_router, prefix="/llm", tags=["LLM Routes"])

# Initialize components
if SYNTHETIC_AVAILABLE:
    generator = SyntheticProfileGenerator()
    
    # Player styles mapping
    PLAYER_STYLES = {
        'lebron': create_lebron_style,
        'curry': create_curry_style,
        'durant': create_durant_style,
        'kawhi': create_kawhi_style,
        'harden': create_harden_style
    }
else:
    generator = None
    PLAYER_STYLES = {}

# Cache for synthetic profiles
synthetic_profiles_cache = {}

def get_synthetic_profile(player_id: str):
    """Get or generate synthetic profile for a player"""
    if not SYNTHETIC_AVAILABLE:
        raise ValueError("Synthetic profiles not available")
        
    if player_id not in synthetic_profiles_cache:
        if player_id in PLAYER_STYLES:
            player_style = PLAYER_STYLES[player_id]()
            profile = generator.generate_profile(player_style)
            synthetic_profiles_cache[player_id] = profile
        else:
            raise ValueError(f"Unknown player: {player_id}")
    
    return synthetic_profiles_cache[player_id]

def mock_analyze_video(video_path: str) -> Dict:
    """Mock video analysis for testing"""
    return {
        "success": True,
        "message": "Video analyzed successfully (mock)",
        "total_frames": 150,
        "duration": 5.0,
        "phases_detected": ["General", "Set-up", "Loading", "Rising", "Release", "Follow-through"]
    }

def mock_compare_with_player(user_data: Dict, player_id: str) -> Dict:
    """Mock comparison with synthetic player data"""
    import random
    
    # Generate mock similarity scores
    phase_scores = {}
    for phase in ["General", "Set-up", "Loading", "Rising", "Release", "Follow-through"]:
        phase_scores[phase] = random.uniform(0.3, 0.9)
    
    overall_similarity = sum(phase_scores.values()) / len(phase_scores)
    
    # Generate mock recommendations
    recommendations = []
    if overall_similarity < 0.6:
        recommendations.append("Focus on improving your shooting form consistency")
        recommendations.append("Practice the release phase for better accuracy")
    elif overall_similarity < 0.8:
        recommendations.append("Good form! Work on fine-tuning your motion")
    else:
        recommendations.append("Excellent form! Keep up the great work")
    
    return {
        "success": True,
        "player_id": player_id,
        "overall_similarity": overall_similarity,
        "phase_scores": phase_scores,
        "recommendations": recommendations,
        "comparison_metrics": {
            "motion_consistency": random.uniform(0.4, 0.9),
            "release_timing": random.uniform(0.3, 0.8),
            "follow_through": random.uniform(0.5, 0.9)
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Basketball Form Analyzer - Synthetic Profiles Integration",
        "version": "1.0.0",
        "endpoints": {
            "analyze_video": "/analysis/analyze-video",
            "compare_with_player": "/analysis/compare-with-player",
            "player_profiles": "/synthetic/player-profiles",
            "generate_profile": "/synthetic/generate-profile"
        }
    }

@app.post("/analysis/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    """Analyze a basketball shot video"""
    try:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            shutil.copyfileobj(video.file, tmp_file)
            video_path = tmp_file.name
        print("end point")
        # Run analysis pipeline
        if ANALYSIS_AVAILABLE:
            pipeline = BasketballShootingIntegratedPipeline()
            result = pipeline.run_pipeline(video_path)
        else:
            result = mock_analyze_video(video_path)
        
        # Clean up temporary file
        os.unlink(video_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analysis/compare-with-player")
async def compare_with_player(
    video: UploadFile = File(...),
    player_id: str = Form(...),
    player_style: str = Form(...)
):
    """Compare user's shot with a specific NBA player's synthetic data"""
    try:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            shutil.copyfileobj(video.file, tmp_file)
            video_path = tmp_file.name
        
        # Get synthetic profile for the player
        synthetic_profile = get_synthetic_profile(player_id)
        
        # Save synthetic profile to JSON format for comparison pipeline
        synthetic_json_path = generator.export_for_comparison_pipeline(player_id, synthetic_profile, "/tmp")
        
        print(f"Using synthetic profile for player: {player_id}")
        print("video_path", video_path)
        print("synthetic_json_path", synthetic_json_path)
        
        # Analyze user's video using the integrated pipeline
        if ANALYSIS_AVAILABLE:
            pipeline = BasketballShootingIntegratedPipeline()
            user_result = pipeline.run_full_pipeline(video_path, overwrite_mode=True, use_existing_extraction=False)
        else:
            user_result = mock_analyze_video(video_path)
        
        # Use comparison pipeline to compare user video with synthetic profile
        comparison_pipeline = ShootingComparisonPipeline()
        
        # Process user video data (this loads the JSON file created by the integrated pipeline)
        user_processed_data = comparison_pipeline.process_video_data(video_path)
        
        # Process synthetic profile data (this loads the JSON file we just created)
        # The comparison pipeline expects the base name without the _normalized_output.json suffix
        synthetic_base_path = f"/tmp/{player_id.lower()}_synthetic_profile"
        synthetic_processed_data = comparison_pipeline.process_video_data(synthetic_base_path)
        
        if user_processed_data and synthetic_processed_data:
            # Set up the comparison pipeline with the processed data
            comparison_pipeline.video1_data = user_processed_data
            comparison_pipeline.video2_data = synthetic_processed_data
            comparison_pipeline.video1_path = video_path
            comparison_pipeline.video2_path = synthetic_base_path
            
            # Perform the comparison
            comparison_result = comparison_pipeline.perform_comparison()
            
            if comparison_result:
                # Extract results from the comparison pipeline
                interpretation = comparison_result.get('interpretation', {})
                phase_statistics = comparison_result.get('phase_statistics', {})
                
                # Calculate overall similarity from phase analysis
                phase_scores = {}
                overall_similarity = 0.0
                phase_count = 0
                
                # Extract phase scores from the comparison results
                for phase_name in ["General", "Set-up", "Loading", "Rising", "Release", "Follow-through"]:
                    # Look for phase analysis in the comparison results
                    phase_key = f"{phase_name.lower().replace('-', '_')}_analysis"
                    if phase_key in comparison_result:
                        phase_analysis = comparison_result[phase_key]
                        # Calculate similarity score (this would need to be implemented based on the actual analysis)
                        phase_score = 0.75  # Default score - this should be calculated from the analysis
                        phase_scores[phase_name] = phase_score
                        overall_similarity += phase_score
                        phase_count += 1
                    else:
                        phase_scores[phase_name] = 0.7  # Default score for missing phases
                        overall_similarity += 0.7
                        phase_count += 1
                
                if phase_count > 0:
                    overall_similarity /= phase_count
                
                # Generate recommendations based on interpretation
                recommendations = []
                if interpretation:
                    insights = interpretation.get('insights', [])
                    for insight in insights:
                        if isinstance(insight, str):
                            recommendations.append(insight)
                        elif isinstance(insight, dict):
                            recommendations.append(insight.get('message', 'Focus on improving your form'))
                
                # Add default recommendations if none from interpretation
                if not recommendations:
                    if overall_similarity < 0.6:
                        recommendations = [
                            "Focus on improving your shooting form consistency",
                            "Practice the release phase for better accuracy",
                            "Work on your follow-through technique"
                        ]
                    elif overall_similarity < 0.8:
                        recommendations = [
                            "Good form! Work on fine-tuning your motion",
                            "Focus on consistency in your release phase",
                            "Your follow-through looks excellent"
                        ]
                    else:
                        recommendations = [
                            "Excellent form! Keep up the great work",
                            "Your technique is very consistent",
                            "Great job on all phases of your shot"
                        ]
                
                # Format the result for the mobile app
                formatted_result = {
                    "success": True,
                    "player_id": player_id,
                    "overall_similarity": round(overall_similarity, 2),
                    "phase_scores": phase_scores,
                    "recommendations": recommendations,
                    "comparison_metrics": {
                        "motion_consistency": round(overall_similarity * 0.9, 2),
                        "release_timing": round(phase_scores.get("Release", 0.7), 2),
                        "follow_through": round(phase_scores.get("Follow-through", 0.7), 2)
                    },
                    "raw_comparison_data": {
                        "phase_statistics": phase_statistics,
                        "interpretation": interpretation
                    }
                }
            else:
                # Fallback to mock comparison if pipeline comparison fails
                comparison_result = mock_compare_with_player(user_result, player_id)
                formatted_result = comparison_result
        else:
            # Fallback to mock comparison if data processing fails
            comparison_result = mock_compare_with_player(user_result, player_id)
            formatted_result = comparison_result

        # Clean up temporary files
        os.unlink(video_path)
        if os.path.exists(synthetic_json_path):
            os.unlink(synthetic_json_path)
        
        return JSONResponse(content=formatted_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@app.get("/synthetic/player-profiles")
async def get_player_profiles():
    """Get available player profiles"""
    profiles = []
    
    for player_id, player_func in PLAYER_STYLES.items():
        player_style = player_func()
        profiles.append({
            "id": player_id,
            "name": player_style.name,
            "style": player_style.motion_curve,
            "total_frames": player_style.total_frames,
            "noise_level": player_style.noise_level,
            "height_scale": player_style.height_scale,
            "phase_distribution": {
                phase.value: ratio for phase, ratio in player_style.phase_distribution.items()
            }
        })
    
    return JSONResponse(content={"players": profiles})

@app.post("/synthetic/generate-profile")
async def generate_synthetic_profile(player_id: str = Form(...)):
    """Generate synthetic profile for a specific player"""
    try:
        profile = get_synthetic_profile(player_id)
        
        # Convert to JSON-serializable format
        json_profile = []
        for frame in profile:
            json_frame = {
                "frame_index": frame.frame_index,
                "phase": frame.phase.value,
                "timestamp": frame.timestamp,
                "keypoints": {},
                "ball_position": list(frame.ball_position),
                "ball_confidence": frame.ball_confidence
            }
            
            for keypoint_name, keypoint_data in frame.keypoints.items():
                json_frame["keypoints"][keypoint_name] = {
                    "x": keypoint_data.x,
                    "y": keypoint_data.y,
                    "confidence": keypoint_data.confidence
                }
            
            json_profile.append(json_frame)
        
        return JSONResponse(content={
            "player_id": player_id,
            "total_frames": len(json_profile),
            "profile": json_profile
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "synthetic_profiles_available": SYNTHETIC_AVAILABLE,
        "player_count": len(PLAYER_STYLES),
        "analysis_available": ANALYSIS_AVAILABLE
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)