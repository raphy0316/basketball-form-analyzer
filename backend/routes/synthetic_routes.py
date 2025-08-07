#!/usr/bin/env python3
"""
Synthetic Profiles Routes for Basketball Form Analyzer

This module provides API endpoints for synthetic motion profiles integration.
"""

import os
import sys
import json
import tempfile
import shutil
from typing import Dict, List, Optional
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

# Import synthetic profiles
from synthetic_profiles.motion_profile_generator import (
    SyntheticProfileGenerator,
    create_lebron_style,
    create_curry_style,
    create_durant_style,
    create_kawhi_style,
    create_harden_style
)

# Import analysis components (if available)
try:
    from basketball_shooting_integrated_pipeline import BasketballShootingIntegratedPipeline
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    print("Warning: Basketball analysis pipeline not available. Using mock analysis.")

# Create router
synthetic_router = APIRouter(prefix="/synthetic", tags=["Synthetic Profiles"])

# Initialize components
generator = SyntheticProfileGenerator()

# Player styles mapping
PLAYER_STYLES = {
    'lebron': create_lebron_style,
    'curry': create_curry_style,
    'durant': create_durant_style,
    'kawhi': create_kawhi_style,
    'harden': create_harden_style
}

# Cache for synthetic profiles
synthetic_profiles_cache = {}

def get_synthetic_profile(player_id: str):
    """Get or generate synthetic profile for a player"""
    if player_id not in synthetic_profiles_cache:
        if player_id in PLAYER_STYLES:
            player_style = PLAYER_STYLES[player_id]()
            profile = generator.generate_profile(player_style)
            synthetic_profiles_cache[player_id] = profile
        else:
            raise ValueError(f"Unknown player: {player_id}")
    
    return synthetic_profiles_cache[player_id]

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

@synthetic_router.get("/player-profiles")
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

@synthetic_router.post("/generate-profile")
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

@synthetic_router.post("/compare-with-player")
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
        
        # Analyze user's video
        if ANALYSIS_AVAILABLE:
            pipeline = BasketballShootingIntegratedPipeline()
            user_result = pipeline.run_pipeline(video_path)
        else:
            user_result = {"success": True, "message": "Video analyzed successfully (mock)"}
        
        # Compare with synthetic player data
        comparison_result = mock_compare_with_player(user_result, player_id)
        
        # Clean up temporary file
        os.unlink(video_path)
        
        return JSONResponse(content=comparison_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@synthetic_router.get("/health")
async def synthetic_health_check():
    """Health check for synthetic profiles"""
    return {
        "status": "healthy",
        "synthetic_profiles_available": len(PLAYER_STYLES),
        "analysis_available": ANALYSIS_AVAILABLE,
        "cached_profiles": len(synthetic_profiles_cache)
    }
