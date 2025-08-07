#!/usr/bin/env python3
"""
Basketball Motion Profiles Generator for Mobile App Integration

This module generates realistic basketball shooting motion profiles for NBA players
that can be used to compare with user videos in the mobile app.
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# --- Enums and Data Classes ---
class ShootingPhase(Enum):
    GENERAL = "General"
    SETUP = "Set-up"
    LOADING = "Loading"
    RISING = "Rising"
    RELEASE = "Release"
    FOLLOW_THROUGH = "Follow-through"

@dataclass
class KeypointData:
    x: float
    y: float
    confidence: float

@dataclass
class FrameData:
    frame_index: int
    phase: ShootingPhase
    keypoints: Dict[str, KeypointData]
    ball_position: Tuple[float, float]
    ball_confidence: float
    timestamp: float

@dataclass
class PlayerStyle:
    name: str
    noise_level: float
    height_scale: float
    total_frames: int
    motion_curve: str
    phase_distribution: Dict[ShootingPhase, float]
    base_positions: Dict[str, Dict[str, Tuple[float, float]]]

# --- Motion Curve Functions ---
def ease_in_out(progress):
    """Smooth S-curve motion"""
    return 0.5 * (1 - np.cos(np.pi * progress))

def fast_accel(progress):
    """Quick acceleration curve"""
    return progress ** 0.7

def linear_motion(progress):
    """Linear motion"""
    return progress

def power_motion(progress):
    """Power-based motion (for LeBron)"""
    return progress ** 0.8

MOTION_CURVES = {
    "smooth": ease_in_out,
    "quick": fast_accel,
    "linear": linear_motion,
    "power": power_motion
}

class SyntheticProfileGenerator:
    def __init__(self, resolution=(1920, 1080)):
        self.resolution = resolution
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

    def normalize_coordinates(self, x, y):
        """Normalize coordinates to 0-1 range"""
        return x / self.resolution[0], y / self.resolution[1]

    def denormalize_coordinates(self, norm_x, norm_y):
        """Convert normalized coordinates back to pixel coordinates"""
        return norm_x * self.resolution[0], norm_y * self.resolution[1]

    def get_default_keypoint_positions(self, height_scale=1.0):
        """Get default keypoint positions for a standing pose"""
        center_x = 0.5
        base_y = 0.5
        
        positions = {}
        positions['nose'] = (center_x, base_y - 0.32 * height_scale)
        positions['left_eye'] = (center_x - 0.02, base_y - 0.33 * height_scale)
        positions['right_eye'] = (center_x + 0.02, base_y - 0.33 * height_scale)
        positions['left_ear'] = (center_x - 0.04, base_y - 0.31 * height_scale)
        positions['right_ear'] = (center_x + 0.04, base_y - 0.31 * height_scale)
        
        positions['left_shoulder'] = (center_x - 0.09, base_y - 0.25 * height_scale)
        positions['right_shoulder'] = (center_x + 0.09, base_y - 0.25 * height_scale)
        positions['left_elbow'] = (center_x - 0.16, base_y - 0.15 * height_scale)
        positions['right_elbow'] = (center_x + 0.16, base_y - 0.15 * height_scale)
        positions['left_wrist'] = (center_x - 0.18, base_y - 0.08 * height_scale)
        positions['right_wrist'] = (center_x + 0.18, base_y - 0.08 * height_scale)
        
        positions['left_hip'] = (center_x - 0.08, base_y)
        positions['right_hip'] = (center_x + 0.08, base_y)
        positions['left_knee'] = (center_x - 0.06, base_y + 0.25 * height_scale)
        positions['right_knee'] = (center_x + 0.06, base_y + 0.25 * height_scale)
        positions['left_ankle'] = (center_x - 0.05, base_y + 0.5 * height_scale)
        positions['right_ankle'] = (center_x + 0.05, base_y + 0.5 * height_scale)
        
        return positions

    def interpolate_keypoints(self, start_kps, end_kps, progress, noise_level, height_scale=1.0):
        """Interpolate between keypoint positions with noise"""
        interpolated = {}
        
        for keypoint in self.keypoint_names:
            if keypoint in start_kps and keypoint in end_kps:
                start_x, start_y = start_kps[keypoint]
                end_x, end_y = end_kps[keypoint]
                
                # Apply motion curve
                curve_progress = MOTION_CURVES.get("smooth", linear_motion)(progress)
                
                # Interpolate
                x = start_x + (end_x - start_x) * curve_progress
                y = start_y + (end_y - start_y) * curve_progress
                
                # Add noise
                noise_x = np.random.normal(0, noise_level * 0.01)
                noise_y = np.random.normal(0, noise_level * 0.01)
                
                x = max(0, min(1, x + noise_x))
                y = max(0, min(1, y + noise_y))
                
                # Confidence decreases with noise
                confidence = max(0.7, 1.0 - noise_level * 0.1)
                
                interpolated[keypoint] = KeypointData(x, y, confidence)
        
        return interpolated

    def generate_ball_trajectory(self, phase, progress, frame_idx, total_frames):
        """Generate realistic ball trajectory"""
        if phase == ShootingPhase.GENERAL:
            # Ball at waist level
            return (0.5, 0.42), 0.9
        elif phase == ShootingPhase.SETUP:
            # Ball moving to shooting position
            ball_y = 0.42 - progress * 0.05
            return (0.5, ball_y), 0.9
        elif phase == ShootingPhase.LOADING:
            # Ball slightly lower for loading
            ball_y = 0.37 - progress * 0.02
            return (0.5, ball_y), 0.9
        elif phase == ShootingPhase.RISING:
            # Ball rising with the shot
            ball_y = 0.35 - progress * 0.25
            ball_x = 0.5 + progress * 0.02  # Slight forward movement
            return (ball_x, ball_y), 0.9
        elif phase == ShootingPhase.RELEASE:
            # Ball at release point
            ball_y = 0.1
            ball_x = 0.52
            return (ball_x, ball_y), 0.8
        elif phase == ShootingPhase.FOLLOW_THROUGH:
            # Ball continuing trajectory
            ball_y = 0.05 - progress * 0.03
            ball_x = 0.52 + progress * 0.01
            return (ball_x, ball_y), 0.7
        else:
            return (0.5, 0.42), 0.9

    def generate_profile(self, style: PlayerStyle) -> List[FrameData]:
        """Generate complete motion profile for a player style"""
        frames = []
        
        # Compute phase ranges
        phase_ranges = self.compute_phase_ranges(style)
        
        frame_idx = 0
        for phase, start_frame, end_frame in phase_ranges:
            phase_frames = end_frame - start_frame
            
            for i in range(phase_frames):
                progress = i / max(1, phase_frames - 1)
                
                # Get start and end positions for this phase
                phase_name = phase.value.lower().replace('-', '_')
                if phase_name in style.base_positions:
                    start_positions = style.base_positions[phase_name]
                    # For simplicity, use same positions with slight variation
                    end_positions = start_positions
                else:
                    # Use default positions
                    start_positions = self.get_default_keypoint_positions(style.height_scale)
                    end_positions = start_positions
                
                # Interpolate keypoints
                keypoints = self.interpolate_keypoints(
                    start_positions, end_positions, progress, 
                    style.noise_level, style.height_scale
                )
                
                # Generate ball trajectory
                ball_pos, ball_conf = self.generate_ball_trajectory(
                    phase, progress, frame_idx, style.total_frames
                )
                
                # Create frame data
                frame_data = FrameData(
                    frame_index=frame_idx,
                    phase=phase,
                    keypoints=keypoints,
                    ball_position=ball_pos,
                    ball_confidence=ball_conf,
                    timestamp=frame_idx / 30.0  # Assuming 30 FPS
                )
                
                frames.append(frame_data)
                frame_idx += 1
        
        return frames

    def compute_phase_ranges(self, style: PlayerStyle) -> List[Tuple[ShootingPhase, int, int]]:
        """Compute frame ranges for each phase based on distribution"""
        ranges = []
        start = 0
        
        phase_order = [
            ShootingPhase.GENERAL,
            ShootingPhase.SETUP,
            ShootingPhase.LOADING,
            ShootingPhase.RISING,
            ShootingPhase.RELEASE,
            ShootingPhase.FOLLOW_THROUGH
        ]
        
        for phase in phase_order:
            if phase in style.phase_distribution:
                ratio = style.phase_distribution[phase]
                end = start + int(ratio * style.total_frames)
                ranges.append((phase, start, end))
                start = end
        
        return ranges

    def export_to_json(self, player_name: str, frames: List[FrameData], output_dir="motion_profiles"):
        """Export motion profile to JSON format"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert frames to JSON-serializable format
        json_frames = []
        for frame in frames:
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
            
            json_frames.append(json_frame)
        
        # Create metadata
        metadata = {
            "player_name": player_name,
            "total_frames": len(frames),
            "fps": 30,
            "resolution": self.resolution,
            "generated_at": datetime.now().isoformat()
        }
        
        # Create final JSON structure
        output_data = {
            "metadata": metadata,
            "frames": json_frames
        }
        
        # Save to file
        output_file = os.path.join(output_dir, f"{player_name.lower()}_profile.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Exported: {output_file}")
        return output_file

# --- Player Style Definitions ---
def create_lebron_style():
    """LeBron James - Power-based, athletic, consistent"""
    return PlayerStyle(
        name="LeBron James",
        noise_level=1.5,
        height_scale=1.05,
        total_frames=90,
        motion_curve="power",
        phase_distribution={
            ShootingPhase.GENERAL: 0.17,
            ShootingPhase.SETUP: 0.11,
            ShootingPhase.LOADING: 0.17,
            ShootingPhase.RISING: 0.28,
            ShootingPhase.RELEASE: 0.06,
            ShootingPhase.FOLLOW_THROUGH: 0.22
        },
        base_positions={
            'setup': {
                'left_shoulder': (0.40, 0.25), 'right_shoulder': (0.60, 0.25),
                'left_elbow': (0.34, 0.35), 'right_elbow': (0.66, 0.35),
                'left_wrist': (0.38, 0.42), 'right_wrist': (0.62, 0.42),
                'left_hip': (0.42, 0.52), 'right_hip': (0.58, 0.52),
                'nose': (0.50, 0.18)
            },
            'loading': {
                'left_shoulder': (0.40, 0.27), 'right_shoulder': (0.60, 0.27),
                'left_elbow': (0.34, 0.38), 'right_elbow': (0.66, 0.38),
                'left_wrist': (0.38, 0.45), 'right_wrist': (0.62, 0.45),
                'left_hip': (0.42, 0.55), 'right_hip': (0.58, 0.55),
                'nose': (0.50, 0.20)
            },
            'rising': {
                'left_shoulder': (0.40, 0.22), 'right_shoulder': (0.60, 0.22),
                'left_elbow': (0.35, 0.28), 'right_elbow': (0.65, 0.28),
                'left_wrist': (0.38, 0.35), 'right_wrist': (0.62, 0.35),
                'left_hip': (0.42, 0.50), 'right_hip': (0.58, 0.50),
                'nose': (0.50, 0.15)
            },
            'release': {
                'left_shoulder': (0.40, 0.20), 'right_shoulder': (0.60, 0.20),
                'left_elbow': (0.35, 0.25), 'right_elbow': (0.64, 0.12),
                'left_wrist': (0.38, 0.32), 'right_wrist': (0.62, 0.08),
                'left_hip': (0.42, 0.48), 'right_hip': (0.58, 0.48),
                'nose': (0.50, 0.13)
            },
            'follow_through': {
                'left_shoulder': (0.40, 0.20), 'right_shoulder': (0.60, 0.20),
                'left_elbow': (0.35, 0.25), 'right_elbow': (0.64, 0.15),
                'left_wrist': (0.38, 0.32), 'right_wrist': (0.62, 0.12),
                'left_hip': (0.42, 0.48), 'right_hip': (0.58, 0.48),
                'nose': (0.50, 0.13)
            }
        }
    )

def create_curry_style():
    """Stephen Curry - Quick release, smooth motion flow"""
    return PlayerStyle(
        name="Stephen Curry",
        noise_level=2.0,
        height_scale=0.95,
        total_frames=85,
        motion_curve="quick",
        phase_distribution={
            ShootingPhase.GENERAL: 0.12,
            ShootingPhase.SETUP: 0.10,
            ShootingPhase.LOADING: 0.15,
            ShootingPhase.RISING: 0.30,
            ShootingPhase.RELEASE: 0.08,
            ShootingPhase.FOLLOW_THROUGH: 0.25
        },
        base_positions={
            'setup': {
                'left_shoulder': (0.41, 0.25), 'right_shoulder': (0.59, 0.25),
                'left_elbow': (0.35, 0.33), 'right_elbow': (0.65, 0.33),
                'left_wrist': (0.39, 0.40), 'right_wrist': (0.61, 0.40),
                'left_hip': (0.43, 0.50), 'right_hip': (0.57, 0.50),
                'nose': (0.50, 0.19)
            },
            'loading': {
                'left_shoulder': (0.41, 0.26), 'right_shoulder': (0.59, 0.26),
                'left_elbow': (0.35, 0.35), 'right_elbow': (0.65, 0.35),
                'left_wrist': (0.39, 0.42), 'right_wrist': (0.61, 0.42),
                'left_hip': (0.43, 0.52), 'right_hip': (0.57, 0.52),
                'nose': (0.50, 0.20)
            },
            'rising': {
                'left_shoulder': (0.41, 0.21), 'right_shoulder': (0.59, 0.21),
                'left_elbow': (0.36, 0.27), 'right_elbow': (0.64, 0.27),
                'left_wrist': (0.39, 0.34), 'right_wrist': (0.61, 0.34),
                'left_hip': (0.43, 0.49), 'right_hip': (0.57, 0.49),
                'nose': (0.50, 0.14)
            },
            'release': {
                'left_shoulder': (0.41, 0.19), 'right_shoulder': (0.59, 0.19),
                'left_elbow': (0.36, 0.24), 'right_elbow': (0.64, 0.11),
                'left_wrist': (0.39, 0.31), 'right_wrist': (0.61, 0.07),
                'left_hip': (0.43, 0.47), 'right_hip': (0.57, 0.47),
                'nose': (0.50, 0.12)
            },
            'follow_through': {
                'left_shoulder': (0.41, 0.19), 'right_shoulder': (0.59, 0.19),
                'left_elbow': (0.36, 0.24), 'right_elbow': (0.64, 0.13),
                'left_wrist': (0.39, 0.31), 'right_wrist': (0.61, 0.10),
                'left_hip': (0.43, 0.47), 'right_hip': (0.57, 0.47),
                'nose': (0.50, 0.12)
            }
        }
    )

def create_durant_style():
    """Kevin Durant - High release, smooth, consistent"""
    return PlayerStyle(
        name="Kevin Durant",
        noise_level=1.0,
        height_scale=1.15,
        total_frames=85,
        motion_curve="smooth",
        phase_distribution={
            ShootingPhase.GENERAL: 0.14,
            ShootingPhase.SETUP: 0.12,
            ShootingPhase.LOADING: 0.15,
            ShootingPhase.RISING: 0.29,
            ShootingPhase.RELEASE: 0.06,
            ShootingPhase.FOLLOW_THROUGH: 0.24
        },
        base_positions={
            'setup': {
                'left_shoulder': (0.41, 0.23), 'right_shoulder': (0.59, 0.23),
                'left_elbow': (0.35, 0.32), 'right_elbow': (0.65, 0.32),
                'left_wrist': (0.39, 0.39), 'right_wrist': (0.61, 0.39),
                'left_hip': (0.43, 0.50), 'right_hip': (0.57, 0.50),
                'nose': (0.50, 0.16)
            },
            'loading': {
                'left_shoulder': (0.41, 0.24), 'right_shoulder': (0.59, 0.24),
                'left_elbow': (0.35, 0.34), 'right_elbow': (0.65, 0.34),
                'left_wrist': (0.39, 0.41), 'right_wrist': (0.61, 0.41),
                'left_hip': (0.43, 0.52), 'right_hip': (0.57, 0.52),
                'nose': (0.50, 0.17)
            },
            'rising': {
                'left_shoulder': (0.41, 0.20), 'right_shoulder': (0.59, 0.20),
                'left_elbow': (0.36, 0.26), 'right_elbow': (0.64, 0.26),
                'left_wrist': (0.39, 0.33), 'right_wrist': (0.61, 0.33),
                'left_hip': (0.43, 0.48), 'right_hip': (0.57, 0.48),
                'nose': (0.50, 0.13)
            },
            'release': {
                'left_shoulder': (0.41, 0.19), 'right_shoulder': (0.59, 0.19),
                'left_elbow': (0.36, 0.23), 'right_elbow': (0.64, 0.10),
                'left_wrist': (0.39, 0.30), 'right_wrist': (0.61, 0.06),
                'left_hip': (0.43, 0.46), 'right_hip': (0.57, 0.46),
                'nose': (0.50, 0.11)
            },
            'follow_through': {
                'left_shoulder': (0.41, 0.19), 'right_shoulder': (0.59, 0.19),
                'left_elbow': (0.36, 0.23), 'right_elbow': (0.64, 0.12),
                'left_wrist': (0.39, 0.30), 'right_wrist': (0.61, 0.09),
                'left_hip': (0.43, 0.46), 'right_hip': (0.57, 0.46),
                'nose': (0.50, 0.11)
            }
        }
    )

def create_kawhi_style():
    """Kawhi Leonard - Controlled, deliberate motion"""
    return PlayerStyle(
        name="Kawhi Leonard",
        noise_level=1.8,
        height_scale=1.02,
        total_frames=88,
        motion_curve="linear",
        phase_distribution={
            ShootingPhase.GENERAL: 0.15,
            ShootingPhase.SETUP: 0.12,
            ShootingPhase.LOADING: 0.18,
            ShootingPhase.RISING: 0.25,
            ShootingPhase.RELEASE: 0.08,
            ShootingPhase.FOLLOW_THROUGH: 0.22
        },
        base_positions={
            'setup': {
                'left_shoulder': (0.40, 0.24), 'right_shoulder': (0.60, 0.24),
                'left_elbow': (0.34, 0.34), 'right_elbow': (0.66, 0.34),
                'left_wrist': (0.38, 0.41), 'right_wrist': (0.62, 0.41),
                'left_hip': (0.42, 0.51), 'right_hip': (0.58, 0.51),
                'nose': (0.50, 0.17)
            },
            'loading': {
                'left_shoulder': (0.40, 0.25), 'right_shoulder': (0.60, 0.25),
                'left_elbow': (0.34, 0.36), 'right_elbow': (0.66, 0.36),
                'left_wrist': (0.38, 0.43), 'right_wrist': (0.62, 0.43),
                'left_hip': (0.42, 0.53), 'right_hip': (0.58, 0.53),
                'nose': (0.50, 0.18)
            },
            'rising': {
                'left_shoulder': (0.40, 0.21), 'right_shoulder': (0.60, 0.21),
                'left_elbow': (0.35, 0.28), 'right_elbow': (0.65, 0.28),
                'left_wrist': (0.38, 0.35), 'right_wrist': (0.62, 0.35),
                'left_hip': (0.42, 0.50), 'right_hip': (0.58, 0.50),
                'nose': (0.50, 0.14)
            },
            'release': {
                'left_shoulder': (0.40, 0.20), 'right_shoulder': (0.60, 0.20),
                'left_elbow': (0.35, 0.25), 'right_elbow': (0.64, 0.13),
                'left_wrist': (0.38, 0.32), 'right_wrist': (0.62, 0.09),
                'left_hip': (0.42, 0.48), 'right_hip': (0.58, 0.48),
                'nose': (0.50, 0.12)
            },
            'follow_through': {
                'left_shoulder': (0.40, 0.20), 'right_shoulder': (0.60, 0.20),
                'left_elbow': (0.35, 0.25), 'right_elbow': (0.64, 0.15),
                'left_wrist': (0.38, 0.32), 'right_wrist': (0.62, 0.11),
                'left_hip': (0.42, 0.48), 'right_hip': (0.58, 0.48),
                'nose': (0.50, 0.12)
            }
        }
    )

def create_harden_style():
    """James Harden - Step-back specialist, unique rhythm"""
    return PlayerStyle(
        name="James Harden",
        noise_level=2.2,
        height_scale=1.03,
        total_frames=87,
        motion_curve="smooth",
        phase_distribution={
            ShootingPhase.GENERAL: 0.13,
            ShootingPhase.SETUP: 0.14,
            ShootingPhase.LOADING: 0.16,
            ShootingPhase.RISING: 0.27,
            ShootingPhase.RELEASE: 0.07,
            ShootingPhase.FOLLOW_THROUGH: 0.23
        },
        base_positions={
            'setup': {
                'left_shoulder': (0.40, 0.24), 'right_shoulder': (0.60, 0.24),
                'left_elbow': (0.34, 0.33), 'right_elbow': (0.66, 0.33),
                'left_wrist': (0.38, 0.40), 'right_wrist': (0.62, 0.40),
                'left_hip': (0.42, 0.50), 'right_hip': (0.58, 0.50),
                'nose': (0.50, 0.17)
            },
            'loading': {
                'left_shoulder': (0.40, 0.25), 'right_shoulder': (0.60, 0.25),
                'left_elbow': (0.34, 0.35), 'right_elbow': (0.66, 0.35),
                'left_wrist': (0.38, 0.42), 'right_wrist': (0.62, 0.42),
                'left_hip': (0.42, 0.52), 'right_hip': (0.58, 0.52),
                'nose': (0.50, 0.18)
            },
            'rising': {
                'left_shoulder': (0.40, 0.21), 'right_shoulder': (0.60, 0.21),
                'left_elbow': (0.35, 0.27), 'right_elbow': (0.65, 0.27),
                'left_wrist': (0.38, 0.34), 'right_wrist': (0.62, 0.34),
                'left_hip': (0.42, 0.49), 'right_hip': (0.58, 0.49),
                'nose': (0.50, 0.14)
            },
            'release': {
                'left_shoulder': (0.40, 0.20), 'right_shoulder': (0.60, 0.20),
                'left_elbow': (0.35, 0.24), 'right_elbow': (0.64, 0.12),
                'left_wrist': (0.38, 0.31), 'right_wrist': (0.62, 0.08),
                'left_hip': (0.42, 0.47), 'right_hip': (0.58, 0.47),
                'nose': (0.50, 0.12)
            },
            'follow_through': {
                'left_shoulder': (0.40, 0.20), 'right_shoulder': (0.60, 0.20),
                'left_elbow': (0.35, 0.24), 'right_elbow': (0.64, 0.14),
                'left_wrist': (0.38, 0.31), 'right_wrist': (0.62, 0.10),
                'left_hip': (0.42, 0.47), 'right_hip': (0.58, 0.47),
                'nose': (0.50, 0.12)
            }
        }
    )

# --- Analysis Functions ---
def analyze_player_characteristics(generator, players):
    """Analyze characteristics of multiple players"""
    analysis = {}
    
    for player in players:
        profile = generator.generate_profile(player)
        
        # Calculate metrics
        total_frames = len(profile)
        phases = {}
        for frame in profile:
            phase = frame.phase.value
            phases[phase] = phases.get(phase, 0) + 1
        
        # Calculate phase percentages
        phase_percentages = {phase: (count/total_frames)*100 for phase, count in phases.items()}
        
        analysis[player] = {
            'total_frames': total_frames,
            'duration_seconds': total_frames / 30.0,
            'phase_distribution': phase_percentages,
            'noise_level': player.noise_level,
            'height_scale': player.height_scale,
            'motion_curve': player.motion_curve
        }
    
    return analysis

def compare_players_metrics(generator, players):
    """Compare metrics between players"""
    analysis = analyze_player_characteristics(generator, players)
    
    comparison = {
        'fastest_release': min(players, key=lambda p: analysis[p]['duration_seconds']),
        'most_consistent': min(players, key=lambda p: analysis[p]['noise_level']),
        'tallest': max(players, key=lambda p: analysis[p]['height_scale']),
        'longest_follow_through': max(players, key=lambda p: analysis[p]['phase_distribution'].get('Follow-through', 0))
    }
    
    return comparison

if __name__ == "__main__":
    # Test the generator
    generator = SyntheticProfileGenerator()
    
    # Generate profiles for all players
    players = [
        create_lebron_style(),
        create_curry_style(),
        create_durant_style(),
        create_kawhi_style(),
        create_harden_style()
    ]
    
    for player in players:
        profile = generator.generate_profile(player)
        generator.export_to_json(player.name, profile)
        print(f"Generated {len(profile)} frames for {player.name}")
