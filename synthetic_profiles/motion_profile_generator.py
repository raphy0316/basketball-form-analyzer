import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

# --- Enum and Data Classes ---
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
    base_positions: Dict[str, Dict[str, Tuple[float, float]]]  # Now normalized coordinates

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

# --- Generator Class ---
class SyntheticProfileGenerator:
    def __init__(self, resolution=(1920, 1080)):
        self.resolution = resolution
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

    def compute_phase_ranges(self, style: PlayerStyle) -> List[Tuple[ShootingPhase, int, int]]:
        """Compute frame ranges for each phase based on distribution"""
        ranges = []
        start = 0
        
        # Ensure phases are in correct order
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

    def normalize_coordinates(self, x, y):
        """Convert pixel coordinates to normalized [0,1] space"""
        return x / self.resolution[0], y / self.resolution[1]

    def denormalize_coordinates(self, norm_x, norm_y):
        """Convert normalized coordinates back to pixel space"""
        return norm_x * self.resolution[0], norm_y * self.resolution[1]

    def get_default_keypoint_positions(self, height_scale=1.0):
        """Generate default normalized keypoint positions based on human proportions"""
        # Base positions in normalized coordinates (0-1)
        # Using human body proportions as reference
        center_x = 0.5
        
        # Vertical positions (normalized by height)
        base_positions = {
            'nose': (center_x, 0.15 * height_scale),
            'left_eye': (center_x - 0.02, 0.14 * height_scale),
            'right_eye': (center_x + 0.02, 0.14 * height_scale),
            'left_ear': (center_x - 0.03, 0.16 * height_scale),
            'right_ear': (center_x + 0.03, 0.16 * height_scale),
            'left_shoulder': (center_x - 0.12, 0.25 * height_scale),
            'right_shoulder': (center_x + 0.12, 0.25 * height_scale),
            'left_elbow': (center_x - 0.18, 0.35 * height_scale),
            'right_elbow': (center_x + 0.18, 0.35 * height_scale),
            'left_wrist': (center_x - 0.20, 0.45 * height_scale),
            'right_wrist': (center_x + 0.20, 0.45 * height_scale),
            'left_hip': (center_x - 0.08, 0.55 * height_scale),
            'right_hip': (center_x + 0.08, 0.55 * height_scale),
            'left_knee': (center_x - 0.08, 0.75 * height_scale),
            'right_knee': (center_x + 0.08, 0.75 * height_scale),
            'left_ankle': (center_x - 0.08, 0.95 * height_scale),
            'right_ankle': (center_x + 0.08, 0.95 * height_scale)
        }
        
        return base_positions

    def interpolate_keypoints(self, start_kps, end_kps, progress, noise_level, height_scale=1.0):
        """Interpolate between keypoint sets with proper noise scaling"""
        kps = {}
        default_positions = self.get_default_keypoint_positions(height_scale)
        
        for kp_name in self.keypoint_names:
            # Use provided positions or fall back to defaults
            start_pos = start_kps.get(kp_name, default_positions[kp_name])
            end_pos = end_kps.get(kp_name, default_positions[kp_name])
            
            # Interpolate in normalized space
            norm_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
            norm_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
            
            # Add noise as percentage of resolution (FIXED)
            noise_x = np.random.normal(0, noise_level * 0.01)  # 1% of normalized space
            noise_y = np.random.normal(0, noise_level * 0.01)
            
            norm_x = np.clip(norm_x + noise_x, 0, 1)
            norm_y = np.clip(norm_y + noise_y, 0, 1)
            
            # Generate realistic confidence based on visibility and clarity
            base_confidence = 0.85
            if kp_name in ['left_eye', 'right_eye', 'nose']:
                base_confidence = 0.90  # Face keypoints usually more reliable
            elif kp_name in ['left_ankle', 'right_ankle']:
                base_confidence = 0.75  # Feet sometimes occluded
            
            confidence = np.clip(
                base_confidence + 0.1 * np.random.random() - noise_level * 0.1,
                0.3, 1.0
            )
            
            kps[kp_name] = KeypointData(norm_x, norm_y, confidence)
        
        return kps

    def generate_ball_trajectory(self, phase, progress, frame_idx, total_frames):
        """Generate realistic ball trajectory based on shooting phase"""
        if phase == ShootingPhase.GENERAL:
            # Ball not visible or far away
            return (0.0, 0.0), 0.1
        
        elif phase in [ShootingPhase.SETUP, ShootingPhase.LOADING]:
            # Ball near shooting hand (right hand)
            ball_x = 0.58 + 0.02 * np.random.random()  # Near right side
            ball_y = 0.45 + 0.05 * np.random.random()  # Mid-body height
            confidence = 0.90 + 0.05 * np.random.random()
            
        elif phase == ShootingPhase.RISING:
            # Ball moving up with hand
            base_x = 0.60
            base_y = 0.40 - 0.15 * progress  # Moving upward
            ball_x = base_x + 0.01 * np.random.random()
            ball_y = base_y + 0.02 * np.random.random()
            confidence = 0.85 + 0.10 * np.random.random()
            
        elif phase == ShootingPhase.RELEASE:
            # Ball leaving hand
            release_progress = progress
            ball_x = 0.61 + 0.10 * release_progress
            ball_y = 0.25 - 0.20 * release_progress
            confidence = 0.80 - 0.20 * release_progress
            
        elif phase == ShootingPhase.FOLLOW_THROUGH:
            # Ball in flight - parabolic trajectory
            flight_progress = progress
            ball_x = 0.71 + 0.25 * flight_progress
            ball_y = 0.05 - 0.30 * flight_progress + 0.20 * (flight_progress ** 2)
            confidence = max(0.30, 0.70 - 0.40 * flight_progress)
            
        else:
            ball_x, ball_y, confidence = 0.0, 0.0, 0.0
        
        return (np.clip(ball_x, 0, 1), np.clip(ball_y, 0, 1)), confidence

    def generate_profile(self, style: PlayerStyle) -> List[FrameData]:
        """Generate complete motion profile for a player"""
        frames = []
        motion_func = MOTION_CURVES.get(style.motion_curve, linear_motion)
        phase_ranges = self.compute_phase_ranges(style)

        for i, (current_phase, start_idx, end_idx) in enumerate(phase_ranges):
            # Determine next phase for interpolation
            if i < len(phase_ranges) - 1:
                next_phase = phase_ranges[i + 1][0]
            else:
                next_phase = current_phase  # Stay at final position
            
            for idx in range(start_idx, end_idx):
                # Calculate progress within current phase
                progress = (idx - start_idx) / (end_idx - start_idx) if end_idx > start_idx else 0
                progress = motion_func(progress)
                
                # Get keypoint positions for current and next phase
                current_positions = style.base_positions.get(current_phase.value.lower().replace('-', '_'), 
                                                           style.base_positions.get('setup', {}))
                next_positions = style.base_positions.get(next_phase.value.lower().replace('-', '_'),
                                                        style.base_positions.get('release', {}))
                
                # Interpolate keypoints
                keypoints = self.interpolate_keypoints(
                    current_positions,
                    next_positions,
                    progress,
                    style.noise_level,
                    style.height_scale
                )
                
                # Generate ball trajectory
                ball_pos, ball_conf = self.generate_ball_trajectory(
                    current_phase, progress, idx, style.total_frames
                )
                
                # Create frame data
                frame = FrameData(
                    frame_index=idx,
                    phase=current_phase,
                    keypoints=keypoints,
                    ball_position=ball_pos,
                    ball_confidence=round(ball_conf, 3),
                    timestamp=round(idx / 30.0, 3)
                )
                frames.append(frame)
        
        return frames

    def export_to_json(self, player_name: str, frames: List[FrameData], output_dir="motion_profiles"):
        """Export frames to JSON in the correct format"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{player_name.lower().replace(' ', '_')}_normalized_output.json")
        
        json_data = {
            "metadata": {
                "player_name": player_name,
                "total_frames": len(frames),
                "fps": 30,
                "resolution": self.resolution,
                "generation_timestamp": datetime.now().isoformat(),
                "data_type": "synthetic_biomechanical_model",
                "coordinate_system": "normalized_0_to_1"
            },
            "frames": []
        }
        
        for f in frames:
            frame_dict = {
                "frame_index": f.frame_index,
                "timestamp": f.timestamp,
                "phase": f.phase.value,
                "normalized_pose": {
                    k: {
                        "x": round(kp.x, 4),
                        "y": round(kp.y, 4),
                        "confidence": round(kp.confidence, 3)
                    } for k, kp in f.keypoints.items()
                },
                "normalized_ball": {
                    "x": round(f.ball_position[0], 4),
                    "y": round(f.ball_position[1], 4),
                    "confidence": round(f.ball_confidence, 3)
                },
                "ball_detected": f.ball_confidence > 0.3
            }
            json_data["frames"].append(frame_dict)
        
        with open(path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✅ Exported: {path}")
        return path

# --- Player Definitions (All 5 NBA Players) ---
def create_lebron_style():
    """LeBron James - Power-based, athletic, consistent"""
    return PlayerStyle(
        name="LeBron James",
        noise_level=1.5,  # Lower noise = more consistent
        height_scale=1.05,  # Taller than average
        total_frames=90,
        motion_curve="power",
        phase_distribution={
            ShootingPhase.GENERAL: 0.17,      # 15 frames
            ShootingPhase.SETUP: 0.11,        # 10 frames  
            ShootingPhase.LOADING: 0.17,      # 15 frames
            ShootingPhase.RISING: 0.28,       # 25 frames
            ShootingPhase.RELEASE: 0.06,      # 5 frames
            ShootingPhase.FOLLOW_THROUGH: 0.22 # 20 frames
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
                'left_hip': (0.42, 0.55), 'right_hip': (0.58, 0.55),  # Lower during loading
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
                'left_elbow': (0.35, 0.25), 'right_elbow': (0.64, 0.12),  # High release
                'left_wrist': (0.38, 0.32), 'right_wrist': (0.62, 0.08),  # Very high
                'left_hip': (0.42, 0.48), 'right_hip': (0.58, 0.48),
                'nose': (0.50, 0.13)
            },
            'follow_through': {
                'left_shoulder': (0.40, 0.20), 'right_shoulder': (0.60, 0.20),
                'left_elbow': (0.35, 0.25), 'right_elbow': (0.64, 0.15),
                'left_wrist': (0.38, 0.32), 'right_wrist': (0.62, 0.12),  # Extended follow-through
                'left_hip': (0.42, 0.48), 'right_hip': (0.58, 0.48),
                'nose': (0.50, 0.13)
            }
        }
    )

def create_durant_style():
    """Kevin Durant - High release, smooth, consistent"""
    return PlayerStyle(
        name="Kevin Durant",
        noise_level=1.0,  # Very consistent
        height_scale=1.15,  # Very tall
        total_frames=85,
        motion_curve="smooth",
        phase_distribution={
            ShootingPhase.GENERAL: 0.14,      # 12 frames
            ShootingPhase.SETUP: 0.12,        # 10 frames
            ShootingPhase.LOADING: 0.15,      # 13 frames
            ShootingPhase.RISING: 0.29,       # 25 frames
            ShootingPhase.RELEASE: 0.06,      # 5 frames
            ShootingPhase.FOLLOW_THROUGH: 0.24 # 20 frames
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
                'left_elbow': (0.36, 0.23), 'right_elbow': (0.64, 0.10),  # Extremely high
                'left_wrist': (0.39, 0.30), 'right_wrist': (0.61, 0.06),  # Highest release
                'left_hip': (0.43, 0.46), 'right_hip': (0.57, 0.46),
                'nose': (0.50, 0.11)
            },
            'follow_through': {
                'left_shoulder': (0.41, 0.19), 'right_shoulder': (0.59, 0.19),
                'left_elbow': (0.36, 0.23), 'right_elbow': (0.64, 0.12),
                'left_wrist': (0.39, 0.30), 'right_wrist': (0.61, 0.08),
                'left_hip': (0.43, 0.46), 'right_hip': (0.57, 0.46),
                'nose': (0.50, 0.11)
            }
        }
    )

def create_curry_style():
    """Stephen Curry - Quick release, incredible consistency"""
    return PlayerStyle(
        name="Stephen Curry",
        noise_level=0.8,  # Extremely consistent
        height_scale=0.95,  # Shorter than average
        total_frames=75,  # Quick shot
        motion_curve="quick",
        phase_distribution={
            ShootingPhase.GENERAL: 0.13,      # 10 frames
            ShootingPhase.SETUP: 0.11,        # 8 frames
            ShootingPhase.LOADING: 0.13,      # 10 frames
            ShootingPhase.RISING: 0.29,       # 22 frames
            ShootingPhase.RELEASE: 0.07,      # 5 frames
            ShootingPhase.FOLLOW_THROUGH: 0.27 # 20 frames
        },
        base_positions={
            'setup': {
                'left_shoulder': (0.42, 0.26), 'right_shoulder': (0.58, 0.26),
                'left_elbow': (0.36, 0.36), 'right_elbow': (0.64, 0.36),
                'left_wrist': (0.40, 0.43), 'right_wrist': (0.60, 0.43),
                'left_hip': (0.44, 0.54), 'right_hip': (0.56, 0.54),
                'nose': (0.50, 0.19)
            },
            'loading': {
                'left_shoulder': (0.42, 0.27), 'right_shoulder': (0.58, 0.27),
                'left_elbow': (0.36, 0.38), 'right_elbow': (0.64, 0.38),
                'left_wrist': (0.40, 0.45), 'right_wrist': (0.60, 0.45),
                'left_hip': (0.44, 0.56), 'right_hip': (0.56, 0.56),
                'nose': (0.50, 0.20)
            },
            'rising': {
                'left_shoulder': (0.42, 0.23), 'right_shoulder': (0.58, 0.23),
                'left_elbow': (0.38, 0.29), 'right_elbow': (0.62, 0.29),
                'left_wrist': (0.40, 0.36), 'right_wrist': (0.60, 0.36),
                'left_hip': (0.44, 0.52), 'right_hip': (0.56, 0.52),
                'nose': (0.50, 0.16)
            },
            'release': {
                'left_shoulder': (0.42, 0.22), 'right_shoulder': (0.58, 0.22),
                'left_elbow': (0.38, 0.26), 'right_elbow': (0.62, 0.13),
                'left_wrist': (0.40, 0.33), 'right_wrist': (0.60, 0.09),  # Perfect release
                'left_hip': (0.44, 0.50), 'right_hip': (0.56, 0.50),
                'nose': (0.50, 0.14)
            },
            'follow_through': {
                'left_shoulder': (0.42, 0.22), 'right_shoulder': (0.58, 0.22),
                'left_elbow': (0.38, 0.26), 'right_elbow': (0.62, 0.15),
                'left_wrist': (0.40, 0.33), 'right_wrist': (0.60, 0.11),
                'left_hip': (0.44, 0.50), 'right_hip': (0.56, 0.50),
                'nose': (0.50, 0.14)
            }
        }
    )

def create_kawhi_style():
    """Kawhi Leonard - Mechanical precision, consistent"""
    return PlayerStyle(
        name="Kawhi Leonard",
        noise_level=0.5,  # Mechanical precision
        height_scale=1.02,
        total_frames=88,
        motion_curve="linear",  # Mechanical consistency
        phase_distribution={
            ShootingPhase.GENERAL: 0.16,      # 14 frames
            ShootingPhase.SETUP: 0.11,        # 10 frames
            ShootingPhase.LOADING: 0.16,      # 14 frames
            ShootingPhase.RISING: 0.27,       # 24 frames
            ShootingPhase.RELEASE: 0.06,      # 5 frames
            ShootingPhase.FOLLOW_THROUGH: 0.24 # 21 frames
        },
        base_positions={
            'setup': {
                'left_shoulder': (0.40, 0.25), 'right_shoulder': (0.60, 0.25),
                'left_elbow': (0.34, 0.35), 'right_elbow': (0.66, 0.35),
                'left_wrist': (0.38, 0.42), 'right_wrist': (0.62, 0.42),
                'left_hip': (0.42, 0.53), 'right_hip': (0.58, 0.53),
                'nose': (0.50, 0.18)
            },
            'loading': {
                'left_shoulder': (0.40, 0.26), 'right_shoulder': (0.60, 0.26),
                'left_elbow': (0.34, 0.37), 'right_elbow': (0.66, 0.37),
                'left_wrist': (0.38, 0.44), 'right_wrist': (0.62, 0.44),
                'left_hip': (0.42, 0.55), 'right_hip': (0.58, 0.55),
                'nose': (0.50, 0.19)
            },
            'rising': {
                'left_shoulder': (0.40, 0.22), 'right_shoulder': (0.60, 0.22),
                'left_elbow': (0.36, 0.28), 'right_elbow': (0.64, 0.28),
                'left_wrist': (0.38, 0.35), 'right_wrist': (0.62, 0.35),
                'left_hip': (0.42, 0.51), 'right_hip': (0.58, 0.51),
                'nose': (0.50, 0.15)
            },
            'release': {
                'left_shoulder': (0.40, 0.21), 'right_shoulder': (0.60, 0.21),
                'left_elbow': (0.36, 0.26), 'right_elbow': (0.64, 0.12),
                'left_wrist': (0.38, 0.33), 'right_wrist': (0.62, 0.08),
                'left_hip': (0.42, 0.49), 'right_hip': (0.58, 0.49),
                'nose': (0.50, 0.13)
            },
            'follow_through': {
                'left_shoulder': (0.40, 0.21), 'right_shoulder': (0.60, 0.21),
                'left_elbow': (0.36, 0.26), 'right_elbow': (0.64, 0.14),
                'left_wrist': (0.38, 0.33), 'right_wrist': (0.62, 0.10),
                'left_hip': (0.42, 0.49), 'right_hip': (0.58, 0.49),
                'nose': (0.50, 0.13)
            }
        }
    )

def create_harden_style():
    """James Harden - Step-back, variable release"""
    return PlayerStyle(
        name="James Harden",
        noise_level=2.0,  # More variable
        height_scale=1.02,
        total_frames=92,
        motion_curve="smooth",
        phase_distribution={
            ShootingPhase.GENERAL: 0.17,      # 16 frames
            ShootingPhase.SETUP: 0.13,        # 12 frames (includes step-back)
            ShootingPhase.LOADING: 0.15,      # 14 frames
            ShootingPhase.RISING: 0.27,       # 25 frames
            ShootingPhase.RELEASE: 0.05,      # 5 frames
            ShootingPhase.FOLLOW_THROUGH: 0.22 # 20 frames
        },
        base_positions={
            'setup': {
                'left_shoulder': (0.39, 0.26), 'right_shoulder': (0.61, 0.26),
                'left_elbow': (0.33, 0.37), 'right_elbow': (0.67, 0.37),
                'left_wrist': (0.37, 0.44), 'right_wrist': (0.63, 0.44),
                'left_hip': (0.41, 0.54), 'right_hip': (0.59, 0.54),  # Step-back position
                'nose': (0.50, 0.19)
            },
            'loading': {
                'left_shoulder': (0.39, 0.27), 'right_shoulder': (0.61, 0.27),
                'left_elbow': (0.33, 0.39), 'right_elbow': (0.67, 0.39),
                'left_wrist': (0.37, 0.46), 'right_wrist': (0.63, 0.46),
                'left_hip': (0.41, 0.56), 'right_hip': (0.59, 0.56),
                'nose': (0.50, 0.20)
            },
            'rising': {
                'left_shoulder': (0.39, 0.23), 'right_shoulder': (0.61, 0.23),
                'left_elbow': (0.35, 0.30), 'right_elbow': (0.65, 0.30),
                'left_wrist': (0.37, 0.37), 'right_wrist': (0.63, 0.37),
                'left_hip': (0.41, 0.52), 'right_hip': (0.59, 0.52),
                'nose': (0.50, 0.16)
            },
            'release': {
                'left_shoulder': (0.39, 0.22), 'right_shoulder': (0.61, 0.22),
                'left_elbow': (0.35, 0.27), 'right_elbow': (0.65, 0.13),
                'left_wrist': (0.37, 0.34), 'right_wrist': (0.63, 0.09),
                'left_hip': (0.41, 0.50), 'right_hip': (0.59, 0.50),
                'nose': (0.50, 0.14)
            },
            'follow_through': {
                'left_shoulder': (0.39, 0.22), 'right_shoulder': (0.61, 0.22),
                'left_elbow': (0.35, 0.27), 'right_elbow': (0.65, 0.15),
                'left_wrist': (0.37, 0.34), 'right_wrist': (0.63, 0.11),
                'left_hip': (0.41, 0.50), 'right_hip': (0.59, 0.50),
                'nose': (0.50, 0.14)
            }
        }
    )

# --- All Players List ---
PLAYERS = [
    create_lebron_style(),
    create_durant_style(),
    create_curry_style(),
    create_kawhi_style(),
    create_harden_style()
]

# --- Analysis Functions ---
def analyze_player_characteristics(generator, players):
    """Analyze and compare player characteristics"""
    print("\n" + "="*60)
    print("PLAYER ANALYSIS SUMMARY")
    print("="*60)
    
    for player in players:
        print(f"\n🏀 {player.name}")
        print(f"   Consistency: {'●'*int(5*(2.5-player.noise_level)/2.5)}{'○'*(5-int(5*(2.5-player.noise_level)/2.5))} ({player.noise_level:.1f})")
        print(f"   Height Scale: {player.height_scale:.2f}x")
        print(f"   Shot Duration: {player.total_frames} frames ({player.total_frames/30:.1f}s)")
        print(f"   Motion Style: {player.motion_curve}")
        
        # Calculate release height from positions
        release_pos = player.base_positions.get('release', {})
        if 'right_wrist' in release_pos:
            release_height = release_pos['right_wrist'][1]
            print(f"   Release Height: {release_height:.3f} (normalized)")
        
        # Show phase timing
        rising_ratio = player.phase_distribution.get(ShootingPhase.RISING, 0)
        release_ratio = player.phase_distribution.get(ShootingPhase.RELEASE, 0)
        print(f"   Rising Phase: {rising_ratio*100:.0f}% | Release: {release_ratio*100:.0f}%")

def compare_players_metrics(generator, players):
    """Generate and compare key metrics across players"""
    print("\n" + "="*60)
    print("COMPARATIVE METRICS")
    print("="*60)
    
    metrics_data = []
    
    for player in players:
        frames = generator.generate_profile(player)
        
        # Calculate metrics
        release_frames = [f for f in frames if f.phase == ShootingPhase.RELEASE]
        rising_frames = [f for f in frames if f.phase == ShootingPhase.RISING]
        
        metrics = {
            'name': player.name,
            'total_frames': len(frames),
            'rising_duration': len(rising_frames),
            'release_duration': len(release_frames),
            'noise_level': player.noise_level,
            'height_scale': player.height_scale
        }
        
        # Calculate release height and consistency
        if release_frames:
            release_heights = []
            for frame in release_frames:
                if 'right_wrist' in frame.keypoints:
                    release_heights.append(frame.keypoints['right_wrist'].y)
            
            if release_heights:
                metrics['avg_release_height'] = np.mean(release_heights)
                metrics['release_consistency'] = np.std(release_heights)
        
        metrics_data.append(metrics)
    
    # Display comparison table
    print(f"{'Player':<15} {'Frames':<7} {'Rising':<7} {'Release':<8} {'Height':<8} {'Consistency':<12}")
    print("-" * 65)
    
    for m in metrics_data:
        consistency_score = f"{1/m['noise_level']:.2f}" if m['noise_level'] > 0 else "N/A"
        release_height = f"{m.get('avg_release_height', 0):.3f}" if 'avg_release_height' in m else "N/A"
        
        print(f"{m['name']:<15} {m['total_frames']:<7} {m['rising_duration']:<7} "
              f"{m['release_duration']:<8} {release_height:<8} {consistency_score:<12}")

# --- Main Execution ---
if __name__ == "__main__":
    print("🏀 Basketball Motion Profile Generator")
    print("=====================================")
    
    # Initialize generator
    generator = SyntheticProfileGenerator(resolution=(1920, 1080))
    
    # Show player analysis
    analyze_player_characteristics(generator, PLAYERS)
    
    # Generate profiles for all players
    print(f"\n📊 Generating motion profiles for {len(PLAYERS)} players...")
    exported_files = []
    
    for player in PLAYERS:
        print(f"\nGenerating profile for {player.name}...")
        frames = generator.generate_profile(player)
        
        # Export to JSON
        output_path = generator.export_to_json(player.name, frames)
        exported_files.append(output_path)
        
        # Show basic stats
        phases = set(f.phase.value for f in frames)
        print(f"  ✓ Generated {len(frames)} frames")
        print(f"  ✓ Phases: {', '.join(sorted(phases))}")
        
        # Show sample keypoint data
        if frames:
            sample_frame = frames[len(frames)//2]  # Middle frame
            if 'right_wrist' in sample_frame.keypoints:
                rw = sample_frame.keypoints['right_wrist']
                print(f"  ✓ Sample right wrist: ({rw.x:.3f}, {rw.y:.3f}) conf={rw.confidence:.2f}")
    
    # Generate comparative analysis
    compare_players_metrics(generator, PLAYERS)
    
    print(f"\n✅ Successfully exported {len(exported_files)} profile files:")
    for file_path in exported_files:
        print(f"   📁 {file_path}")
    
    print(f"\n🎯 All profiles use normalized coordinates [0.0-1.0]")
    print(f"🎯 Compatible with basketball-form-analyzer pipeline")
    print(f"🎯 Ready for phase segmentation and analysis!")
    
    # Show usage example
    print(f"\n" + "="*50)
    print("USAGE EXAMPLE:")
    print("="*50)
    print("""
# Load a specific player profile:
with open('motion_profiles/stephen_curry_normalized_output.json') as f:
    curry_data = json.load(f)

# Access frame data:
frame_0 = curry_data['frames'][0]
right_wrist = frame_0['normalized_pose']['right_wrist']
print(f"Wrist position: ({right_wrist['x']}, {right_wrist['y']})")

# Filter by shooting phase:
release_frames = [f for f in curry_data['frames'] 
                 if f['phase'] == 'Release']
""")
    
    print("\n🚀 Generation complete!")