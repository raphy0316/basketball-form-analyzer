"""
Basketball Motion Profiles Generator

A synthetic data generation toolkit for creating realistic basketball shooting motion profiles
for biomechanical analysis and form comparison.
"""

__version__ = "1.0.0"
__author__ = "Basketball Form Analyzer Team"

# Import main classes for easy access
from .generator import (
    SyntheticProfileGenerator,
    ShootingPhase,
    PlayerStyle,
    FrameData,
    KeypointData,
    PLAYERS,
    MOTION_CURVES
)

from .players import (
    create_lebron_style,
    create_durant_style,
    create_curry_style,
    create_kawhi_style,
    create_harden_style
)

from .analysis import (
    analyze_player_characteristics,
    compare_players_metrics,
    load_motion_profile,
    extract_phase_data,
    calculate_shooting_metrics
)

__all__ = [
    # Core classes
    'SyntheticProfileGenerator',
    'ShootingPhase',
    'PlayerStyle',
    'FrameData',
    'KeypointData',
    
    # Constants
    'PLAYERS',
    'MOTION_CURVES',
    
    # Player creation functions
    'create_lebron_style',
    'create_durant_style',
    'create_curry_style',
    'create_kawhi_style',
    'create_harden_style',
    
    # Analysis functions
    'analyze_player_characteristics',
    'compare_players_metrics',
    'load_motion_profile',
    'extract_phase_data',
    'calculate_shooting_metrics',
]

# Package metadata
SUPPORTED_FORMATS = ['json']
DEFAULT_RESOLUTION = (1920, 1080)
DEFAULT_FPS = 30