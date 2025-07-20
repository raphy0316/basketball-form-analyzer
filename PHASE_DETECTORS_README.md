# Phase Detection Module

This module provides multiple strategies for basketball shooting phase detection, fully integrated with the main analysis pipeline.

## 📁 Structure

```
phase_detection/
├── __init__.py                    # Module initialization
├── base_phase_detector.py         # Base abstract class
├── ball_based_phase_detector.py   # Ball size based
├── torso_based_phase_detector.py  # Body size based
├── hybrid_fps_phase_detector.py   # Hybrid (FPS-aware)
```

## 🎯 Available Detectors

### 1. BallBasedPhaseDetector (Ball size based)
- **Description**: Uses the size and position of the ball to determine phase transitions.
- **Pros**: Intuitive, adapts to real-time ball movement.
- **Cons**: Fails if ball detection is unstable or missing.
- **Best for**: Videos with reliable ball detection.

### 2. TorsoBasedPhaseDetector (Body size based)
- **Description**: Uses torso length and body keypoints for phase transitions.
- **Pros**: Robust to resolution/camera distance, works even if ball is not detected.
- **Cons**: Sensitive to player body shape, requires accurate pose estimation.
- **Best for**: Diverse players, variable camera setups.

### 3. HybridFPSPhaseDetector (Hybrid, FPS-aware)
- **Description**: Combines ball and body features, adapts thresholds to video FPS and torso length.
- **Pros**: Most robust and accurate, handles edge cases, recommended for most use cases.
- **Cons**: Slightly more complex and computationally intensive.
- **Best for**: All scenarios, especially when accuracy is critical.

## 🚀 Usage

### Basic Usage

```python
from phase_detection import (
    BallBasedPhaseDetector,
    TorsoBasedPhaseDetector,
    HybridFPSPhaseDetector
)

# Initialize Detector
ball_detector = BallBasedPhaseDetector()
torso_detector = TorsoBasedPhaseDetector()
hybrid_detector = HybridFPSPhaseDetector()

# Check Phase Transition
next_phase = detector.check_phase_transition(
    current_phase="General",
    frame_idx=0,
    pose_data=pose_data,
    ball_data=ball_data
)
```

### Using with Analyzer

```python
from basketball_shooting_analyzer import BasketballShootingAnalyzer

analyzer = BasketballShootingAnalyzer()

# Select Detector Type
analyzer.segment_shooting_phases(detector_type="ball")      # Ball size based
analyzer.segment_shooting_phases(detector_type="torso")    # Body size based
analyzer.segment_shooting_phases(detector_type="hybrid")   # Hybrid (recommended)
```

### Direct Detector Change

```python
# Use specific detector instance
detector = HybridFPSPhaseDetector()
analyzer.current_detector = detector
```

## 📊 Performance Comparison

| Detector      | Accuracy | Robustness | Complexity | Ball Required | Resolution Independent |
|--------------|----------|------------|------------|---------------|-----------------------|
| Ball-based   |   ★★★★   |   ★★       |   ★★★★     |   Yes         |   No                  |
| Torso-based  |   ★★★★   |   ★★★★     |   ★★       |   No          |   Yes                 |
| Hybrid (FPS) |   ★★★★★  |   ★★★★★    |   ★★★      |   No          |   Yes                 |

## 🔧 Customization

### Threshold Adjustment

```python
detector = HybridFPSPhaseDetector(
    min_phase_duration=5,  # Minimum phase duration (frames)
    noise_threshold=6      # Noise filtering threshold
)
```

### Hybrid Weight Adjustment

```python
hybrid_detector = HybridFPSPhaseDetector(
    ball_weight=0.7,    # Ball size weight
    torso_weight=0.3    # Body size weight
)
```

## 📝 Phase Definitions

All detectors recognize the following phases:

1. **General**: Neutral state
2. **Set-up**: Ball is held, ready to shoot
3. **Loading**: Lower body bends, preparing to jump
4. **Rising**: Arms and body rise for the shot
5. **Release**: Ball is released
6. **Follow-through**: Arm remains extended after release

## ⚠️ Precautions

1. **Data Format**: `pose_data` and `ball_data` must match the format used by the analyzer.
2. **Performance**: Hybrid detector is most accurate but slightly slower.
3. **Memory**: Torso-based detector uses a cache for smoothing torso length.

## 🔄 Migration

To use a new detector in your code:

```python
# Old code
analyzer.segment_shooting_phases()

# New code
analyzer.segment_shooting_phases(detector_type="hybrid")  # Recommended
analyzer.segment_shooting_phases(detector_type="ball")    # Ball-based
analyzer.segment_shooting_phases(detector_type="torso")   # Torso-based
```

## 📈 Future Improvements

1. **Machine learning-based**: Data-driven optimization
2. **Adaptive thresholds**: Real-time learning
3. **Multi-camera support**: Analysis from multiple angles
4. **Real-time processing**: GPU acceleration 