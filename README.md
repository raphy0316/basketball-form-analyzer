# Basketball Shooting Form Analysis System (MoveNet-based)

This system analyzes basketball shooting form using MoveNet and provides integrated pose/ball extraction, phase segmentation, and visualization.

## Features

- **Pose Extraction**: Extracts player pose data from video using MoveNet.
- **Ball Trajectory Extraction**: Detects and tracks the basketball in video frames.
- **Shooting Phase Segmentation**: Automatically segments the shooting motion into phases (General, Set-up, Loading, Rising, Release, Follow-through) based on 2D keypoint and ball trajectory analysis.
- **Visualization**: Generates side-by-side videos showing original and normalized pose/ball data with phase labels.
- **Reference Comparison**: Compare extracted data with reference datasets for advanced analysis.

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download MoveNet Model

The MoveNet model will be downloaded automatically on first run.

## Usage

### Integrated Pipeline (Recommended)

Run the full pipeline (extraction → normalization → phase segmentation → visualization):

```bash
python basketball_shooting_integrated_pipeline.py
```

You will be prompted to select a video and choose overwrite options for existing data.

### Main Classes

- `BasketballShootingIntegratedPipeline`: One-click pipeline for extraction, normalization, phase segmentation, and visualization.
- `BasketballShootingAnalyzer`: Core analysis and phase segmentation logic.
- `PoseExtractionPipeline`, `BallExtractionPipeline`: For separate pose/ball extraction.

### Example: Programmatic Use

```python
from basketball_shooting_integrated_pipeline import BasketballShootingIntegratedPipeline

pipeline = BasketballShootingIntegratedPipeline()
pipeline.run_full_pipeline("data/video/your_video.mp4", overwrite_mode=True)
```

## Output Data Format

### Normalized Pose/Ball Data (JSON)

```json
{
  "metadata": {
    "video_path": "path/to/video.mp4",
    "total_frames": 300,
    "normalization_method": "ball_radius_based",
    "phase_detection_method": "hybrid_fps"
  },
  "frames": [
    {
      "frame_index": 0,
      "phase": "Set-up",
      "normalized_pose": { ... },
      "normalized_ball": { ... },
      "original_hip_center": [x, y],
      "scaling_factor": 12.3,
      "ball_detected": true
    }
  ]
}
```

## Shooting Phase Definitions

| Phase           | Description                                              |
|-----------------|---------------------------------------------------------|
| General         | Neutral state before shooting sequence                  |
| Set-up          | Ball is held, ready to shoot                            |
| Loading         | Lower body bends, preparing to jump                     |
| Rising          | Arms and body rise for the shot                         |
| Release         | Ball is released                                        |
| Follow-through  | Arm remains extended after release                      |

## Shooting Phase Transition Summary

- **General → Set-up**: Ball is held in hand (close contact)
- **Set-up → Loading**: Hip and/or shoulder move downward
- **Loading → Rising**: Wrist and elbow move upward relative to hip
- **Rising → Release**: Ball is released (distance from wrist increases, proper form)
- **Release → Follow-through**: Ball has fully left the hand (distance > threshold)
- **Follow-through → General**: Wrist drops below eyes relative to hip, or ball is caught

All transition logic is implemented in the HybridFPSPhaseDetector and matches the latest research and codebase.

## Keypoints

MoveNet detects 17 keypoints:
- nose, left_eye, right_eye, left_ear, right_ear
- left_shoulder, right_shoulder, left_elbow, right_elbow
- left_wrist, right_wrist, left_hip, right_hip
- left_knee, right_knee, left_ankle, right_ankle

## Model Files

**Note:** The YOLOv8 model file (`yolov8n736-customContinue.pt`) required for ball detection is not included in this repository due to its large size.

To use the shooting analysis model, please download the `.pt` file from the following link:

[Download yolov8n736-customContinue.pt from Google Drive](https://drive.google.com/file/d/1ndN5pBUZ4IDE31kZioMTKsHTJu2x2_IK/view)

After downloading, place the file in the `ball_extraction/models` folder:

```
ball_extraction/models/yolov8n736-customContinue.pt
```

Without this file, ball detection and full shooting analysis will not work.

## Video Preparation

- Place the video you want to analyze in the `data/video` folder.
- The program will only show videos located in this folder for selection.

## Notes
- All code, logs, and comments are now in English.
- For best results, use the HybridFPSPhaseDetector (default in the integrated pipeline).
- If you encounter issues with large files, use Git LFS and check your .gitignore settings. 