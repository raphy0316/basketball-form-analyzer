# Basketball Shooting Form Analysis System (MoveNet-based)

This system analyzes basketball shooting form using MoveNet and provides integrated pose/ball extraction, phase segmentation, and visualization.

## Features

- **Pose Extraction**: Extracts player pose data from video using MoveNet.
- **Ball Trajectory Extraction**: Detects and tracks the basketball in video frames.
- **Shooting Phase Segmentation**: Automatically segments the shooting motion into phases (Set-up, Loading, Rising, Release, Follow-through, Recovery) based on 2D keypoint and ball trajectory analysis.
- **Visualization**: Generates side-by-side videos showing original and normalized pose/ball data with phase labels.
- **Comparison with Reference Data**: Compare extracted data with reference datasets for advanced analysis.

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
    "phase_detection_method": "sequential_transition"
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

| Phase           | Frame-Based Definition Summary                                 |
|-----------------|---------------------------------------------------------------|
| Set-up          | Stable posture before loading                                 |
| Loading         | Maximum knee bend                                             |
| Rising          | Jump upward while arms lift                                   |
| Release         | Wrist and ball begin downward motion                          |
| Follow-through  | Arm stays extended as lower body descends                     |
| Recovery        | Motion ends and body stabilizes                               |

## Shooting Phase Transition Table

| From → To             | Condition                                                                                 |
|----------------------|------------------------------------------------------------------------------------------|
| General → Set-up     | The ball is held in hand for at least 3 consecutive frames *(Not implemented yet)*        |
| Set-up → Loading     | At least 2 out of 3: knee, wrist, or ball are moving downward (y-coordinate decreasing)   |
| Loading → Rising     | Wrist, elbow, and ball are all moving upward simultaneously (y increasing)                |
| Rising → Release     | - Ball leaves the hand *(Currently disabled)*<br>- Left or right elbow angle ≥ 120°       |
| Release → Follow-through | - Ball has fully left the hand<br>- Knee or hip begins to descend (y starts decreasing) |
| Follow-through → Recovery | - Knee starts rising again after descending<br>- Elbow angle ≤ 80° (arm begins to fold) |
| Recovery → General   | Knee, wrist, and hip movement becomes minimal for several frames (abs(Δy) < threshold)    |

**Note:**
- Conditions marked as *Currently disabled* or *Not implemented yet* are temporarily excluded due to inconsistent values and detection errors, which affect accurate phase detection.
- Once the data has been cleaned and stabilized, these conditions will be implemented according to the logic described here.
- If the logic is later modified, please document and update the changes in this section.

## Keypoints

MoveNet detects 17 keypoints:
- nose, left_eye, right_eye, left_ear, right_ear
- left_shoulder, right_shoulder, left_elbow, right_elbow
- left_wrist, right_wrist, left_hip, right_hip
- left_knee, right_knee, left_ankle, right_ankle


## Download Required Model File (YOLOv8 .pt)

**Note:** The YOLOv8 model file (`yolov8n736-customContinue.pt`) required for ball detection is not included in this repository due to its large size.

To use the shooting analysis model, please download the `.pt` file from the following link:

[Download yolov8n736-customContinue.pt from Google Drive](https://drive.google.com/file/d/1ndN5pBUZ4IDE31kZioMTKsHTJu2x2_IK/view)

After downloading, place the file in the `ball_extraction` folder:

```
ball_extraction/yolov8n736-customContinue.pt
```

Without this file, ball detection and full shooting analysis will not work.


## Video Preparation

- **Place the video you want to analyze in the `data/video` folder.**
- The program will only show videos located in this folder for selection.

If the 'data/visualized_video' folder is not auto-created by the code, add a note instructing users to create it manually. If it is auto-created, no note is needed. 