# Configuration settings for the basketball form analyzer

# Ball detection settings
MIN_BALL_SIZE = 0.01
MIN_BALL_CONFIDENCE = 0.3
MIN_RIM_CONFIDENCE = 0.4

# Video settings
DEFAULT_FPS = 22

# File settings
BASE_FILENAME = "demo"
OUTPUT_DIR = "data/extracted_data"

# Pose detection settings
POSE_CONFIDENCE_THRESHOLD = 0.3

# Keypoint names
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
