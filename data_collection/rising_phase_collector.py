import json
import numpy as np
from base_phase_collector import BasePhaseCollector
import math
class RisingPhaseCollector(BasePhaseCollector):

    def __init__(self):
        super().__init__()
        self.data = []
        self.elbow_angles = [] # angle between shoulder, elbow, and wrist
        self.armpit_angles = [] # angle between elbow, shoulder, and hip
        self.knee_angles = [] # angle between hip, knee, and ankle
        self.incline_angles = [] # angle between the straight line extended from hip, hip, and shoulder
        self.ball_vertical_distances = [] # vertical distance between the ball and the eyes
        self.ball_horizontal_distances = [] # horizontal distance between the ball and the eyes
        self.windup = [] # windup data
        self.moveming_upward = []
        self.selected_hand = "right"
        self.video_fps = 30  # Default FPS, can be overridden by metadata
    
    def _get_fps_ratio(self):
        """
        Get the ratio of video FPS to base FPS.
        
        Returns:
            Ratio of video FPS to base FPS
        """
        return math.ceil(self.video_fps) // self.BASE_VIDEO_FPS
 
    def _get_setpoint(self, rising_pose, rising_ball):
        """
        Get the setpoint frame index from rising phase where the ball is closest to the eyes.

        Args:
            rising_pose: List of pose dicts per frame (right-handed shooter assumed).
            rising_ball: List of ball dicts per frame (aligned with rising_pose).

        Returns:
            Index of the setpoint frame in rising_pose list.
        """
        if len(rising_pose) != len(rising_ball) or len(rising_pose) == 0:
            return 0  # fallback

        min_dist = float("inf")
        setpoint_idx = -1

        for i, (pose, ball) in enumerate(zip(rising_pose, rising_ball)):
            # Extract ball coords and correct x
            ball_x = ball.get("center_x", 0)
            ball_y = ball.get("center_y", 0)

            # Extract eye coords and correct x
            left_eye = pose.get("left_eye", {"x": 0, "y": 0})
            right_eye = pose.get("right_eye", {"x": 0, "y": 0})
            eye_x = ((left_eye.get("x", 0) + right_eye.get("x", 0)) / 2)
            eye_y = (left_eye.get("y", 0) + right_eye.get("y", 0)) / 2

            # Compute distance
            dist = np.linalg.norm([ball_x - eye_x, ball_y - eye_y])
            if dist < min_dist:
                min_dist = dist
                setpoint_idx = i
        
        return setpoint_idx

    def _get_setpoint_toward_rim(self, rising_ball, release_x) -> int:
        """
        Scan backward to find the last frame before the ball begins consistent inward movement.

        Args:
            rising_ball: List of ball dicts with 'center_x', 'center_y'
            release_x: Rim x in normalized coordinates
            fps_ratio: How many frames to backtrack from transition point

        Returns:
            Setpoint frame index
        """
        if not rising_ball or len(rising_ball) < 6:
            return 0
        fps_ratio = self._get_fps_ratio()
        lookback = max(1, 3 * fps_ratio)

        # Step 1: Compute distances to rim
        distances = [abs(ball.get("center_x", 0) - release_x) for ball in rising_ball]
    
        # Step 2: Traverse backward to get the last frame where the horizontal distance is decreasing 
        for i in range(len(distances) - lookback - 1, lookback - 1, -1):
            if distances[i] < distances[i + 1]:
                return i
        return 0

    def _get_movement_after_setpoint(self, rising_pose, setpoint_idx):
        """
        Get the movement data after the setpoint frame.

        Args:
            rising_pose: List of pose dicts per frame.
            setpoint_idx: Index of the setpoint frame.

        Returns:
            Movement data after the setpoint.
        """
        if setpoint_idx < 0 or setpoint_idx >= len(rising_pose):
            return []

        rising_pose_after_setpoint = rising_pose[setpoint_idx + 1:]
        setpoint_right_hip_y = rising_pose[setpoint_idx].get("right_hip", {}).get("y", 0)
        setpoint_left_hip_y = rising_pose[setpoint_idx].get("left_hip", {}).get("y", 0)
       
        right_hip_dy = sum(pose.get("right_hip", {}).get("y", 0) - setpoint_right_hip_y for pose in rising_pose_after_setpoint)
        left_hip_dy = sum(pose.get("left_hip", {}).get("y", 0) - setpoint_left_hip_y for pose in rising_pose_after_setpoint)

        self.moveming_upward.append((right_hip_dy + left_hip_dy) / 2 < 0)  # True if hips are moving upward after setpoint
    
    def _get_ref_point(self, rising_pose):
        """
        Get the reference point for the release phase based on the first frame's pose.

        Args:
            rising_pose: The first frame's pose dict.

        Returns:
            Reference point (float).
        """
        ref_point = rising_pose.get("right_shoulder", {}).get("x", 0)
        if ref_point == 0:
            ref_point = rising_pose.get("left_shoulder", {}).get("x", 0)
        if ref_point == 0:
            ref_point = rising_pose.get("right_eye", {}).get("x", 0)
        if ref_point == 0:
            ref_point = rising_pose.get("left_eye", {}).get("x", 0)
        if ref_point == 0:
            ref_point = rising_pose.get("right_hip", {}).get("x", 0)
        if ref_point == 0:
            ref_point = rising_pose.get("left_hip", {}).get("x", 0)
 
        return ref_point

    def _calculate_setpoint_data(self, rising_pose, rising_ball, setpoint_idx):
        
        fps_ratio = self._get_fps_ratio()
        start_idx = max(0, setpoint_idx - fps_ratio)
        end_idx = min(len(rising_pose), setpoint_idx +fps_ratio)
        # cnt = end_idx - start_idx

        elbow_angle = []
        armpit_angle = []
        knee_angle = []
        incline_angle = []
        ball_vertical_distance = []
        ball_horizontal_distance = []

        for i in range(start_idx, end_idx):
            pose = rising_pose[i]
            ball = rising_ball[i]

            elbow = pose.get(f"{self.selected_hand}_elbow", {})
            shoulder = pose.get(f"{self.selected_hand}_shoulder", {})
            wrist = pose.get(f"{self.selected_hand}_wrist", {})
            hip = pose.get(f"{self.selected_hand}_hip", {})
            knee = pose.get(f"{self.selected_hand}_knee", {})
            ankle = pose.get(f"{self.selected_hand}_ankle", {})
            eyes = pose.get(f"{self.selected_hand}_eye", {})

            if shoulder and elbow and wrist:
                elbow_angle.append(self.calculate_angle(
                    shoulder['x'], shoulder['y'],
                    elbow['x'], elbow['y'],
                    wrist['x'], wrist['y']
                ))

            if elbow and shoulder and hip:
                armpit_angle.append(self.calculate_angle(
                    elbow['x'], elbow['y'],
                    shoulder['x'], shoulder['y'],
                    hip['x'], hip['y']
                ))

            if hip and knee and ankle:
                knee_angle.append(self.calculate_angle(
                    hip['x'], hip['y'],
                    knee['x'], knee['y'],
                    ankle['x'], ankle['y']
                ))

            if hip and shoulder:
                incline_angle.append(self.calculate_incline_angle(hip, shoulder))

            if ball and eyes:
                ball_vertical_distance.append(ball.get("center_y", 0) - eyes.get("y", 0))
                ball_horizontal_distance.append(ball.get("center_x", 0) - eyes.get("x", 0))


        self.elbow_angles.append(np.mean(elbow_angle) if elbow_angle else 0.0)
        self.armpit_angles.append(np.mean(armpit_angle) if armpit_angle else 0.0)
        self.knee_angles.append(np.mean(knee_angle) if knee_angle else 0.0)
        self.incline_angles.append(np.mean(incline_angle) if incline_angle else 0.0)
        self.ball_vertical_distances.append(np.mean(ball_vertical_distance) if ball_vertical_distance else 0.0)
        self.ball_horizontal_distances.append(np.mean(ball_horizontal_distance) if ball_horizontal_distance else 0.0)
       
    def collect_data(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.selected_hand = self.data["metadata"].get("hand", "right")
        self.video_fps = self.data["metadata"].get("fps", 30) 

        i = 0
        n = len(self.data["frames"])

        # two pointers to find the release phase
        while i < n:
            frame = self.data["frames"][i]
            pose = frame.get("normalized_pose", {})
            phase = frame.get("phase", "General")
            ball = frame.get("normalized_ball", {})
            if not pose:
                i += 1
                continue
            elif phase != "Rising":
                i += 1
                continue

            j = i
            rising_pose = []          
            windup = []

            while j < n and phase == "Rising":
        
                windup.append(ball)
                rising_pose.append(pose)

                j += 1
                if j < n:
                    frame = self.data["frames"][j]
                    pose = frame.get("normalized_pose", {})
                    phase = frame.get("phase", "General")
                    ball = frame.get("normalized_ball", {})


            ref_point = self._get_ref_point(rising_pose[0])
            release_x = ball.get("center_x", 0) + (windup[0].get("center_x", 0) - ref_point) * 100000

            setpoint_idx = self._get_setpoint_toward_rim(rising_ball=windup, release_x=release_x)
            print(f"Setpoint index: {setpoint_idx + i}, Windup length: {len(windup)}")
            self._calculate_setpoint_data(rising_pose, windup, setpoint_idx)
            self._get_movement_after_setpoint(rising_pose, setpoint_idx)
            self.windup.append(windup)
            i = j

if __name__ == "__main__":

    collector = RisingPhaseCollector()
    json_file = "../data/results/bakke_closeview_normalized_output.json"  
    collector.collect_data(json_file)
    print("elbow angles: ", collector.elbow_angles)
    print("armpit angles: ", collector.armpit_angles)   
    print("knee angles: ", collector.knee_angles)
    print("incline angles: ", collector.incline_angles)
    print("vertical ball distance: ", collector.ball_vertical_distances)
    print("horizontal ball distance: ", collector.ball_horizontal_distances)
    print("windup: ", collector.windup)
    print("moving upward: ", collector.moveming_upward)
