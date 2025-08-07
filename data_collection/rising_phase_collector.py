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

        self.dip_knee_angles = [] # angle between hip, knee, and ankle at dip
        self.dip_hip_angles = []
        self.dip_incline_angles = []
        self.dip_ball_vertical_distances = [] # vertical distance between the ball and hip at dip
        self.dip_foot_horizontal_distances = [] # horizontal distance between the foot and the eyes at dip
        self.dip_foot_toward_rim = [] # which foot is closer to the rim at dip

        self.selected_hand = "right"
        self.video_fps = 30  # Default FPS, can be overridden by metadata
        self.set_point_threshold = 0.1  # Threshold for setpoint detection

    def _get_fps_ratio(self):
        """
        Get the ratio of video FPS to base FPS.
        
        Returns:
            Ratio of video FPS to base FPS
        """
        return math.ceil(self.video_fps) // self.BASE_VIDEO_FPS
 
    def _get_dip(self, rising_ball, release_y):
        """
        Get the dip frame index based on the last frame before the ball begins moving upward

        Args:
            rising_pose: The first frame's pose dict.

        Returns:
            Dip value (float).
        """
        fps_ratio = self._get_fps_ratio()
        if not rising_ball or len(rising_ball) < 2 * fps_ratio:
            return 0

        lookback = max(1, (int) (2 * fps_ratio))

        # Step 1: Compute distances to rim
        distances = []
        for ball in rising_ball:
            if not ball or "center_y" not in ball:
                distances.append(-1)
                continue
            distances.append(abs(ball.get("center_y", 0) - release_y))
    
        # Step 2: Traverse backward to get the last frame where the vertical distance is decreasing 
        for i in range(len(distances) - lookback, lookback, -1):
            look_back_arr = []
            look_ahead_arr = []
            for j in range(i - lookback, i):
                if distances[j] == -1:
                    continue
                look_back_arr.append(distances[j])
            for j in range(i + 1, i + lookback):
                if distances[j] == -1:
                    continue
                look_ahead_arr.append(distances[j])
        
            if np.mean(look_back_arr) < np.mean(look_ahead_arr):
                return i
        return 0

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
        fps_ratio = self._get_fps_ratio()
        if not rising_ball or len(rising_ball) < 2 * fps_ratio:
            return 0

        lookback = max(1, (int) (2 * fps_ratio))

        # Step 1: Compute distances to rim
        distances = []
        for ball in rising_ball:
            if not ball or "center_x" not in ball:
                distances.append(-1)
                continue
            distances.append(abs(ball.get("center_x", 0) - release_x))
    
        # Step 2: Traverse backward to get the last frame where the horizontal distance is decreasing 
        for i in range(len(distances) - lookback, lookback, -1):
            look_back_arr = []
            look_ahead_arr = []
            for j in range(i - lookback, i):
                if distances[j] == -1:
                    continue
                look_back_arr.append(distances[j])
            for j in range(i + 1, i + lookback):
                if distances[j] == -1:
                    continue
                look_ahead_arr.append(distances[j])
        
            if np.mean(look_back_arr) < np.mean(look_ahead_arr):
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

    def _get_first_ball_x(self, windup):
        """
        Get the first ball's x coordinate in normalized coordinates.

        Args:
            windup: List of ball dicts with 'center_x', 'center_y'
            fps_ratio: How many frames to backtrack from transition point

        Returns:
            First ball's x coordinate
        """
        for i in range(len(windup)):
            ball = windup[i]
            if not ball or "center_x" not in ball:
                continue
            first_ball_x = ball.get("center_x", 0)
            if first_ball_x != 0.0:
                return first_ball_x
        return 0.0

    def _get_last_ball_x(self, windup):
        """
        Get the last ball's x coordinate in normalized coordinates.

        Args:
            windup: List of ball dicts with 'center_x', 'center_y'

        Returns:
            Last ball's x coordinate
        """
        for i in range(len(windup) - 1, -1, -1):
            ball = windup[i]
            if not ball or "center_x" not in ball:
                continue
            last_ball_x = ball.get("center_x", 0)
            if last_ball_x != 0.0:
                return last_ball_x
        return 0.0
    
    def _get_last_ball_y(self, windup):
        """
        Get the last ball's y coordinate in normalized coordinates.

        Args:
            windup: List of ball dicts with 'center_x', 'center_y'

        Returns:
            Last ball's y coordinate
        """
        for i in range(len(windup) - 1, -1, -1):
            ball = windup[i]
            if not ball or "center_y" not in ball:
                continue
            last_ball_y = ball.get("center_y", 0)
            if last_ball_y != 0.0:
                return last_ball_y
        return 0.0

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

    def _calculate_dip_data(self, rising_pose, rising_ball, dip_idx, release_x):
        """
        Calculate the dip data based on the rising pose and ball data.

        Args:
            rising_pose: List of pose dicts per frame.
            rising_ball: List of ball dicts per frame.
            dip_idx: Index of the dip frame.
        """
        if dip_idx < 0 or dip_idx >= len(rising_pose):
            return

        pose = rising_pose[dip_idx]
        ball = rising_ball[dip_idx]
        fps_ratio = self._get_fps_ratio()
        start_idx = max(0, dip_idx - fps_ratio)
        end_idx = min(len(rising_pose), dip_idx +fps_ratio)

        knee_angle = []
        hip_angle = []
        incline_angle = []
        ball_vertical_distances = []
        dip_foot_horizontal_distances = []
        dip_foot_toward_rim = []
        for i in range(start_idx, end_idx):
            pose = rising_pose[i]
            ball = rising_ball[i]
            ankle = pose.get(f"{self.selected_hand}_ankle", {})
            hip = pose.get(f"{self.selected_hand}_hip", {})
            knee = pose.get(f"{self.selected_hand}_knee", {})
            shoulder = pose.get(f"{self.selected_hand}_shoulder", {})

            if hip and knee and ankle:
                knee_angle.append(self.calculate_angle(
                    hip['x'], hip['y'],
                    knee['x'], knee['y'],
                    ankle['x'], ankle['y']
                ))

            if hip and shoulder:
                incline_angle.append(self.calculate_incline_angle(hip, shoulder))

            if hip and knee and shoulder:
                hip_angle.append(self.calculate_angle(
                    knee['x'], knee['y'],
                    hip['x'], hip['y'],
                    shoulder['x'], shoulder['y']
                ))

            if ball and hip:
                ball_vertical_distance = self.calculate_ball_vertical_distance(ball, hip)
                ball_vertical_distances.append(ball_vertical_distance)

            if pose.get("right_ankle") and pose.get("left_ankle"):
                right_ankle = pose["right_ankle"]
                left_ankle = pose["left_ankle"]
                if right_ankle.get("x", 0) != 0.0 and left_ankle.get("x", 0) != 0.0:
                    right_foot_distance = abs(right_ankle["x"] - release_x)
                    left_foot_distance = abs(left_ankle["x"] - release_x)
                    diff_distance = right_ankle.get("x", 0) - left_ankle.get("x", 0)
                    dip_foot_horizontal_distances.append(
                        diff_distance if diff_distance >= 0 else -diff_distance
                    )
                    dip_foot_toward_rim.append(
                        0 if right_foot_distance < left_foot_distance else 1
                    )

        self.dip_knee_angles.append(np.mean(knee_angle) if knee_angle else 0.0)
        self.dip_hip_angles.append(np.mean(hip_angle) if hip_angle else 0.0)
        self.dip_incline_angles.append(np.mean(incline_angle) if incline_angle else 0.0)
        self.dip_ball_vertical_distances.append(np.mean(ball_vertical_distances) if ball_vertical_distances else 0.0)
        self.dip_foot_horizontal_distances.append(np.mean(dip_foot_horizontal_distances) if dip_foot_horizontal_distances else 0.0)
        self.dip_foot_toward_rim.append("right" if np.mean(dip_foot_toward_rim) < 0.5 else "left" if np.mean(dip_foot_toward_rim) > 0.5 else "equal")

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
             # move the ball to an absolute reference point
            first_ball_x = self._get_first_ball_x(windup)
            release_ball_x = ball.get("center_x", 0) if ball.get("center_x", 0) != 0.0 else self._get_last_ball_x(windup)
            release_ball_y = ball.get("center_y", 0) if ball.get("center_y", 0) != 0.0 else self._get_last_ball_y(windup)
    
            release_x = release_ball_x + (first_ball_x - ref_point) * 100000 # move the relase ball x further to ensure consistent reference point

            setpoint_idx = self._get_setpoint_toward_rim(rising_ball=windup, release_x=release_x)
            dip_idx = self._get_dip(windup, release_ball_y)
            print(f"Setpoint index: {setpoint_idx + i}, Windup length: {len(windup)}")
            print(f"dip index: {dip_idx + i}")
            self._calculate_setpoint_data(rising_pose, windup, setpoint_idx)
            self._calculate_dip_data(rising_pose, windup, dip_idx, release_x)
            self._get_movement_after_setpoint(rising_pose, setpoint_idx)
            self.windup.append(windup)
            i = j

if __name__ == "__main__":
    collector = RisingPhaseCollector()
    json_file = "../data/results/two_score_two_miss_normalized_output.json"
    json_file = "../data/results/bakke_closeview_normalized_output.json"  

    collector.collect_data(json_file)
    print("elbow angles: ", collector.elbow_angles)
    print("armpit angles: ", collector.armpit_angles)
    print("knee angles: ", collector.knee_angles)
    print("incline angles: ", collector.incline_angles)
    print("vertical ball distance: ", collector.ball_vertical_distances)
    print("horizontal ball distance: ", collector.ball_horizontal_distances)
    # print("windup: ", collector.windup)
    print("moving upward: ", collector.moveming_upward)

    print("dip knee angles: ", collector.dip_knee_angles)
    print("dip hip angles: ", collector.dip_hip_angles)
    print("dip incline angles: ", collector.dip_incline_angles)
    print("dip ball vertical distances: ", collector.dip_ball_vertical_distances)
    print("dip foot horizontal distances: ", collector.dip_foot_horizontal_distances)
    print("dip foot toward rim: ", collector.dip_foot_toward_rim)