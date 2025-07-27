import json
import numpy as np
from base_phase_collector import BasePhaseCollector

class ReleasePhaseCollector(BasePhaseCollector):

    def __init__(self):
        self.data = []
        self.elbow_angles = [] # angle between shoulder, elbow, and wrist
        self.armpit_angles = [] # angle between elbow, shoulder, and hip
        self.knee_angles = [] # angle between hip, knee, and ankle
        self.incline_angles = [] # angle between the straight line extended from hip, hip, and shoulder
        self.ball_distances = [] # distance between the ball and the eyes
        self.elbow_distances = [] # distance between the elbow and the eyes

    def collect_data(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        selected_hand = self.data["metadata"].get("hand", "right")

        i = 0
        n = len(self.data["frames"])

        # two pointers to find the release phase
        while i < n:
            frame = self.data["frames"][i]
            pose = frame.get("normalized_pose", {})
            phase = frame.get("phase", "General")

            if not pose:
                i += 1
                continue
            elif phase != "Release":
                i += 1
                continue

            j = i
            # cnt = 0
            elbow_angle = []
            armpit_angle = []
            knee_angle = []
            incline_angle = []
            ball_distance = []
            elbow_distance = []

            while j < n and phase == "Release":
                elbow = pose.get(f"{selected_hand}_elbow", {})
                shoulder = pose.get(f"{selected_hand}_shoulder", {})
                wrist = pose.get(f"{selected_hand}_wrist", {})
                hip = pose.get(f"{selected_hand}_hip", {})
                knee = pose.get(f"{selected_hand}_knee", {})
                ball = frame.get("normalized_ball", {})
                ankle = pose.get(f"{selected_hand}_ankle", {})
                eyes = pose.get(f"{selected_hand}_eye", {})

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
                    incline_angle.append(self.calculate_incline_angle(hip, shoulder))

                if hip and knee and ankle:
                    knee_angle.append(self.calculate_angle(
                        hip['x'], hip['y'],
                        knee['x'], knee['y'],
                        ankle['x'], ankle['y']
                    ))

                if ball and eyes:
                    ball_distance.append(self.calculate_ball_distance(ball, eyes))

                if elbow and eyes:
                    elbow_distance.append(self.calculate_vertical_distance(elbow, eyes))

                j += 1

                if j < n:
                    frame = self.data["frames"][j]
                    pose = frame.get("normalized_pose", {})
                    phase = frame.get("phase", "General")
    
            # Average the angles and distances over the number of frames in the release phase
            self.elbow_angles.append(np.mean(elbow_angle))
            self.armpit_angles.append(np.mean(armpit_angle))
            self.knee_angles.append(np.mean(knee_angle))
            self.incline_angles.append(np.mean(incline_angle))
            self.ball_distances.append(np.mean(ball_distance))
            self.elbow_distances.append(np.mean(elbow_distance))
            i = j
    
if __name__ == "__main__":

    collector = ReleasePhaseCollector()
    json_file = "../data/results/bakke_closeview_normalized_output.json"  
    collector.collect_data(json_file)
    print("elbow angles: ", collector.elbow_angles)
    print("armpit angles: ", collector.armpit_angles)
    print("knee angles: ", collector.knee_angles)
    print("incline angles: ", collector.incline_angles)
    print("ball distance: ", collector.ball_distances)
    print("elbow distance: ", collector.elbow_distances)