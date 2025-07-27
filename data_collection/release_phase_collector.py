import json
import numpy as np

class ReleasePhaseCollector:

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
            cnt = 0
            elbow_angle = 0.0
            armpit_angle = 0.0
            knee_angle = 0.0
            incline_angle = 0.0
            ball_distance = 0.0
            elbow_distance = 0.0

            while(j < n and phase == "Release"):
                elbow = pose.get(f"{selected_hand}_elbow", {})
                shoulder = pose.get(f"{selected_hand}_shoulder", {})
                wrist = pose.get(f"{selected_hand}_wrist", {})
                hip = pose.get(f"{selected_hand}_hip", {})
                knee = pose.get(f"{selected_hand}_knee", {})
                ball = frame.get("normalized_ball", {})
                ankle = pose.get(f"{selected_hand}_ankle", {})
                eyes = pose.get(f"{selected_hand}_eye", {})

                if shoulder and elbow and wrist:
                    elbow_angle += self._calculate_angle(
                        shoulder['x'], shoulder['y'],
                        elbow['x'], elbow['y'],
                        wrist['x'], wrist['y']
                    )

                if elbow and shoulder and hip:
                    armpit_angle += self._calculate_angle(
                        elbow['x'], elbow['y'],
                        shoulder['x'], shoulder['y'],
                        hip['x'], hip['y']
                    )
                    incline_angle += self._calculate_incline_angle(hip, shoulder)

                if hip and knee and ankle:
                    knee_angle += self._calculate_angle(
                        hip['x'], hip['y'],
                        knee['x'], knee['y'],
                        ankle['x'], ankle['y']
                    )

                if ball and eyes:
                    ball_distance += self._calculate_ball_distance(ball, eyes)

                if elbow and eyes:
                    elbow_distance += self._calculate_vertical_distance(elbow, eyes)

                j += 1
                cnt += 1
                frame = self.data["frames"][j]
                pose = frame.get("normalized_pose", {})
                phase = frame.get("phase", "General")
    
            # Average the angles and distances over the number of frames in the release phase
            if cnt > 0:
                elbow_angle /= cnt
                armpit_angle /= cnt
                knee_angle /= cnt
                incline_angle /= cnt
                ball_distance /= cnt
                elbow_distance /= cnt
                self.elbow_angles.append(elbow_angle)
                self.armpit_angles.append(armpit_angle)
                self.knee_angles.append(knee_angle)
                self.incline_angles.append(incline_angle)
                self.ball_distances.append(ball_distance)
                self.elbow_distances.append(elbow_distance)
            i = j

    def _calculate_vertical_distance(self, point1, point2):
        """
        Calculate the vertical distance between two points.
        
        Args:
            point1: Dictionary with 'x' and 'y' coordinates of the first point
            point2: Dictionary with 'x' and 'y' coordinates of the second point
            
        Returns:
            Vertical distance between the two points
        """
        return - (point1['y'] - point2['y'])

    def _calculate_incline_angle(self, hip, shoulder):
        """
        Calculate the incline angle between the straight line extended from hip to shoulder.
        
        Args:
            hip: Dictionary with 'x' and 'y' coordinates of the hip
            shoulder: Dictionary with 'x' and 'y' coordinates of the shoulder
            
        Returns:
            Incline angle in degrees
        """
        if not hip or not shoulder:
            return 0.0
        
        ab_x = 0
        ab_y = 1

        cb_x = shoulder['x'] - hip['x']
        cb_y = shoulder['y'] - hip['y']

        # Dot product
        dot_product = ab_x * cb_x + ab_y * cb_y

        # Magnitudes
        ab_magnitude = np.sqrt(ab_x**2 + ab_y**2)
        cb_magnitude = np.sqrt(cb_x**2 + cb_y**2)

        # Avoid division by zero
        if ab_magnitude == 0 or cb_magnitude == 0:
            return 0.0

        # Cosine of angle
        cos_angle = dot_product / (ab_magnitude * cb_magnitude)

        # Clamp to valid range
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Convert to degrees
        angle = np.arccos(cos_angle) * 180 / np.pi

        return 180 - angle

    def _calculate_ball_distance(self, ball, eyes):
        """
        Calculate the distance between the ball and the eyes.
        
        Args:
            ball: Dictionary with 'x' and 'y' coordinates of the ball
            eyes: Dictionary with 'x' and 'y' coordinates of the eyes
        Returns:
            Distance between the ball and the eyes
        """
        return np.sqrt((ball['center_x'] - eyes['x']) ** 2 + (ball['center_y'] - eyes['y']) ** 2)

    def _calculate_angle(self, ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
        """
        Calculate angle between three points.

        Args:
            ax, ay: First point coordinates
            bx, by: Second point coordinates (vertex)
            cx, cy: Third point coordinates

        Returns:
            Angle in degrees
        """
        # Vector AB
        ab_x = ax - bx
        ab_y = ay - by

        # Vector CB
        cb_x = cx - bx
        cb_y = cy - by

        # Dot product
        dot_product = ab_x * cb_x + ab_y * cb_y

        # Magnitudes
        ab_magnitude = np.sqrt(ab_x**2 + ab_y**2)
        cb_magnitude = np.sqrt(cb_x**2 + cb_y**2)

        # Avoid division by zero
        if ab_magnitude == 0 or cb_magnitude == 0:
            return 0.0

        # Cosine of angle
        cos_angle = dot_product / (ab_magnitude * cb_magnitude)

        # Clamp to valid range
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Convert to degrees
        angle = np.arccos(cos_angle) * 180 / np.pi

        return angle

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