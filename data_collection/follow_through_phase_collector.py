import json
import numpy as np
import math

class FollowThroughPhaseCollector:
    """
    Follow-through Phase Collector
    
    Collects data for follow-through phase analysis:
    - Maximum angle between wrist, elbow, and shoulder
    - Vertical distance between elbow and eyes when maximum angle is reached
    - Time to reach maximum angle from release phase
    - Angle between hip, knee, and ankle when maximum angle is reached
    """

    def __init__(self):
        self.data = []
        self.max_arm_angles = []  # Maximum angle between wrist, elbow, and shoulder
        self.elbow_eye_distances = []  # Vertical distance between elbow and eyes at max angle
        self.time_to_max_angle = []  # Time (frames) to reach maximum angle from release
        self.knee_angles_at_max = []  # Angle between hip, knee, ankle at max arm angle
        self.selected_hand = "right"
        self.video_fps = 30  # Default FPS, can be overridden by metadata
        self.BASE_VIDEO_FPS = 30
    
    def _get_fps_ratio(self):
        """
        Get the ratio of video FPS to base FPS.
        
        Returns:
            Ratio of video FPS to base FPS
        """
        return math.ceil(self.video_fps) // self.BASE_VIDEO_FPS
    
    def calculate_angle(self, ax, ay, bx, by, cx, cy):
        """
        Calculate angle between three points (ax,ay)-(bx,by)-(cx,cy) in degrees.
        
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
    
    def _find_max_arm_angle_frame(self, follow_through_pose, selected_hand="right"):
        """
        Find the frame where the arm angle (wrist-elbow-shoulder) is maximum.
        
        Args:
            follow_through_pose: List of pose dicts per frame
            selected_hand: "left" or "right"
            
        Returns:
            Index of frame with maximum arm angle
        """
        if not follow_through_pose:
            return 0
        
        max_angle = -1
        max_angle_frame = 0
        
        for i, pose in enumerate(follow_through_pose):
            # Get keypoints for selected hand (same as rising collector)
            shoulder = pose.get(f"{selected_hand}_shoulder", {})
            elbow = pose.get(f"{selected_hand}_elbow", {})
            wrist = pose.get(f"{selected_hand}_wrist", {})
            
            # Check if all keypoints are available
            if (shoulder and elbow and wrist and
                'x' in shoulder and 'y' in shoulder and
                'x' in elbow and 'y' in elbow and
                'x' in wrist and 'y' in wrist):
                
                # Calculate arm angle
                angle = self.calculate_angle(
                    wrist.get('x', 0), wrist.get('y', 0),
                    elbow.get('x', 0), elbow.get('y', 0),
                    shoulder.get('x', 0), shoulder.get('y', 0)
                )
                
                if angle > max_angle:
                    max_angle = angle
                    max_angle_frame = i
        
        return max_angle_frame
    
    def _calculate_follow_through_data(self, follow_through_pose, follow_through_ball, max_angle_frame, release_frame_idx):
        """
        Calculate follow-through phase data at the maximum arm angle frame.
        
        Args:
            follow_through_pose: List of pose dicts per frame
            follow_through_ball: List of ball dicts per frame
            max_angle_frame: Index of frame with maximum arm angle
            release_frame_idx: Index of release phase start frame
        """
        if max_angle_frame >= len(follow_through_pose):
            return
        
        pose = follow_through_pose[max_angle_frame]
        
        # Get keypoints for selected hand (same as rising collector)
        shoulder = pose.get(f"{self.selected_hand}_shoulder", {})
        elbow = pose.get(f"{self.selected_hand}_elbow", {})
        wrist = pose.get(f"{self.selected_hand}_wrist", {})
        
        # Get eye keypoints (same as rising collector)
        left_eye = pose.get("left_eye", {})
        right_eye = pose.get("right_eye", {})
        
        # Get hip, knee, ankle keypoints (same as rising collector)
        hip = pose.get(f"{self.selected_hand}_hip", {})
        knee = pose.get(f"{self.selected_hand}_knee", {})
        ankle = pose.get(f"{self.selected_hand}_ankle", {})
        
        # 1. Maximum angle between wrist, elbow, and shoulder
        if (shoulder and elbow and wrist and
            'x' in shoulder and 'y' in shoulder and
            'x' in elbow and 'y' in elbow and
            'x' in wrist and 'y' in wrist):
            
            max_arm_angle = self.calculate_angle(
                wrist.get('x', 0), wrist.get('y', 0),
                elbow.get('x', 0), elbow.get('y', 0),
                shoulder.get('x', 0), shoulder.get('y', 0)
            )
            self.max_arm_angles.append(max_arm_angle)
        else:
            self.max_arm_angles.append(0.0)
        
        # 2. Vertical distance between elbow and eyes when maximum angle is reached
        if (elbow and left_eye and right_eye and
            'x' in elbow and 'y' in elbow and
            'x' in left_eye and 'y' in left_eye and
            'x' in right_eye and 'y' in right_eye):
            
            elbow_y = elbow.get('y', 0)
            left_eye_y = left_eye.get('y', 0)
            right_eye_y = right_eye.get('y', 0)
            eye_y = (left_eye_y + right_eye_y) / 2
            
            elbow_eye_distance = abs(elbow_y - eye_y)
            self.elbow_eye_distances.append(elbow_eye_distance)
        else:
            self.elbow_eye_distances.append(0.0)
        
        # 3. Time to reach maximum angle from release phase
        time_to_max = max_angle_frame  # Frames from start of follow-through
        self.time_to_max_angle.append(time_to_max)
        
        # 4. Angle between hip, knee, and ankle when maximum angle is reached
        # Use selected hand side leg (same as rising collector)
        if (hip and knee and ankle and
            'x' in hip and 'y' in hip and
            'x' in knee and 'y' in knee and
            'x' in ankle and 'y' in ankle):
            
            knee_angle = self.calculate_angle(
                hip.get('x', 0), hip.get('y', 0),
                knee.get('x', 0), knee.get('y', 0),
                ankle.get('x', 0), ankle.get('y', 0)
            )
            self.knee_angles_at_max.append(knee_angle)
        else:
            self.knee_angles_at_max.append(0.0)
    
    def collect_data(self, json_file):
        """
        Collect follow-through phase data from JSON file.
        
        Args:
            json_file: Path to JSON file with normalized data
        """
        with open(json_file, "r") as f:
            self.data = json.load(f)
        
        self.selected_hand = self.data["metadata"].get("hand", "right")
        self.video_fps = self.data["metadata"].get("fps", 30)
        
        i = 0
        n = len(self.data["frames"])
        
        # Find follow-through phases
        while i < n:
            frame = self.data["frames"][i]
            pose = frame.get("normalized_pose", {})
            phase = frame.get("phase", "General")
            ball = frame.get("normalized_ball", {})
            
            if not pose:
                i += 1
                continue
            elif phase != "Follow-through":
                i += 1
                continue
            
            j = i
            follow_through_pose = []
            follow_through_ball = []
            release_frame_idx = i  # Frame where follow-through starts
            
            # Collect all follow-through frames
            while j < n and phase == "Follow-through":
                follow_through_pose.append(pose)
                follow_through_ball.append(ball)
                
                j += 1
                if j < n:
                    frame = self.data["frames"][j]
                    pose = frame.get("normalized_pose", {})
                    phase = frame.get("phase", "General")
                    ball = frame.get("normalized_ball", {})
            
            # Find frame with maximum arm angle
            max_angle_frame = self._find_max_arm_angle_frame(follow_through_pose, self.selected_hand)
            print(f"Max angle frame: {max_angle_frame + i}, Follow-through length: {len(follow_through_pose)}")
            
            # Calculate follow-through data
            self._calculate_follow_through_data(follow_through_pose, follow_through_ball, max_angle_frame, release_frame_idx)
            
            i = j

if __name__ == "__main__":
    collector = FollowThroughPhaseCollector()
    json_file = "../data/results/bakke_closeview_normalized_output.json"
    collector.collect_data(json_file)
    
    print("Follow-through Phase Analysis Results:")
    print("=" * 50)
    print(f"Max arm angles: {collector.max_arm_angles}")
    print(f"Elbow-eye distances: {collector.elbow_eye_distances}")
    print(f"Time to max angle (frames): {collector.time_to_max_angle}")
    print(f"Knee angles at max arm angle: {collector.knee_angles_at_max}")
    
    # Calculate averages
    if collector.max_arm_angles:
        avg_max_angle = np.mean(collector.max_arm_angles)
        avg_elbow_eye_dist = np.mean(collector.elbow_eye_distances)
        avg_time_to_max = np.mean(collector.time_to_max_angle)
        avg_knee_angle = np.mean(collector.knee_angles_at_max)
        
        print("\nAverages:")
        print(f"Average max arm angle: {avg_max_angle:.2f}°")
        print(f"Average elbow-eye distance: {avg_elbow_eye_dist:.2f}")
        print(f"Average time to max angle: {avg_time_to_max:.1f} frames")
        print(f"Average knee angle at max: {avg_knee_angle:.2f}°")