import json
import numpy as np

class BasePhaseCollector:

    def __init__(self):
        self.BASE_VIDEO_FPS = 30  # Default FPS
    def calculate_vertical_distance(self, point1, point2):
        """
        Calculate the vertical distance between two points.
        
        Args:
            point1: Dictionary with 'x' and 'y' coordinates of the first point
            point2: Dictionary with 'x' and 'y' coordinates of the second point
            
        Returns:
            Vertical distance between the two points
        """
        return -(point1['y'] - point2['y'])

    def calculate_incline_angle(self, hip, shoulder):
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

    def calculate_ball_distance(self, ball, eyes):
        """
        Calculate the distance between the ball and the eyes.
        
        Args:
            ball: Dictionary with 'x' and 'y' coordinates of the ball
            eyes: Dictionary with 'x' and 'y' coordinates of the eyes
        Returns:
            Distance between the ball and the eyes
        """
        return np.sqrt((ball['center_x'] - eyes['x']) ** 2 + (ball['center_y'] - eyes['y']) ** 2)

    def calculate_angle(self, ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
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
