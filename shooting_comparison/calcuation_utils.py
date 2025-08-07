import numpy as np


def _calculate_angle_two_points(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate angle between two points with respect to horizontal axis."""
        return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def _calculate_angle(self, ax: float, ay: float, bx: float, by: float, 
                        cx: float, cy: float) -> float:
        """Calculate angle between three points."""
        # Vector AB
        ab_x = bx - ax
        ab_y = by - ay
        
        # Vector CB
        cb_x = bx - cx
        cb_y = by - cy
        
        # Dot product
        dot_product = ab_x * cb_x + ab_y * cb_y
        
        # Magnitudes
        ab_magnitude = np.sqrt(ab_x**2 + ab_y**2)
        cb_magnitude = np.sqrt(cb_x**2 + cb_y**2)
        
        # Avoid division by zero
        if ab_magnitude == 0 or cb_magnitude == 0:
            return 0.0
        
        # Calculate angle
        cos_angle = dot_product / (ab_magnitude * cb_magnitude)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to valid range
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle