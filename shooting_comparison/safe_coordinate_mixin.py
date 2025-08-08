"""
Safe coordinate calculation mixin for shooting comparison analyzers.
Provides type-safe arithmetic operations and critical frame handling.
"""
import numpy as np
from typing import List, Dict, Optional, Union


class SafeCoordinateMixin:
    """Mixin class providing safe coordinate calculations for shooting analysis."""
    
    def _is_critical_frame(self, frame_type: str, context: Dict = None) -> bool:
        """
        Determine if this is a critical frame that requires special handling.
        
        Args:
            frame_type: Type of frame ('release_point', 'setup_point', etc.)
            context: Additional context information
            
        Returns:
            True if this is a critical frame
        """
        critical_frames = {
            'release_point': True,
            'setup_point': True, 
            'dip_point': True,
            'peak_height': True,
            'follow_through_end': True
        }
        return critical_frames.get(frame_type, False)
    
    def _get_best_ball_coordinate(self, frames: List[Dict], center_frame: int, 
                                 search_range: List[int], coord: str) -> Optional[float]:
        """
        Find the best available ball coordinate within search range for critical frames.
        
        Args:
            frames: List of frame data
            center_frame: Index of the target frame
            search_range: List of frame offsets to search (e.g., [-2, -1, 0, 1, 2])
            coord: Coordinate name ('center_x', 'center_y')
            
        Returns:
            Best available coordinate value or None
        """
        for offset in search_range:
            frame_idx = center_frame + offset
            if 0 <= frame_idx < len(frames):
                ball = frames[frame_idx].get('ball', {})
                value = ball.get(coord)
                if value is not None:
                    return value
        return None
    
    def _get_weighted_ball_position(self, frames: List[Dict], center_frame: int, 
                                   weights: List[float], search_range: List[int]) -> Optional[Dict]:
        """
        Calculate weighted average ball position from surrounding frames.
        
        Args:
            frames: List of frame data
            center_frame: Index of the target frame
            weights: Weights for each frame in search_range
            search_range: List of frame offsets
            
        Returns:
            Dict with 'x' and 'y' coordinates or None
        """
        valid_positions = []
        valid_weights = []
        
        for i, offset in enumerate(search_range):
            frame_idx = center_frame + offset
            if 0 <= frame_idx < len(frames):
                ball = frames[frame_idx].get('ball', {})
                x, y = ball.get('center_x'), ball.get('center_y')
                if x is not None and y is not None:
                    valid_positions.append({'x': x, 'y': y})
                    valid_weights.append(weights[i])
        
        if not valid_positions:
            return None
            
        # Calculate weighted average
        total_weight = sum(valid_weights)
        if total_weight == 0:
            return None
            
        avg_x = sum(pos['x'] * w for pos, w in zip(valid_positions, valid_weights)) / total_weight
        avg_y = sum(pos['y'] * w for pos, w in zip(valid_positions, valid_weights)) / total_weight
        
        return {'x': avg_x, 'y': avg_y}
    
    def _estimate_ball_from_wrist(self, frame_data: Dict) -> Optional[Dict]:
        """
        Estimate ball position from wrist position (useful for release point).
        
        Args:
            frame_data: Frame containing pose data
            
        Returns:
            Estimated ball position dict or None
        """
        pose = frame_data.get('pose', {})
        
        # Try to find shooting hand wrist (prefer right, fallback to left)
        shooting_wrist = pose.get('right_wrist') or pose.get('left_wrist')
        if not shooting_wrist:
            return None
            
        wrist_x = shooting_wrist.get('x')
        wrist_y = shooting_wrist.get('y')
        
        if wrist_x is None or wrist_y is None:
            return None
            
        # At release point, ball is usually slightly forward and up from wrist
        # These offsets may need calibration based on video resolution
        estimated_x = wrist_x + 20  # Adjust based on video scale
        estimated_y = wrist_y - 30  # Adjust based on video scale
        
        return {'center_x': estimated_x, 'center_y': estimated_y}
    
    def _safe_ball_coordinate(self, frames: List[Dict], frame_idx: int, coord: str, 
                             frame_type: str = None) -> Optional[float]:
        """
        Safely extract ball coordinate with critical frame handling.
        
        Args:
            frames: List of frame data
            frame_idx: Target frame index
            coord: Coordinate name ('center_x', 'center_y')
            frame_type: Type of frame for critical frame handling
            
        Returns:
            Safe coordinate value or None
        """
        if frame_idx < 0 or frame_idx >= len(frames):
            return None
            
        # Try direct access first
        ball = frames[frame_idx].get('ball', {})
        value = ball.get(coord)
        
        if value is not None:
            return value
            
        # If this is a critical frame, try enhanced recovery
        if frame_type and self._is_critical_frame(frame_type):
            # Try extended search range
            value = self._get_best_ball_coordinate(
                frames, frame_idx, [-2, -1, 0, 1, 2], coord
            )
            
            if value is not None:
                return value
                
            # Try biomechanical estimation for x,y coordinates
            if coord in ['center_x', 'center_y']:
                estimated = self._estimate_ball_from_wrist(frames[frame_idx])
                if estimated:
                    return estimated.get(coord)
                    
            # Try weighted average as last resort
            weighted_pos = self._get_weighted_ball_position(
                frames, frame_idx, [0.3, 0.4, 0.3], [-1, 0, 1]
            )
            if weighted_pos:
                return weighted_pos.get('x' if 'x' in coord else 'y')
        
        return None
    
    def _safe_coordinate_diff(self, coord1: Optional[float], coord2: Optional[float], 
                            default: Optional[float] = None) -> Optional[float]:
        """
        Safely calculate difference between two coordinates.
        
        Args:
            coord1: First coordinate
            coord2: Second coordinate  
            default: Default value if calculation fails
            
        Returns:
            Coordinate difference or default
        """
        if coord1 is not None and coord2 is not None:
            return coord1 - coord2
        return default
    
    def _safe_distance(self, point1: Dict, point2: Dict, 
                      default: Optional[float] = None) -> Optional[float]:
        """
        Safely calculate Euclidean distance between two points.
        
        Args:
            point1: Point with 'x' and 'y' keys
            point2: Point with 'x' and 'y' keys
            default: Default value if calculation fails
            
        Returns:
            Distance or default
        """
        x1, y1 = point1.get('x'), point1.get('y')
        x2, y2 = point2.get('x'), point2.get('y')
        
        if all(coord is not None for coord in [x1, y1, x2, y2]):
            return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return default
    
    def _safe_hip_center(self, pose: Dict, use_interpolation: bool = True) -> Optional[Dict]:
        """
        Safely calculate hip center position with fallback strategies.
        
        Args:
            pose: Pose data dictionary
            use_interpolation: Whether to use single-sided values as fallback
            
        Returns:
            Hip center position dict or None
        """
        left_hip = pose.get('left_hip', {})
        right_hip = pose.get('right_hip', {})
        
        left_x, left_y = left_hip.get('x'), left_hip.get('y')
        right_x, right_y = right_hip.get('x'), right_hip.get('y')
        
        # Prefer bilateral calculation
        if all(coord is not None for coord in [left_x, left_y, right_x, right_y]):
            return {
                'x': (left_x + right_x) / 2,
                'y': (left_y + right_y) / 2
            }
        
        # Fallback to unilateral if interpolation allowed
        if use_interpolation:
            if left_x is not None and left_y is not None:
                return {'x': left_x, 'y': left_y}
            elif right_x is not None and right_y is not None:
                return {'x': right_x, 'y': right_y}
        
        return None
    
    def _safe_shoulder_center(self, pose: Dict, use_interpolation: bool = True) -> Optional[Dict]:
        """
        Safely calculate shoulder center position with fallback strategies.
        
        Args:
            pose: Pose data dictionary
            use_interpolation: Whether to use single-sided values as fallback
            
        Returns:
            Shoulder center position dict or None
        """
        left_shoulder = pose.get('left_shoulder', {})
        right_shoulder = pose.get('right_shoulder', {})
        
        left_x, left_y = left_shoulder.get('x'), left_shoulder.get('y')
        right_x, right_y = right_shoulder.get('x'), right_shoulder.get('y')
        
        # Prefer bilateral calculation
        if all(coord is not None for coord in [left_x, left_y, right_x, right_y]):
            return {
                'x': (left_x + right_x) / 2,
                'y': (left_y + right_y) / 2
            }
        
        # Fallback to unilateral if interpolation allowed
        if use_interpolation:
            if left_x is not None and left_y is not None:
                return {'x': left_x, 'y': left_y}
            elif right_x is not None and right_y is not None:
                return {'x': right_x, 'y': right_y}
        
        return None