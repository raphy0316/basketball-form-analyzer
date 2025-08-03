"""
Shooting Comparison Visualizer Module

This module creates side-by-side comparison videos showing DTW frame matching
between two basketball shooting videos with matching scores displayed.
"""

import cv2
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class ShootingComparisonVisualizer:
    """
    Visualizer for shooting comparison results.
    
    Creates side-by-side videos showing DTW frame matching with:
    - Frame alignment based on DTW warping path
    - Phase-specific matching scores
    - Color-coded similarity indicators
    """
    
    def __init__(self):
        self.comparison_results = None
        self.video1_path = None
        self.video2_path = None
        self.output_dir = "shooting_comparison/visualized_video"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Color schemes for different phases
        self.phase_colors = {
            'General': (128, 128, 128),      # Gray
            'Set-up': (255, 165, 0),         # Orange
            'Loading': (255, 0, 0),          # Red
            'Loading-Rising': (255, 100, 100), # Light Red
            'Rising': (0, 255, 0),           # Green
            'Release': (0, 0, 255),          # Blue
            'Follow-through': (128, 0, 128), # Purple
            'overall': (255, 255, 255)       # White
        }
        
        # Create similarity colormap (red = bad, yellow = ok, green = good)
        self.similarity_colormap = LinearSegmentedColormap.from_list(
            'similarity', ['red', 'yellow', 'green'], N=256
        )
    
    def load_comparison_results(self, results_file: str) -> bool:
        """
        Load comparison results from JSON file.
        
        Args:
            results_file: Path to the comparison results JSON file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                self.comparison_results = json.load(f)
            
            # Extract video paths from results
            self.video1_path = self.comparison_results.get('video1_path')
            self.video2_path = self.comparison_results.get('video2_path')
            
            return True
        except Exception as e:
            print(f"❌ Error loading comparison results: {e}")
            return False
    
    def _normalize_distance(self, distance: float, max_distance: float = 100.0) -> float:
        """
        Normalize DTW distance to 0-1 range for color mapping.
        
        Args:
            distance: Raw DTW distance
            max_distance: Maximum expected distance for normalization
            
        Returns:
            Normalized distance (0 = similar, 1 = very different)
        """
        return min(distance / max_distance, 1.0)
    
    def _get_similarity_color(self, distance: float, max_distance: float = 100.0) -> Tuple[int, int, int]:
        """
        Get BGR color based on similarity score.
        
        Args:
            distance: DTW distance
            max_distance: Maximum distance for normalization
            
        Returns:
            BGR color tuple
        """
        normalized = self._normalize_distance(distance, max_distance)
        # Invert so that 0 (similar) = green, 1 (different) = red
        similarity_score = 1.0 - normalized
        
        # Convert matplotlib color to BGR
        rgba = self.similarity_colormap(similarity_score)
        bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
        
        return bgr
    
    def _draw_info_panel(self, frame_width: int, frame_height: int, 
                        frame1_idx: int, frame2_idx: int, 
                        phase1: str, phase2: str,
                        dtw_scores: Dict) -> np.ndarray:
        """
        Create information panel showing matching details.
        
        Args:
            frame_width: Width of video frames
            frame_height: Height of video frames
            frame1_idx: Current frame index for video 1
            frame2_idx: Current frame index for video 2
            phase1: Current phase in video 1
            phase2: Current phase in video 2
            dtw_scores: DTW scores for different phases
            
        Returns:
            Info panel image
        """
        panel_height = 200
        panel = np.zeros((panel_height, frame_width * 2, 3), dtype=np.uint8)
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Draw frame indices
        cv2.putText(panel, f"Frame {frame1_idx:4d}", (20, 30), 
                   font, font_scale, (255, 255, 255), thickness)
        cv2.putText(panel, f"Frame {frame2_idx:4d}", (frame_width + 20, 30), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Draw phases with color coding
        phase1_color = self.phase_colors.get(phase1, (255, 255, 255))
        phase2_color = self.phase_colors.get(phase2, (255, 255, 255))
        
        cv2.putText(panel, f"Phase: {phase1}", (20, 60), 
                   font, font_scale, phase1_color, thickness)
        cv2.putText(panel, f"Phase: {phase2}", (frame_width + 20, 60), 
                   font, font_scale, phase2_color, thickness)
        
        # Draw DTW scores
        y_pos = 100
        cv2.putText(panel, "DTW Scores:", (20, y_pos), 
                   font, font_scale, (255, 255, 255), thickness)
        
        y_pos += 30
        for phase_name, score_data in dtw_scores.items():
            if isinstance(score_data, dict) and 'dtw_distance' in score_data:
                distance = score_data['dtw_distance']
                color = self._get_similarity_color(distance)
                
                cv2.putText(panel, f"{phase_name}: {distance:.2f}", 
                           (20, y_pos), font, font_scale * 0.8, color, thickness)
                y_pos += 25
                
                if y_pos > panel_height - 20:
                    break
        
        return panel
    
    def _get_frame_phase(self, frame_data: List[Dict], frame_idx: int) -> str:
        """
        Get phase for a specific frame index.
        
        Args:
            frame_data: List of frame data
            frame_idx: Frame index to look up
            
        Returns:
            Phase name for the frame
        """
        if frame_idx < len(frame_data):
            return frame_data[frame_idx].get('phase', 'General')
        return 'General'
    
    def _create_dtw_comprehensive_matching(self, phase_warping_paths: Dict, frame_data1: List[Dict], frame_data2: List[Dict]) -> List[Dict]:
        """
        Create comprehensive DTW matching showing all matched frames from all phases.
        For overlapping phases (Loading-Rising), show matches from both Loading and Rising DTW.
        
        Args:
            phase_warping_paths: Dictionary of warping paths for each phase
            frame_data1: Frame data for video 1 (reference)
            frame_data2: Frame data for video 2 (comparison)
            
        Returns:
            List of match dictionaries with frame indices and phase info
        """
        all_matches = []
        
        # Create phase-to-frame mappings
        def get_phase_frames(frame_data):
            phase_frames = {}
            for i, frame in enumerate(frame_data):
                phase = frame.get('phase', 'General')
                if phase not in phase_frames:
                    phase_frames[phase] = []
                phase_frames[phase].append(i)
            return phase_frames
        
        phase_frames1 = get_phase_frames(frame_data1)
        phase_frames2 = get_phase_frames(frame_data2)
        
        print(f"\n🔍 Phase frame distribution:")
        print(f"Video1: {[(p, len(f)) for p, f in phase_frames1.items()]}")
        print(f"Video2: {[(p, len(f)) for p, f in phase_frames2.items()]}")
        
        # Process each DTW phase separately to capture all matches
        for dtw_phase, warping_path in phase_warping_paths.items():
            if not warping_path:
                continue
                
            print(f"\n📊 Processing {dtw_phase} DTW matches...")
            
            # Determine which actual phases this DTW covers
            # Loading-Rising frames are included in both Loading and Rising analysis
            if dtw_phase == 'loading':
                target_phases = ['Loading', 'Loading-Rising']
            elif dtw_phase == 'rising':
                target_phases = ['Rising', 'Loading-Rising']
            elif dtw_phase == 'release':
                target_phases = ['Release']
            elif dtw_phase == 'follow_through':
                target_phases = ['Follow-through']
            else:
                continue  # Skip coordinate_overall and feature_overall
            
            # Get frames for this DTW phase from both videos
            video1_dtw_frames = []
            video2_dtw_frames = []
            
            for phase_name in target_phases:
                video1_dtw_frames.extend(phase_frames1.get(phase_name, []))
                video2_dtw_frames.extend(phase_frames2.get(phase_name, []))
            
            video1_dtw_frames.sort()
            video2_dtw_frames.sort()
            
            print(f"   Video1 {dtw_phase} frames: {len(video1_dtw_frames)}")
            print(f"   Video2 {dtw_phase} frames: {len(video2_dtw_frames)}")
            print(f"   DTW warping path length: {len(warping_path)}")
            
            # Map DTW warping path to actual frame indices
            for path_idx1, path_idx2 in warping_path:
                if path_idx1 < len(video1_dtw_frames) and path_idx2 < len(video2_dtw_frames):
                    actual_frame1 = video1_dtw_frames[path_idx1]
                    actual_frame2 = video2_dtw_frames[path_idx2]
                    
                    phase1 = frame_data1[actual_frame1].get('phase', 'General')
                    phase2 = frame_data2[actual_frame2].get('phase', 'General')
                    
                    # Create match record
                    match_info = {
                        'frame1_idx': actual_frame1,
                        'frame2_idx': actual_frame2,
                        'phase1': phase1,
                        'phase2': phase2,
                        'dtw_source': dtw_phase,
                        'match_type': 'dtw_matched'
                    }
                    
                    all_matches.append(match_info)
        
        # Sort matches by video1 frame index
        all_matches.sort(key=lambda x: x['frame1_idx'])
        
        # Remove duplicates but preserve Loading-Rising overlaps
        unique_matches = []
        seen_frame1_indices = set()
        
        for match in all_matches:
            frame1_idx = match['frame1_idx']
            
            # For Loading-Rising frames, allow multiple matches (from loading and rising DTW)
            phase1 = match['phase1']
            if phase1 == 'Loading-Rising':
                # Always add Loading-Rising matches to show both loading and rising perspectives
                unique_matches.append(match)
            elif frame1_idx not in seen_frame1_indices:
                # For other phases, add only once
                unique_matches.append(match)
                seen_frame1_indices.add(frame1_idx)
        
        print(f"\n📈 Total DTW matches found: {len(unique_matches)}")
        
        # Group by frame1_idx to show overlapping matches
        matches_by_frame1 = {}
        for match in unique_matches:
            frame1_idx = match['frame1_idx']
            if frame1_idx not in matches_by_frame1:
                matches_by_frame1[frame1_idx] = []
            matches_by_frame1[frame1_idx].append(match)
        
        # Create final comprehensive matching list
        comprehensive_matches = []
        for frame1_idx in sorted(matches_by_frame1.keys()):
            frame_matches = matches_by_frame1[frame1_idx]
            
            if len(frame_matches) == 1:
                # Single match - straightforward
                comprehensive_matches.append(frame_matches[0])
            else:
                # Multiple matches for same frame1 (Loading-Rising overlap)
                # Add all matches to show different DTW perspectives
                for i, match in enumerate(frame_matches):
                    match_copy = match.copy()
                    match_copy['overlap_index'] = i + 1
                    match_copy['overlap_total'] = len(frame_matches)
                    match_copy['match_type'] = 'overlapping_dtw'
                    comprehensive_matches.append(match_copy)
        
        print(f"📊 Comprehensive matches (including overlaps): {len(comprehensive_matches)}")
        
        return comprehensive_matches
    
    def create_comparison_video(self, output_filename: str = None) -> str:
        """
        Create side-by-side comparison video with DTW matching.
        
        Args:
            output_filename: Custom output filename (optional)
            
        Returns:
            Path to the created video file
        """
        if not self.comparison_results:
            raise ValueError("No comparison results loaded. Call load_comparison_results() first.")
        
        if not self.video1_path or not self.video2_path:
            raise ValueError("Video paths not found in comparison results.")
        
        # Generate output filename if not provided
        if not output_filename:
            video1_name = os.path.splitext(os.path.basename(self.video1_path))[0]
            video2_name = os.path.splitext(os.path.basename(self.video2_path))[0]
            output_filename = f"{video1_name}_vs_{video2_name}_comparison.mp4"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        print(f"🎬 Creating comparison video: {output_filename}")
        
        # Open video captures
        cap1 = cv2.VideoCapture(self.video1_path)
        cap2 = cv2.VideoCapture(self.video2_path)
        
        if not cap1.isOpened() or not cap2.isOpened():
            raise ValueError("Could not open one or both video files.")
        
        # Get video properties
        fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
        fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use the FPS of the first video for output
        output_fps = fps1
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        panel_height = 200
        output_width = frame_width * 2
        output_height = frame_height + panel_height
        
        out = cv2.VideoWriter(output_path, fourcc, output_fps, 
                             (output_width, output_height))
        
        # Get DTW warping paths from phase-specific comparisons
        phase_warping_paths = {}
        dtw_scores = {}
        
        # Extract DTW results for each phase
        for analysis_type, results in self.comparison_results.items():
            if analysis_type in ['coordinate_overall', 'feature_overall', 'loading', 'rising', 'release', 'follow_through']:
                if isinstance(results, dict):
                    if 'warping_path' in results:
                        phase_warping_paths[analysis_type] = results['warping_path']
                    dtw_scores[analysis_type] = results
        
        # Load frame data for phase information
        video1_data_path = self.comparison_results.get('video1_data_path')
        video2_data_path = self.comparison_results.get('video2_data_path')
        
        frame_data1 = []
        frame_data2 = []
        
        if video1_data_path and os.path.exists(video1_data_path):
            with open(video1_data_path, 'r') as f:
                data1 = json.load(f)
                frame_data1 = data1.get('frames', [])
        
        if video2_data_path and os.path.exists(video2_data_path):
            with open(video2_data_path, 'r') as f:
                data2 = json.load(f)
                frame_data2 = data2.get('frames', [])
        
        # Create comprehensive DTW matching showing all matched frames
        dtw_matches = self._create_dtw_comprehensive_matching(phase_warping_paths, frame_data1, frame_data2)
        
        print(f"📊 Processing {len(dtw_matches)} DTW matched frame pairs...")
        
        # Process each DTW match
        for pair_idx, match_info in enumerate(dtw_matches):
            frame1_idx = match_info['frame1_idx']
            frame2_idx = match_info['frame2_idx']
            # Set frame positions
            cap1.set(cv2.CAP_PROP_POS_FRAMES, frame1_idx)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, frame2_idx)
            
            # Read frames
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                print(f"⚠️  Could not read frames at indices {frame1_idx}, {frame2_idx}")
                continue
            
            # Resize frames if necessary
            if frame1.shape[:2] != (frame_height, frame_width):
                frame1 = cv2.resize(frame1, (frame_width, frame_height))
            if frame2.shape[:2] != (frame_height, frame_width):
                frame2 = cv2.resize(frame2, (frame_width, frame_height))
            
            # Get phases for current frames
            phase1 = self._get_frame_phase(frame_data1, frame1_idx)
            phase2 = self._get_frame_phase(frame_data2, frame2_idx)
            
            # Add frame indices as overlay
            cv2.putText(frame1, f"F:{frame1_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame2, f"F:{frame2_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add phase indicators with matching info
            phase1_color = self.phase_colors.get(phase1, (255, 255, 255))
            phase2_color = self.phase_colors.get(phase2, (255, 255, 255))
            
            # Show phase and matching status
            match_status = "OK" if phase1 == phase2 else "DIFF"
            match_color = (0, 255, 0) if phase1 == phase2 else (0, 255, 255)
            
            # Extract DTW source and overlap info
            dtw_source = match_info.get('dtw_source', 'unknown')
            match_type = match_info.get('match_type', 'dtw_matched')
            overlap_info = ""
            
            if match_type == 'overlapping_dtw':
                overlap_idx = match_info.get('overlap_index', 1)
                overlap_total = match_info.get('overlap_total', 1)
                overlap_info = f" ({overlap_idx}/{overlap_total})"
            
            # Display enhanced phase information
            cv2.putText(frame1, f"{phase1} [{match_status}]", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase1_color, 2)
            cv2.putText(frame2, f"{phase2} [{match_status}]", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase2_color, 2)
            
            # Add DTW source information
            dtw_color = (255, 255, 0)  # Yellow for DTW info
            cv2.putText(frame1, f"DTW: {dtw_source.upper()}{overlap_info}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, dtw_color, 2)
            
            # Add matching indicator
            cv2.putText(frame1, f"Match: {match_status}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, match_color, 2)
            
            # Combine frames side by side
            combined_frame = np.hstack([frame1, frame2])
            
            # Create info panel
            info_panel = self._draw_info_panel(
                frame_width, frame_height,
                frame1_idx, frame2_idx,
                phase1, phase2, dtw_scores
            )
            
            # Combine video and info panel
            final_frame = np.vstack([combined_frame, info_panel])
            
            # Write frame
            out.write(final_frame)
            
            # Progress indicator
            if pair_idx % 50 == 0:
                progress = (pair_idx / len(dtw_matches)) * 100
                print(f"⏳ Progress: {progress:.1f}% ({pair_idx}/{len(dtw_matches)})")
        
        # Cleanup
        cap1.release()
        cap2.release()
        out.release()
        
        print(f"✅ Comparison video created: {output_path}")
        
        return output_path


def create_shooting_comparison_visualization(comparison_results_file: str, 
                                           output_filename: str = None) -> str:
    """
    Standalone function to create shooting comparison visualization.
    
    Args:
        comparison_results_file: Path to comparison results JSON file
        output_filename: Custom output filename (optional)
        
    Returns:
        Path to the created visualization video
    """
    visualizer = ShootingComparisonVisualizer()
    
    if not visualizer.load_comparison_results(comparison_results_file):
        raise ValueError(f"Could not load comparison results from {comparison_results_file}")
    
    return visualizer.create_comparison_video(output_filename)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python shooting_comparison_visualizer.py <results_file> [output_filename]")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output_path = create_shooting_comparison_visualization(results_file, output_name)
        print(f"🎉 Visualization complete: {output_path}")
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        sys.exit(1)