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
        self.output_dir = "data/visualized_video/comparison"
        
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
    
    def create_comprehensive_visualizations(self, video1_path: str, video2_path: str, 
                                         video1_data: Dict, video2_data: Dict,
                                         comparison_results: Dict, video_info: Dict) -> Dict[str, str]:
        """
        Create comprehensive visualizations for all DTW analysis types
        
        Args:
            video1_path: Path to first video
            video2_path: Path to second video
            video1_data: First video analysis data
            video2_data: Second video analysis data
            comparison_results: DTW comparison results
            video_info: Video metadata information
            
        Returns:
            Dictionary of output video paths
        """
        output_videos = {}
        
        # Create output directory
        base_name1 = os.path.splitext(os.path.basename(video1_path))[0]
        base_name2 = os.path.splitext(os.path.basename(video2_path))[0]
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎬 Creating comprehensive visualizations...")
        
        # 1. Overall Coordinate-based Visualization
        if 'coordinate_overall' in comparison_results:
            print("📊 Creating overall coordinate-based visualization...")
            coord_overall_path = os.path.join(output_dir, f"{base_name1}_vs_{base_name2}_coord_overall.mp4")
            success = self._create_overall_visualization(
                video1_path, video2_path, coord_overall_path,
                video1_data, video2_data, comparison_results, video_info,
                'coordinate_overall', 'Coordinate-based Overall'
            )
            if success:
                output_videos['coord_overall'] = coord_overall_path
        
        # 2. Overall Feature-based Visualization
        if 'feature_overall' in comparison_results:
            print("📊 Creating overall feature-based visualization...")
            feature_overall_path = os.path.join(output_dir, f"{base_name1}_vs_{base_name2}_feature_overall.mp4")
            success = self._create_overall_visualization(
                video1_path, video2_path, feature_overall_path,
                video1_data, video2_data, comparison_results, video_info,
                'feature_overall', 'Feature-based Overall'
            )
            if success:
                output_videos['feature_overall'] = feature_overall_path
        
        # 3. Phase-specific Coordinate-based Visualization
        print("📊 Creating phase-specific coordinate-based visualization...")
        coord_phase_path = os.path.join(output_dir, f"{base_name1}_vs_{base_name2}_coord_phases.mp4")
        success = self._create_phase_specific_visualization(
            video1_path, video2_path, coord_phase_path,
            video1_data, video2_data, comparison_results, video_info,
            'coordinate', 'Coordinate-based Phase'
        )
        if success:
            output_videos['coord_phases'] = coord_phase_path
        
        # 4. Phase-specific Feature-based Visualization
        print("📊 Creating phase-specific feature-based visualization...")
        feature_phase_path = os.path.join(output_dir, f"{base_name1}_vs_{base_name2}_feature_phases.mp4")
        success = self._create_phase_specific_visualization(
            video1_path, video2_path, feature_phase_path,
            video1_data, video2_data, comparison_results, video_info,
            'feature', 'Feature-based Phase'
        )
        if success:
            output_videos['feature_phases'] = feature_phase_path
        
        return output_videos
    
    def _create_overall_visualization(self, video1_path: str, video2_path: str, output_path: str,
                                     video1_data: Dict, video2_data: Dict,
                                     comparison_results: Dict, video_info: Dict,
                                     analysis_type: str, title: str) -> bool:
        """
        Create overall visualization (coordinate or feature based)
        
        Args:
            analysis_type: 'coordinate_overall' or 'feature_overall'
            title: Title for the visualization
        """
        try:
            # Get overall warping path
            overall_result = comparison_results.get(analysis_type, {})
            warping_path = overall_result.get('warping_path', [])
            actual_frame_path = overall_result.get('actual_frame_path', warping_path)
            
            if not actual_frame_path:
                print(f"❌ No warping path found for {analysis_type}")
                return False
            
            # Create video writer
            cap1 = cv2.VideoCapture(video1_path)
            cap2 = cv2.VideoCapture(video2_path)
            
            if not cap1.isOpened() or not cap2.isOpened():
                print("❌ Error opening video files")
                return False
            
            # Get video properties
            fps1 = video_info.get('video1_fps', 30.0)
            fps2 = video_info.get('video2_fps', 30.0)
            width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
            height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
            height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_width = max(width1, width2) * 2
            output_height = max(height1, height2)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, min(fps1, fps2), (output_width, output_height))
            
            if not out.isOpened():
                print("❌ Error creating output video")
                return False
            
            print(f"📊 Processing {len(actual_frame_path)} overall matched frames...")
            
            # Process each matched frame pair
            for i, (frame1_idx, frame2_idx) in enumerate(actual_frame_path):
                # Read frames
                cap1.set(cv2.CAP_PROP_POS_FRAMES, frame1_idx)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, frame2_idx)
                
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    print(f"⚠️  Failed to read frames: V1:{frame1_idx}, V2:{frame2_idx}")
                    continue
                
                # Resize frames
                frame1_resized = cv2.resize(frame1, (output_width // 2, output_height))
                frame2_resized = cv2.resize(frame2, (output_width // 2, output_height))
                
                # Create side-by-side frame
                combined_frame = np.hstack([frame1_resized, frame2_resized])
                
                # Add title and frame info
                combined_frame = self._add_overlay_text(
                    combined_frame, title, frame1_idx, frame2_idx, i, len(actual_frame_path)
                )
                
                out.write(combined_frame)
                
                # Progress indicator
                if i % 30 == 0:
                    progress = (i / len(actual_frame_path)) * 100
                    print(f"   📈 Progress: {progress:.1f}%")
            
            cap1.release()
            cap2.release()
            out.release()
            
            print(f"✅ {title} visualization created: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating {title} visualization: {e}")
            return False
    
    def _create_phase_specific_visualization(self, video1_path: str, video2_path: str, output_path: str,
                                           video1_data: Dict, video2_data: Dict,
                                           comparison_results: Dict, video_info: Dict,
                                           analysis_type: str, title: str) -> bool:
        """
        Create phase-specific visualization (coordinate or feature based)
        
        Args:
            analysis_type: 'coordinate' or 'feature'
            title: Title for the visualization
        """
        try:
            # Get phase-specific warping paths
            phase_paths = {}
            phase_order = ['loading', 'rising', 'release', 'follow_through']
            
            print(f"\n🔍 Available comparison results keys: {list(comparison_results.keys())}")
            
            for phase in phase_order:
                phase_key = f"{phase}_{analysis_type}" if analysis_type == 'coordinate' else phase
                print(f"   Looking for key: {phase_key}")
                if phase_key in comparison_results:
                    phase_result = comparison_results[phase_key]
                    warping_path = phase_result.get('warping_path', [])
                    actual_frame_path = phase_result.get('actual_frame_path', [])
                    print(f"   Found {phase_key}: {len(warping_path)} warping path entries")
                    print(f"   Actual frame path: {len(actual_frame_path)} entries")
                    if warping_path:
                        phase_paths[phase] = {
                            'warping_path': warping_path,
                            'actual_frame_path': actual_frame_path
                        }
                else:
                    print(f"   ❌ Key {phase_key} not found in comparison results")
            
            if not phase_paths:
                print(f"❌ No phase-specific warping paths found for {analysis_type}")
                return False
            
            # Create video writer
            cap1 = cv2.VideoCapture(video1_path)
            cap2 = cv2.VideoCapture(video2_path)
            
            if not cap1.isOpened() or not cap2.isOpened():
                print("❌ Error opening video files")
                return False
            
            # Get video properties
            fps1 = video_info.get('video1_fps', 30.0)
            fps2 = video_info.get('video2_fps', 30.0)
            width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
            height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
            height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_width = max(width1, width2) * 2
            output_height = max(height1, height2)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, min(fps1, fps2), (output_width, output_height))
            
            if not out.isOpened():
                print("❌ Error creating output video")
                return False
            
            # Get frame data for phase mapping
            frame_data1 = video1_data.get('frames', [])
            frame_data2 = video2_data.get('frames', [])
            
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
            
            # Process each phase in order
            total_frames = 0
            phase_frame_mappings = {}
            
            # First pass: calculate total frames and create mappings
            for phase in phase_order:
                if phase not in phase_paths:
                    continue
                
                phase_data = phase_paths[phase]
                warping_path = phase_data['warping_path']
                actual_frame_path = phase_data['actual_frame_path']
                
                print(f"   📊 {phase} phase:")
                print(f"      Warping path length: {len(warping_path)}")
                print(f"      Actual frame path length: {len(actual_frame_path)}")
                
                # Use actual_frame_path if available, otherwise convert warping_path
                if actual_frame_path:
                    phase_matches = actual_frame_path
                    print(f"      Using actual_frame_path: {len(phase_matches)} matches")
                    # Show first few matches for debugging
                    if phase_matches:
                        print(f"      First 3 matches: {phase_matches[:3]}")
                else:
                    # Convert warping path to actual frame indices (fallback)
                    phase_matches = []
                    # Determine which actual phases this DTW covers
                    if phase == 'loading':
                        target_phases = ['Loading', 'Loading-Rising']
                    elif phase == 'rising':
                        target_phases = ['Rising', 'Loading-Rising']
                    elif phase == 'release':
                        target_phases = ['Release']
                    elif phase == 'follow_through':
                        target_phases = ['Follow-through']
                    else:
                        continue
                    
                    # Get frames for this DTW phase from both videos
                    video1_dtw_frames = []
                    video2_dtw_frames = []
                    
                    for phase_name in target_phases:
                        video1_dtw_frames.extend(phase_frames1.get(phase_name, []))
                        video2_dtw_frames.extend(phase_frames2.get(phase_name, []))
                    
                    video1_dtw_frames.sort()
                    video2_dtw_frames.sort()
                    
                    print(f"      Video1 DTW frames: {len(video1_dtw_frames)}")
                    print(f"      Video2 DTW frames: {len(video2_dtw_frames)}")
                    
                    for path_idx1, path_idx2 in warping_path:
                        if path_idx1 < len(video1_dtw_frames) and path_idx2 < len(video2_dtw_frames):
                            actual_frame1 = video1_dtw_frames[path_idx1]
                            actual_frame2 = video2_dtw_frames[path_idx2]
                            phase_matches.append((actual_frame1, actual_frame2))
                    
                    print(f"      Converted warping_path: {len(phase_matches)} matches")
                    if phase_matches:
                        print(f"      First 3 converted matches: {phase_matches[:3]}")
                
                phase_frame_mappings[phase] = phase_matches
                total_frames += len(phase_matches)
            
            print(f"📊 Total phase-specific matched frames: {total_frames}")
            
            # Second pass: create visualization
            frame_count = 0
            
            for phase in phase_order:
                if phase not in phase_frame_mappings:
                    continue
                
                phase_matches = phase_frame_mappings[phase]
                print(f"   📊 Processing {phase} phase: {len(phase_matches)} frames")
                
                # Process each matched frame pair in this phase
                for i, (frame1_idx, frame2_idx) in enumerate(phase_matches):
                    # Read frames
                    cap1.set(cv2.CAP_PROP_POS_FRAMES, frame1_idx)
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame2_idx)
                    
                    ret1, frame1 = cap1.read()
                    ret2, frame2 = cap2.read()
                    
                    if not ret1 or not ret2:
                        print(f"⚠️  Failed to read frames: V1:{frame1_idx}, V2:{frame2_idx}")
                        continue
                    
                    # Resize frames
                    frame1_resized = cv2.resize(frame1, (output_width // 2, output_height))
                    frame2_resized = cv2.resize(frame2, (output_width // 2, output_height))
                    
                    # Create side-by-side frame
                    combined_frame = np.hstack([frame1_resized, frame2_resized])
                    
                    # Add phase info and frame info
                    combined_frame = self._add_phase_overlay_text(
                        combined_frame, title, phase, frame1_idx, frame2_idx, 
                        frame_count, total_frames
                    )
                    
                    out.write(combined_frame)
                    frame_count += 1
                    
                    # Progress indicator
                    if frame_count % 30 == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"   📈 Progress: {progress:.1f}%")
            
            cap1.release()
            cap2.release()
            out.release()
            
            print(f"✅ {title} visualization created: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating {title} visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _add_overlay_text(self, frame: np.ndarray, title: str, frame1_idx: int, 
                          frame2_idx: int, current_frame: int, total_frames: int) -> np.ndarray:
        """Add overlay text to frame"""
        # Add title
        cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add frame information
        frame_info = f"Frame {current_frame+1}/{total_frames}"
        cv2.putText(frame, frame_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add frame indices
        cv2.putText(frame, f"V1: {frame1_idx}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"V2: {frame2_idx}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def _add_phase_overlay_text(self, frame: np.ndarray, title: str, phase: str, 
                               frame1_idx: int, frame2_idx: int, 
                               current_frame: int, total_frames: int) -> np.ndarray:
        """Add phase-specific overlay text to frame"""
        # Add title
        cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add phase information
        phase_display = phase.replace('_', ' ').title()
        cv2.putText(frame, f"Phase: {phase_display}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Add frame information
        frame_info = f"Frame {current_frame+1}/{total_frames}"
        cv2.putText(frame, frame_info, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add frame indices
        cv2.putText(frame, f"V1: {frame1_idx}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"V2: {frame2_idx}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame


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