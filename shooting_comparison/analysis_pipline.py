import os
import sys
import json
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime

from data_collection.dtw_processor import DTWProcessor
from shooting_comparison.shooting_comparison_visualizer import ShootingComparisonVisualizer
from shooting_comparison.analysis_utils import get_analysis_utils
from shooting_comparison.setup_analyzer import SetupAnalyzer
from shooting_comparison.loading_analyzer import LoadingAnalyzer
from shooting_comparison.rising_analyzer import RisingAnalyzer
from shooting_comparison.release_analyzer import ReleaseAnalyzer
from shooting_comparison.follow_through_analyzer import FollowThroughAnalyzer
from shooting_comparison.landing_analyzer import LandingAnalyzer
from shooting_comparison.analysis_interpreter import AnalysisInterpreter

class AnalysisPipeline:
    def __init__(self, video1_path: str):
        self.video1_path = video1_path
        self.video1_data = None
        self.setup_analyzer = SetupAnalyzer()
        self.loading_analyzer = LoadingAnalyzer()
        self.rising_analyzer = RisingAnalyzer()
        self.release_analyzer = ReleaseAnalyzer()
        self.follow_through_analyzer = FollowThroughAnalyzer()
        self.landing_analyzer = LandingAnalyzer()
        self.analysis_interpreter = AnalysisInterpreter()
        self.results_dir = "data/results"

    def _get_base_name(self, video_path: str) -> str:
        """Get base name from video path"""
        return os.path.splitext(os.path.basename(video_path))[0]
    
    def process_video_data(self, video_path: str) -> Optional[Dict]:
        """
        Load existing processed video data from JSON file and count shots
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing processed data or None if failed
        """
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        result_file = os.path.join(self.results_dir, f"{base_name}_normalized_output.json")
        
        print(f"\nProcessing: {os.path.basename(video_path)}")

        # Check if results already exist
        if os.path.exists(result_file):
            print(f"Found existing results: {os.path.basename(result_file)}")
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"Loaded {len(data.get('frames', []))} frames")
                    
                    # Count shots in the data
                    shots = data.get('metadata', {}).get('shots', {})
                    
                    # Handle both list and dictionary formats for shots
                    if isinstance(shots, list):
                        shot_count = len(shots)
                        print(f"Found {shot_count} shots in the video")
                        
                        # Display shot information for list format
                        if shot_count > 0:
                            print("Shot Information:")
                            for i, shot_info in enumerate(shots):
                                if isinstance(shot_info, dict):
                                    shot_id = shot_info.get('shot_id', i+1)
                                    original_id = shot_info.get('original_shot_id', 'N/A')
                                    start_frame = shot_info.get('start_frame', 'N/A')
                                    end_frame = shot_info.get('end_frame', 'N/A')
                                    fixed_torso = shot_info.get('fixed_torso', 'N/A')
                                    if original_id != 'N/A' and original_id != shot_id:
                                        print(f"   shot{shot_id}: Frames {start_frame}-{end_frame}, Torso: {fixed_torso} (Original ID: {original_id})")
                                    else:
                                        print(f"   shot{shot_id}: Frames {start_frame}-{end_frame}, Torso: {fixed_torso}")
                                else:
                                    print(f"   shot{i+1}: {shot_info}")
                    else:
                        # Dictionary format
                        shot_count = len(shots)
                        print(f"Found {shot_count} shots in the video")
                        
                        # Display shot information for dictionary format
                        if shot_count > 0:
                            print("Shot Information:")
                            for shot_id, shot_info in shots.items():
                                start_frame = shot_info.get('start_frame', 'N/A')
                                end_frame = shot_info.get('end_frame', 'N/A')
                                fixed_torso = shot_info.get('fixed_torso', 'N/A')
                                print(f"   {shot_id}: Frames {start_frame}-{end_frame}, Torso: {fixed_torso}")
                    
                    # Store metadata for analysis
                    if video_path == self.video1_path:
                        self.video1_metadata = data.get('metadata', {})
                    
                    return data
            except Exception as e:
                print(f"Error loading existing results: {e}")
                return None
        
        # No processed data found
        print(f"   No processed data found for {base_name}")
        print(f"   Expected file: {result_file}")
        print(f"   Please run the integrated pipeline first to process this video.")
        return None

    def extract_phase_data(self, data: Dict) -> Dict[str, List[Dict]]:
        """
        Extract frames grouped by phases
        
        Args:
            data: Video analysis results
            
        Returns:
            Dictionary with phase names as keys and frame lists as values
        """
        phase_data = {}
        frames = data.get('frames', [])
        
        for frame in frames:
            phase = frame.get('phase', 'Unknown')
            if phase not in phase_data:
                phase_data[phase] = []
            phase_data[phase].append(frame)
        
        return phase_data

    def _filter_data_by_shot(self, data: Dict, selected_shot: Optional[str]) -> Optional[Dict]:
        """
        Filter data to include only frames from the selected shot
        
        Args:
            data: Original video data
            selected_shot: Selected shot ID (None for all shots)
            
        Returns:
            Filtered data or None if no frames match
        """
        if selected_shot is None:
             # Check if there are multiple shots - if so, use only the first shot
            shots = data.get('metadata', {}).get('shots', {})
            if isinstance(shots, list) and len(shots) > 1:
                print("Multiple shots detected - using only the first shot instead of full integration")
                selected_shot = "shot1"  # Force to use first shot
            elif isinstance(shots, dict) and len(shots) > 1:
                print("Multiple shots detected - using only the first shot instead of full integration") 
                first_shot_key = list(shots.keys())[0]
                selected_shot = first_shot_key  # Use first shot key
            else:
                return data  # Return all data if single shot or no shots
        
        frames = data.get('frames', [])
        shots = data.get('metadata', {}).get('shots', {})
        
        # Debug shot information
        print(f"Debug: Available shots: {shots}")
        print(f"Debug: Selected shot: {selected_shot}")
        print(f"Debug: Shots type: {type(shots)}")
        
        # Handle both list and dictionary formats for shots
        if isinstance(shots, list):
            # Convert shot number to index (e.g., "shot1" -> 0)
            if selected_shot.startswith("shot"):
                try:
                    shot_index = int(selected_shot[4:]) - 1  # "shot1" -> 0
                    if shot_index < 0 or shot_index >= len(shots):
                        print(f"Selected shot '{selected_shot}' not found in data (index {shot_index} out of range)")
                        return None
                    shot_info = shots[shot_index]
                except ValueError:
                    print(f"Invalid shot format: {selected_shot}")
                    return None
            else:
                print(f"Invalid shot format: {selected_shot}")
                return None
        else:
            # Dictionary format
            if selected_shot not in shots:
                print(f"Selected shot '{selected_shot}' not found in data")
                print(f"Debug: Available shot keys: {list(shots.keys()) if shots else 'None'}")
                return None
            shot_info = shots[selected_shot]
        
        start_frame = shot_info.get('start_frame', 0)
        end_frame = shot_info.get('end_frame', len(frames))
        
        print(f"Debug: Shot info - start_frame: {start_frame}, end_frame: {end_frame}")
        print(f"Debug: Total frames: {len(frames)}")
        
        # Filter frames that belong to the selected shot
        filtered_frames = []
        shot_id_count = 0
        range_count = 0
        
        for frame in frames:
            frame_idx = frame.get('frame_index', frame.get('frame_idx', 0))
            
            # Get shot_id from frame data (basketball_shooting_analyzer.py uses 'shot' field)
            shot_id = frame.get('shot')  # 먼저 'shot' 필드 확인
            
            # Fallback to 'shot_id' field if 'shot' is not available
            if shot_id is None:
                shot_id = frame.get('shot_id')
            
            # Include frame if it belongs to the selected shot
            # Handle both string and numeric shot IDs
            shot_matches = False
            
            if shot_id is not None:
                # Convert shot_id to string for comparison
                shot_id_str = str(shot_id)
                if selected_shot.startswith("shot"):
                    # Extract shot number from selected_shot (e.g., "shot1" -> "1")
                    shot_number = selected_shot[4:]  # "shot1" -> "1"
                    # Convert shot_number to integer for comparison with shot_id
                    try:
                        shot_number_int = int(shot_number)
                        shot_matches = shot_id == shot_number_int
                    except ValueError:
                        shot_matches = shot_id_str == shot_number
                    # Only print debug info for first few frames
                    if frame_idx < 5:
                        print(f"🔍 Debug: shot_id={shot_id} (type: {type(shot_id)}), shot_number={shot_number_int}, matches={shot_matches}")
                else:
                    shot_matches = shot_id_str == selected_shot
            
            # Also check frame index range as fallback
            frame_in_range = start_frame <= frame_idx <= end_frame
            
            # Always include frames that match by shot_id
            if shot_matches:
                shot_id_count += 1
                filtered_frames.append(frame)
            # Only include range matches if no shot_id matches were found
            elif frame_in_range and shot_id_count == 0:
                range_count += 1
                filtered_frames.append(frame)
        
        print(f"🔍 Debug: Frames matched by shot_id: {shot_id_count}")
        print(f"🔍 Debug: Frames matched by frame range: {range_count}")
        
        if not filtered_frames:
            print(f"No frames found for shot '{selected_shot}'")
            return None
        
        # Create filtered data
        filtered_data = data.copy()
        filtered_data['frames'] = filtered_frames
        print(filtered_data)
        print(f"Filtered {len(filtered_frames)} frames for shot '{selected_shot}'")
        
        return filtered_data

    def _extract_phase_transitions(self, video_data: Dict) -> List[str]:
        """
        Extract phase transition sequence from video data.
        
        Args:
            video_data: Video analysis data containing frames
            
        Returns:
            List of phase names in order of appearance
        """
        frames = video_data.get('frames', [])
        phase_transitions = []
        current_phase = None
        
        for frame in frames:
            phase = frame.get('phase', 'General')
            
            # Only add phase if it's different from the current phase
            if phase != current_phase:
                phase_transitions.append(phase)
                current_phase = phase
        
        return phase_transitions

    def select_shots(self, video1_data: Dict, video2_data: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        Select specific shots for comparison
        
        Args:
            video1_data: First video data
            video2_data: Second video data
            
        Returns:
            Tuple of (selected_shot1, selected_shot2) or (None, None) for all shots
        """
        print("\n STEP 2: Select Shots for Comparison")
        print("=" * 50)
        
        # Get shots from both videos
        shots1 = video1_data.get('metadata', {}).get('shots', {})
        
        shot_count1 = len(shots1)
        
        print(f"📹 Video 1 ({os.path.basename(self.video1_path)}): {shot_count1} shots")
        
        # Ask user for comparison type
        print("\n🔍 Comparison Options:")
        print("1. Compare all shots (overall comparison)")
        print("2. Compare specific shots (individual shot comparison)")
        
        while True:
            try:
                choice = input("\nSelect option (1 or 2): ").strip()
                if choice == "1":
                    print("Selected: Compare all shots")
                    return None, None  # None means all shots
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except KeyboardInterrupt:
                print("\nSelection cancelled.")
                return None, None

    def run_basic_analysis(self) -> Optional[Dict]:
        print("\n STEP 1: Basic Analysis of Individual Videos")

        self.video1_data = self.process_video_data(self.video1_path)
        # selected_shot1, _ = self.select_shots(self.video1_data, self.video1_data)
         # Check if we have valid shot selections
        shots1 = self.video1_data.get('metadata', {}).get('shots', [])
        # print(f"Debug: shots1 = {shots1}, selected_shot1 = {selected_shot1}")
        # # base_name = self._get_base_name(self.video1_path)
        # # analysis_file = f"data/results/{base_name}_normalized_output.json"  
        
        # if len(shots1) == 0:
        #     print("No shots detected in either video. Cannot perform comparison.")
        #     return False
        # elif selected_shot1 is None:
        #     print("All shots selected for comparison.")
        # elif selected_shot1 is None:
        #     print("Please select specific shots for comparison or 'all' for all shots.")
        #     return False
        # else:
        #     print(f"Selected specific shots: Video 1 - {selected_shot1}")
        if not self.video1_data:
            print("Video 1 data not available for basic analysis")
            return None
        try:
            print("🔍 [DEBUG] Starting perform_comparison...")
            
            # Initialize comparison results
            comparison_results = {}
            
            # Filter data by selected shots
            filtered_video1_data = self._filter_data_by_shot(self.video1_data, None)
            
            print("🔍 [DEBUG] Filtered data:")
            
            # Check if filtered data is None before accessing
            if filtered_video1_data is None:
                print("Video 1 data filtering failed")
                return None
                
            print(f"   Video 1: {len(filtered_video1_data.get('frames', []))} frames")
            
            # Get selected hand
            selected_hand = filtered_video1_data.get('metadata', {}).get('hand', 'right')
            print(f"🔍 [DEBUG] Selected hand: {selected_hand}")
            self.selected_hand = selected_hand  # Store for analyzers
            # Perform Set-up phase analysis
            print("📊 Performing set-up phase analysis...")
            setup_analysis1 = self.setup_analyzer.analyze_setup_phase(filtered_video1_data)
            comparison_results['setup_analysis'] = {
                'video1': setup_analysis1,
            }
            print("🔍 [DEBUG] Setup analysis completed")
            
            # Perform Loading phase analysis
            print("📊 Performing loading phase analysis...")
            loading_analysis1 = self.loading_analyzer.analyze_loading_phase(filtered_video1_data)
            comparison_results['loading_analysis'] = {
                'video1': loading_analysis1,
            }
            print("🔍 [DEBUG] Loading analysis completed")
            
            # Perform Rising phase analysis
            print("📊 Performing rising phase analysis...")
            rising_analysis1 = self.rising_analyzer.analyze_rising_phase(filtered_video1_data, self.selected_hand)
            comparison_results['rising_analysis'] = {
                'video1': rising_analysis1,
            }
            print("🔍 [DEBUG] Rising analysis completed")
            
            # Perform Release phase analysis
            print("📊 Performing release phase analysis...")
            release_analysis1 = self.release_analyzer.analyze_release_phase(filtered_video1_data, self.selected_hand)
            comparison_results['release_analysis'] = {
                'video1': release_analysis1,
            }
            print("🔍 [DEBUG] Release analysis completed")
            
            # Perform Follow-through phase analysis
            print("📊 Performing follow-through phase analysis...")
            follow_through_analysis1 = self.follow_through_analyzer.analyze_follow_through_phase(filtered_video1_data, self.selected_hand)
            comparison_results['follow_through_analysis'] = {
                'video1': follow_through_analysis1,
            }
            print("🔍 [DEBUG] Follow-through analysis completed")
            
            # Perform Landing phase analysis
            print("📊 Performing landing phase analysis...")
            landing_analysis1 = self.landing_analyzer.analyze_landing_phase(filtered_video1_data)
            comparison_results['landing_analysis'] = {
                'video1': landing_analysis1,
            }
            print("🔍 [DEBUG] Landing analysis completed")
            print("🔍 [DEBUG] All phase comparisons completed")

            # Add metadata
            comparison_results['metadata'] = {
                'video1_path': self.video1_path,
                'video1_frames': len(self.video1_data.get('frames', [])),
                'video1_fps': self.video1_metadata.get('fps', 30.0) if self.video1_metadata else 30.0,
                'selected_hand': selected_hand,
                'comparison_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'video1_phase_transitions': self._extract_phase_transitions(self.video1_data),
            }
            print("🔍 [DEBUG] Metadata added")
            
            # Add phase statistics
            video1_phases = self.extract_phase_data(filtered_video1_data)
            
            comparison_results['phase_statistics'] = {
                'video1_phases': {phase: len(frames) for phase, frames in video1_phases.items()},
            }
            print("🔍 [DEBUG] Phase statistics added")
            
            # Interpret the comparison results
            print("🔍 [DEBUG] Starting interpretation of comparison results")
            interpretation = self.analysis_interpreter.interpret_comparison_results(comparison_results)
            
            # Add interpretation to results
            comparison_results['interpretation'] = interpretation
            
            self.comparison_results = comparison_results
            print("🔍 [DEBUG] Comparison results stored with interpretation")
            return comparison_results

        except Exception as e:
            print(f"Error during comparison: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Clean up temporary files
            pass
