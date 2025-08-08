#!/usr/bin/env python3
"""
Basketball Shooting Form Comparison Pipeline

This module provides functionality to compare shooting forms between two videos
by analyzing extracted pose and ball data from the integrated pipeline.
"""

import os
import sys
import json
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class ShootingComparisonPipeline:
    """Pipeline for comparing basketball shooting forms between two videos"""
    
    def __init__(self):
        """Initialize the comparison pipeline"""
        self.video_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "video")
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results")
        self.comparison_results_dir = os.path.join(os.path.dirname(__file__), "results")
        self.selected_hand = 'right'  # Default hand
        self.prompt_file_name = None  # Path to saved LLM prompt
        # Create comparison results directory if it doesn't exist
        os.makedirs(self.comparison_results_dir, exist_ok=True)

        # Initialize analyzers
        self.setup_analyzer = SetupAnalyzer()
        self.loading_analyzer = LoadingAnalyzer()
        self.rising_analyzer = RisingAnalyzer()
        self.release_analyzer = ReleaseAnalyzer()
        self.follow_through_analyzer = FollowThroughAnalyzer()
        self.landing_analyzer = LandingAnalyzer()

        # Initialize analysis interpreter
        self.analysis_interpreter = AnalysisInterpreter()

        # Initialize DTW processor
        self.dtw_processor = DTWProcessor()
        self.analysis_utils = get_analysis_utils()

        # Video and data storage
        self.video1_path = None
        self.video2_path = None
        self.video1_data = None
        self.video2_data = None
        self.video1_metadata = None
        self.video2_metadata = None
        self.comparison_results = None

    def select_videos(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Select two videos for comparison using file dialog.
        
        Returns:
            Tuple of (video1_path, video2_path) or (None, None) if cancelled
        """
        print("📹 Select the first video (Reference):")
        video1_path = self._select_video_file()
        if not video1_path:
            print("❌ No video selected for Video 1")
            return None, None
        
        print("📹 Select the second video (Comparison):")
        video2_path = self._select_video_file()
        if not video2_path:
            print("❌ No video selected for Video 2")
            return None, None
        
        # Allow same video selection
        if video1_path == video2_path:
            print("ℹ️  Same video selected for both. This will compare different shots from the same video.")
        
        # Set the video paths
        self.video1_path = video1_path
        self.video2_path = video2_path
        
        return video1_path, video2_path
    
    def _select_video_file(self) -> Optional[str]:
        """
        Select a video file using file dialog.
        
        Returns:
            Selected video path or None if cancelled
        """
        # Hide main tkinter window
        root = tk.Tk()
        root.withdraw()
        
        try:
            video_path = filedialog.askopenfilename(
                title="Select Video File",
                initialdir=self.video_dir,
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ]
            )
            
            if video_path:
                print(f"✅ Video selected: {os.path.basename(video_path)}")
                return video_path
            else:
                print("❌ No video selected.")
                return None
                
        except Exception as e:
            print(f"❌ Error selecting video: {e}")
            return None
        finally:
            root.destroy()
    
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
        
        print(f"\n🔍 Processing: {os.path.basename(video_path)}")
        
        # Check if results already exist
        if os.path.exists(result_file):
            print(f"✅ Found existing results: {os.path.basename(result_file)}")
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"📊 Loaded {len(data.get('frames', []))} frames")
                    
                    # Count shots in the data
                    shots = data.get('metadata', {}).get('shots', {})
                    
                    # Handle both list and dictionary formats for shots
                    if isinstance(shots, list):
                        shot_count = len(shots)
                        print(f"🎯 Found {shot_count} shots in the video")
                        
                        # Display shot information for list format
                        if shot_count > 0:
                            print("📋 Shot Information:")
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
                        print(f"🎯 Found {shot_count} shots in the video")
                        
                        # Display shot information for dictionary format
                        if shot_count > 0:
                            print("📋 Shot Information:")
                            for shot_id, shot_info in shots.items():
                                start_frame = shot_info.get('start_frame', 'N/A')
                                end_frame = shot_info.get('end_frame', 'N/A')
                                fixed_torso = shot_info.get('fixed_torso', 'N/A')
                                print(f"   {shot_id}: Frames {start_frame}-{end_frame}, Torso: {fixed_torso}")
                    
                    # Store metadata for analysis
                    if video_path == self.video1_path:
                        self.video1_metadata = data.get('metadata', {})
                    elif video_path == self.video2_path:
                        self.video2_metadata = data.get('metadata', {})
                    
                    return data
            except Exception as e:
                print(f"❌ Error loading existing results: {e}")
                return None
        
        # No processed data found
        print(f"❌ No processed data found for {base_name}")
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
    
    def perform_comparison(self, selected_shot1: Optional[str] = None, selected_shot2: Optional[str] = None) -> Optional[Dict]:
        """
        Perform DTW-based comparison between the two videos
        
        Args:
            selected_shot1: Selected shot from video 1 (None for all shots)
            selected_shot2: Selected shot from video 2 (None for all shots)
        
        Returns:
            Comparison results dictionary or None if failed
        """
        if not self.video1_data or not self.video2_data:
            print("❌ Video data not available for comparison")
            return None
            
        print("\n🔄 STEP 4: Performing DTW Comparison")
        print("=" * 50)
        
        try:
            print("🔍 [DEBUG] Starting perform_comparison...")
            
            # Initialize comparison results
            comparison_results = {}
            
            # Filter data by selected shots
            filtered_video1_data = self._filter_data_by_shot(self.video1_data, selected_shot1)
            filtered_video2_data = self._filter_data_by_shot(self.video2_data, selected_shot2)
            
            print("🔍 [DEBUG] Filtered data:")
            
            # Check if filtered data is None before accessing
            if filtered_video1_data is None:
                print("❌ Video 1 data filtering failed")
                return None
            if filtered_video2_data is None:
                print("❌ Video 2 data filtering failed")
                return None
                
            print(f"   Video 1: {len(filtered_video1_data.get('frames', []))} frames")
            print(f"   Video 2: {len(filtered_video2_data.get('frames', []))} frames")
            
            # Get selected hand
            selected_hand = filtered_video1_data.get('metadata', {}).get('hand', 'right')
            print(f"🔍 [DEBUG] Selected hand: {selected_hand}")
            self.selected_hand = selected_hand  # Store for analyzers
            # Perform Set-up phase analysis
            print("📊 Performing set-up phase analysis...")
            setup_analysis1 = self.setup_analyzer.analyze_setup_phase(filtered_video1_data)
            setup_analysis2 = self.setup_analyzer.analyze_setup_phase(filtered_video2_data)
            comparison_results['setup_analysis'] = {
                'video1': setup_analysis1,
                'video2': setup_analysis2
            }
            print("🔍 [DEBUG] Setup analysis completed")
            
            # Perform Loading phase analysis
            print("📊 Performing loading phase analysis...")
            loading_analysis1 = self.loading_analyzer.analyze_loading_phase(filtered_video1_data)
            loading_analysis2 = self.loading_analyzer.analyze_loading_phase(filtered_video2_data)
            comparison_results['loading_analysis'] = {
                'video1': loading_analysis1,
                'video2': loading_analysis2
            }
            print("🔍 [DEBUG] Loading analysis completed")
            
            # Perform Rising phase analysis
            print("📊 Performing rising phase analysis...")
            rising_analysis1 = self.rising_analyzer.analyze_rising_phase(filtered_video1_data, self.selected_hand)
            rising_analysis2 = self.rising_analyzer.analyze_rising_phase(filtered_video2_data, self.selected_hand)
            comparison_results['rising_analysis'] = {
                'video1': rising_analysis1,
                'video2': rising_analysis2
            }
            print("🔍 [DEBUG] Rising analysis completed")
            
            # Perform Release phase analysis
            print("📊 Performing release phase analysis...")
            release_analysis1 = self.release_analyzer.analyze_release_phase(filtered_video1_data, self.selected_hand)
            release_analysis2 = self.release_analyzer.analyze_release_phase(filtered_video2_data, self.selected_hand)
            comparison_results['release_analysis'] = {
                'video1': release_analysis1,
                'video2': release_analysis2
            }
            print("🔍 [DEBUG] Release analysis completed")
            
            # Perform Follow-through phase analysis
            print("📊 Performing follow-through phase analysis...")
            follow_through_analysis1 = self.follow_through_analyzer.analyze_follow_through_phase(filtered_video1_data, self.selected_hand)
            follow_through_analysis2 = self.follow_through_analyzer.analyze_follow_through_phase(filtered_video2_data, self.selected_hand)
            comparison_results['follow_through_analysis'] = {
                'video1': follow_through_analysis1,
                'video2': follow_through_analysis2
            }
            print("🔍 [DEBUG] Follow-through analysis completed")
            
            # Perform Landing phase analysis
            print("📊 Performing landing phase analysis...")
            landing_analysis1 = self.landing_analyzer.analyze_landing_phase(filtered_video1_data)
            landing_analysis2 = self.landing_analyzer.analyze_landing_phase(filtered_video2_data)
            comparison_results['landing_analysis'] = {
                'video1': landing_analysis1,
                'video2': landing_analysis2
            }
            print("🔍 [DEBUG] Landing analysis completed")
            
            # DTW Analysis temporarily disabled
            print("⏸️  DTW Analysis temporarily disabled - focusing on Set-up and Loading analysis")
            
            # # DTW Analysis (temporarily commented out)
            # print("📊 Performing coordinate-based overall comparison...")
            # overall_coord_result = self.dtw_processor.analyze_overall_phases_coordinate(
            #     filtered_video1_data, filtered_video2_data, selected_hand
            # )
            # comparison_results['coordinate_overall'] = overall_coord_result
            # print("🔍 [DEBUG] Overall coordinate comparison completed")
            
            # print("📊 Performing feature-based overall comparison...")
            # overall_feature_result = self.dtw_processor.analyze_overall_phases_feature(
            #     filtered_video1_data, filtered_video2_data, selected_hand
            # )
            # comparison_results['feature_overall'] = overall_feature_result
            # print("🔍 [DEBUG] Overall feature comparison completed")
            
            # # Phase-specific comparisons
            # phases = ['loading', 'rising', 'release', 'follow_through']
            
            # for phase in phases:
            #     print(f"📊 Performing {phase} phases coordinate comparison...")
            #     try:
            #         if phase == 'loading':
            #             coord_result = self.dtw_processor.analyze_loading_phases_coordinate(
            #                 filtered_video1_data, filtered_video2_data, selected_hand
            #             )
            #         elif phase == 'rising':
            #             coord_result = self.dtw_processor.analyze_rising_phases_coordinate(
            #                 filtered_video1_data, filtered_video2_data, selected_hand
            #             )
            #         elif phase == 'release':
            #             coord_result = self.dtw_processor.analyze_release_phases_coordinate(
            #                 filtered_video1_data, filtered_video2_data, selected_hand
            #             )
            #         elif phase == 'follow_through':
            #             coord_result = self.dtw_processor.analyze_follow_through_phases_coordinate(
            #                 filtered_video1_data, filtered_video2_data, selected_hand
            #             )
            #         comparison_results[f'{phase}_coordinate'] = coord_result
            #         print(f"🔍 [DEBUG] {phase} coordinate comparison completed")
            #     except Exception as e:
            #         print(f"⚠️  Error in {phase} coordinate comparison: {e}")
            #         comparison_results[f'{phase}_coordinate'] = None
                
            #     print(f"📊 Performing {phase} phases feature comparison...")
            #     try:
            #         if phase == 'loading':
            #             feature_result = self.dtw_processor.analyze_loading_phases(
            #                 filtered_video1_data, filtered_video2_data, selected_hand
            #             )
            #         elif phase == 'rising':
            #             feature_result = self.dtw_processor.analyze_rising_phases(
            #                 filtered_video1_data, filtered_video2_data, selected_hand
            #             )
            #         elif phase == 'release':
            #             feature_result = self.dtw_processor.analyze_release_phases(
            #                 filtered_video1_data, filtered_video2_data, selected_hand
            #             )
            #         elif phase == 'follow_through':
            #             feature_result = self.dtw_processor.analyze_follow_through_phases(
            #                 filtered_video1_data, filtered_video2_data, selected_hand
            #             )
            #         comparison_results[phase] = feature_result
            #         print(f"🔍 [DEBUG] {phase} feature comparison completed")
            #     except Exception as e:
            #         print(f"⚠️  Error in {phase} feature comparison: {e}")
            #         comparison_results[phase] = None
            
            print("🔍 [DEBUG] All phase comparisons completed")
            
            # Add metadata
            comparison_results['metadata'] = {
                'video1_path': self.video1_path,
                'video2_path': self.video2_path,
                'video1_frames': len(self.video1_data.get('frames', [])),
                'video2_frames': len(self.video2_data.get('frames', [])),
                'video1_fps': self.video1_metadata.get('fps', 30.0) if self.video1_metadata else 30.0,
                'video2_fps': self.video2_metadata.get('fps', 30.0) if self.video2_metadata else 30.0,
                'selected_hand': selected_hand,
                'selected_shot1': selected_shot1,
                'selected_shot2': selected_shot2,
                'comparison_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'video1_phase_transitions': self._extract_phase_transitions(self.video1_data),
                'video2_phase_transitions': self._extract_phase_transitions(self.video2_data)
            }
            print("🔍 [DEBUG] Metadata added")
            
            # Add phase statistics
            video1_phases = self.extract_phase_data(filtered_video1_data)
            video2_phases = self.extract_phase_data(filtered_video2_data)
            
            comparison_results['phase_statistics'] = {
                'video1_phases': {phase: len(frames) for phase, frames in video1_phases.items()},
                'video2_phases': {phase: len(frames) for phase, frames in video2_phases.items()}
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
            print(f"❌ Error during comparison: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Clean up temporary files
            pass
    
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
            return data  # Return all data
        
        frames = data.get('frames', [])
        shots = data.get('metadata', {}).get('shots', {})
        
        # Debug shot information
        print(f"🔍 Debug: Available shots: {shots}")
        print(f"🔍 Debug: Selected shot: {selected_shot}")
        print(f"🔍 Debug: Shots type: {type(shots)}")
        
        # Handle both list and dictionary formats for shots
        if isinstance(shots, list):
            # Convert shot number to index (e.g., "shot1" -> 0)
            if selected_shot.startswith("shot"):
                try:
                    shot_index = int(selected_shot[4:]) - 1  # "shot1" -> 0
                    if shot_index < 0 or shot_index >= len(shots):
                        print(f"❌ Selected shot '{selected_shot}' not found in data (index {shot_index} out of range)")
                        return None
                    shot_info = shots[shot_index]
                except ValueError:
                    print(f"❌ Invalid shot format: {selected_shot}")
                    return None
            else:
                print(f"❌ Invalid shot format: {selected_shot}")
                return None
        else:
            # Dictionary format
            if selected_shot not in shots:
                print(f"❌ Selected shot '{selected_shot}' not found in data")
                print(f"🔍 Debug: Available shot keys: {list(shots.keys()) if shots else 'None'}")
                return None
            shot_info = shots[selected_shot]
        
        start_frame = shot_info.get('start_frame', 0)
        end_frame = shot_info.get('end_frame', len(frames))
        
        print(f"🔍 Debug: Shot info - start_frame: {start_frame}, end_frame: {end_frame}")
        print(f"🔍 Debug: Total frames: {len(frames)}")
        
        # Filter frames that belong to the selected shot
        filtered_frames = []
        shot_id_count = 0
        range_count = 0
        
        for frame in frames:
            frame_idx = frame.get('frame_idx', 0)
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
            print(f"❌ No frames found for shot '{selected_shot}'")
            return None
        
        # Create filtered data
        filtered_data = data.copy()
        filtered_data['frames'] = filtered_frames
        print(filtered_data)
        print(f"✅ Filtered {len(filtered_frames)} frames for shot '{selected_shot}'")
        
        return filtered_data
    
    def save_comparison_results(self) -> bool:
        """
        Save comparison results to file
        
        Returns:
            True if successful, False otherwise
        """
        if not self.comparison_results:
            print("❌ No comparison results to save")
            return False
            
        print("\n💾 STEP 4: Saving Comparison Results")
        print("=" * 50)
        
        try:
            # Generate filename based on video names
            video1_name = os.path.splitext(os.path.basename(self.video1_path))[0]
            video2_name = os.path.splitext(os.path.basename(self.video2_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"comparison_{video1_name}_vs_{video2_name}_{timestamp}.json"
            output_path = os.path.join(self.comparison_results_dir, filename)
            
            # Add video paths and data paths to results for visualization
            enhanced_results = dict(self.comparison_results)
            print(self.comparison_results.keys())
            enhanced_results['video1_path'] = os.path.abspath(self.video1_path)
            enhanced_results['video2_path'] = os.path.abspath(self.video2_path)
            
            # Add data file paths if they exist
            video1_result_name = f"{video1_name}_normalized_output.json"
            video2_result_name = f"{video2_name}_normalized_output.json"
            video1_data_path = os.path.join(self.results_dir, video1_result_name)
            video2_data_path = os.path.join(self.results_dir, video2_result_name)
            
            if os.path.exists(video1_data_path):
                enhanced_results['video1_data_path'] = os.path.abspath(video1_data_path)
            if os.path.exists(video2_data_path):
                enhanced_results['video2_data_path'] = os.path.abspath(video2_data_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Comparison results saved: {filename}")
            print(f"📁 Location: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
    
    def print_comparison_summary(self):
        """Print a comprehensive summary of the comparison results."""
        try:
            print("[DEBUG] comparison_results keys:", self.comparison_results.keys())
            metadata = self.comparison_results.get('metadata')
            print("[DEBUG] metadata:", metadata)
            if not metadata:
                print("❌ metadata is None!")
                return
            print("[DEBUG] metadata keys:", metadata.keys() if isinstance(metadata, dict) else metadata)
            selected_hand = metadata.get('selected_hand', 'right')
            print(f"🖐 Selected Hand: {selected_hand}")
            print(f"📊 Video 1: {len(self.video1_data.get('frames', []))} frames @ {self.video1_metadata.get('fps', 30.0) if self.video1_metadata else 30.0}fps")
            print(f"📊 Video 2: {len(self.video2_data.get('frames', []))} frames @ {self.video2_metadata.get('fps', 30.0) if self.video2_metadata else 30.0}fps")

            video1_transitions = metadata.get('video1_phase_transitions')
            video2_transitions = metadata.get('video2_phase_transitions')
            print(f"[DEBUG] video1_transitions: {video1_transitions}")
            print(f"[DEBUG] video2_transitions: {video2_transitions}")
            if not video1_transitions or not video2_transitions:
                print("❌ One of the phase transitions is None!")
                return
            print(f"\n🔄 PHASE TRANSITION SEQUENCES:")
            print(f"   Video 1: {' → '.join(video1_transitions)}")
            print(f"   Video 2: {' → '.join(video2_transitions)}")

            # Display Set-up analysis
            if 'setup_analysis' in self.comparison_results:
                self._display_setup_analysis(self.comparison_results['setup_analysis'])
            
            # Display Loading analysis
            if 'loading_analysis' in self.comparison_results:
                self._display_loading_analysis(self.comparison_results['loading_analysis'])
            
            # Display Rising analysis
            if 'rising_analysis' in self.comparison_results:
                self._display_rising_analysis(self.comparison_results['rising_analysis'])
            
            # Display Release analysis
            if 'release_analysis' in self.comparison_results:
                self._display_release_analysis(self.comparison_results['release_analysis'])
            
            # Display Follow-through analysis
            if 'follow_through_analysis' in self.comparison_results:
                self._display_follow_through_analysis(self.comparison_results['follow_through_analysis'])
            
            # Display Landing analysis
            if 'landing_analysis' in self.comparison_results:
                self._display_landing_analysis(self.comparison_results['landing_analysis'])
            
            # Display interpretation results
            if 'interpretation' in self.comparison_results:
                self._display_interpretation_results(self.comparison_results['interpretation'])
            
            print("\n" + "=" * 50)
            print("✅ COMPARISON SUMMARY COMPLETE")
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Error printing summary: {e}")
            import traceback
            traceback.print_exc()
    
    def _display_phase_transition_analysis(self, phase_analysis: Dict):
        """Display phase transition analysis results."""
        print("\n🔄 PHASE TRANSITION ANALYSIS:")
        print("=" * 50)
        
        # Get metadata for phase transitions
        metadata = self.comparison_results.get('metadata', {})
        video1_transitions = metadata.get('video1_phase_transitions', [])
        video2_transitions = metadata.get('video2_phase_transitions', [])
        
        # Display Video 1 pattern
        video1_pattern = phase_analysis.get('video1_pattern', {})
        print(f"\n📹 Video 1 Phase Transitions:")
        print(f"   Sequence: {' → '.join(video1_transitions)}")
        print(f"   Pattern: {video1_pattern.get('pattern', 'unknown')}")
        print(f"   Description: {video1_pattern.get('description', 'No description')}")
        if video1_pattern.get('issues'):
            print("   Issues:")
            for issue in video1_pattern['issues']:
                print(f"     • {issue}")
        
        # Display Video 2 pattern
        video2_pattern = phase_analysis.get('video2_pattern', {})
        print(f"\n📹 Video 2 Phase Transitions:")
        print(f"   Sequence: {' → '.join(video2_transitions)}")
        print(f"   Pattern: {video2_pattern.get('pattern', 'unknown')}")
        print(f"   Description: {video2_pattern.get('description', 'No description')}")
        if video2_pattern.get('issues'):
            print("   Issues:")
            for issue in video2_pattern['issues']:
                print(f"     • {issue}")
        
        # Display comparison
        comparison = phase_analysis.get('comparison', {})
        print(f"\n🔄 Pattern Comparison:")
        print(f"   Similarity: {comparison.get('similarity', 'unknown')}")
        if comparison.get('differences'):
            print("   Differences:")
            for diff in comparison['differences']:
                print(f"     • {diff}")
        if comparison.get('recommendations'):
            print("   Recommendations:")
            for rec in comparison['recommendations']:
                print(f"     • {rec}")
        


    def _display_setup_analysis(self, setup_analysis: Dict):
        """Display set-up analysis results."""
        video1_setup = setup_analysis.get('video1', {})
        video2_setup = setup_analysis.get('video2', {})
        
        if 'error' in video1_setup or 'error' in video2_setup:
            print("  ⚠️  Set-up analysis errors:")
            if 'error' in video1_setup:
                print(f"    Video 1: {video1_setup['error']}")
            if 'error' in video2_setup:
                print(f"    Video 2: {video2_setup['error']}")
            return
        
        print("  📊 Video 1 Set-up Analysis:")
        print(f"    Frame Range: {video1_setup.get('frame_range', 'N/A')}")
        print(f"    Frame Count: {video1_setup.get('frame_count', 'N/A')}")
        print(f"    FPS: {video1_setup.get('fps', 'N/A')}")
        print(f"    Duration: {video1_setup.get('target_duration', 'N/A')}s")
        if 'note' in video1_setup:
            print(f"    Note: {video1_setup['note']}")
        
        # Display angles
        # angles1 = video1_setup.get('hip_knee_ankle_angles', {})
        # print(f"    Hip-Knee-Ankle Angles:")
        # print(f"      Left: {angles1.get('left', {}).get('average', 'Undefined')}°")
        # print(f"      Right: {angles1.get('right', {}).get('average', 'Undefined')}°")
        
        # Display foot positions
        foot_pos1 = video1_setup.get('foot_positions', {})
        left_foot1 = foot_pos1.get('left_foot', {})
        right_foot1 = foot_pos1.get('right_foot', {})
        print(f"    Foot Positions:")
        print(f"      Left: ({left_foot1.get('average_x', None)}, {left_foot1.get('average_y', None)})")
        print(f"      Right: ({right_foot1.get('average_x', None)}, {right_foot1.get('average_y', None)})")
        
        # Display shoulder tilt
        shoulder_tilt1 = video1_setup.get('shoulder_tilt', {})
        print(f"    Shoulder Tilt: {shoulder_tilt1.get('average', None)}°")
        
        # Display ball-hip distances
        ball_dist1 = video1_setup.get('ball_hip_distances', {})
        print(f"    Ball-Hip Distances:")
        print(f"      Vertical: {ball_dist1.get('average_vertical', None)}")
        print(f"      Horizontal: {ball_dist1.get('average_horizontal', None)}")
        
        print("  📊 Video 2 Set-up Analysis:")
        print(f"    Frame Range: {video2_setup.get('frame_range', None)}")
        print(f"    Frame Count: {video2_setup.get('frame_count', None)}")
        print(f"    FPS: {video2_setup.get('fps', None)}")
        print(f"    Duration: {video2_setup.get('target_duration', None)}s")
        if 'note' in video2_setup:
            print(f"    Note: {video2_setup['note']}")
        
        # Display angles
        # angles2 = video2_setup.get('hip_knee_ankle_angles', {})
        # print(f"    Hip-Knee-Ankle Angles:")
        # print(f"      Left: {angles2.get('left', {}).get('average', 'Undefined')}°")
        # print(f"      Right: {angles2.get('right', {}).get('average', 'Undefined')}°")
        
        # Display foot positions
        foot_pos2 = video2_setup.get('foot_positions', {})
        left_foot2 = foot_pos2.get('left_foot', {})
        right_foot2 = foot_pos2.get('right_foot', {})
        print(f"    Foot Positions:")
        print(f"      Left: ({left_foot2.get('average_x', None)}, {left_foot2.get('average_y', 'Undefined')})")
        print(f"      Right: ({right_foot2.get('average_x', 'Undefined')}, {right_foot2.get('average_y', 'Undefined')})")
        
        # Display shoulder tilt
        shoulder_tilt2 = video2_setup.get('shoulder_tilt', {})
        print(f"    Shoulder Tilt: {shoulder_tilt2.get('average', 'Undefined')}°")
        
        # Display ball-hip distances
        ball_dist2 = video2_setup.get('ball_hip_distances', {})
        print(f"    Ball-Hip Distances:")
        print(f"      Vertical: {ball_dist2.get('average_vertical', 'Undefined')}")
        print(f"      Horizontal: {ball_dist2.get('average_horizontal', 'Undefined')}")
    
    def _display_loading_analysis(self, loading_analysis: Dict):
        """Display loading analysis results."""
        video1_loading = loading_analysis.get('video1', {})
        video2_loading = loading_analysis.get('video2', {})
        
        if 'error' in video1_loading or 'error' in video2_loading:
            print("  ⚠️  Loading analysis errors:")
            if 'error' in video1_loading:
                print(f"    Video 1: {video1_loading['error']}")
            if 'error' in video2_loading:
                print(f"    Video 2: {video2_loading['error']}")
            return
        
        print("  📊 Video 1 Loading Analysis:")
        print(f"    Total Loading Time: {video1_loading.get('total_loading_time', 'N/A')}s")
        print(f"    Loading-Rising Time: {video1_loading.get('loading_rising_time', 'N/A')}s")
        print(f"    Loading Frames: {video1_loading.get('loading_frames', 'N/A')} (Loading phase only)")
        print(f"    Loading-Rising Frames: {video1_loading.get('loading_rising_frames', 'N/A')} (Loading-Rising phase only)")
        print(f"    Combined Loading Frames: {video1_loading.get('total_loading_frames', 'N/A')} (Loading + Loading-Rising)")
        
        # Display max leg angles
        leg_angles1 = video1_loading.get('max_leg_angles', {})
        print(f"    Max Leg Angles:")
        print(f"      Left: {leg_angles1.get('left', {}).get('max_angle', 'Undefined')}°")
        print(f"      Right: {leg_angles1.get('right', {}).get('max_angle', 'Undefined')}°")
        
        # Display max upper body tilt
        upper_body_tilt1 = video1_loading.get('max_upper_body_tilt', {})
        print(f"    Max Upper Body Tilt: {upper_body_tilt1.get('max_tilt', 'Undefined')}°")
        
        # Display max angle to transition timing
        max_angle_timing1 = video1_loading.get('max_angle_to_transition', {})
        if max_angle_timing1:
            print(f"    Max Angle to Transition:")
            print(f"      Left Max: {max_angle_timing1.get('left_max_angle', 'Undefined')}° (Frame {max_angle_timing1.get('left_max_frame', 'N/A')})")
            print(f"      Right Max: {max_angle_timing1.get('right_max_angle', 'Undefined')}° (Frame {max_angle_timing1.get('right_max_frame', 'N/A')})")
            print(f"      Both Max Frame: {max_angle_timing1.get('both_max_frame', 'N/A')}")
            print(f"      Time to Transition: {max_angle_timing1.get('time_to_transition', 'Undefined'):.3f}s")
            print(f"      Next Transition: {max_angle_timing1.get('next_transition_phase', 'Unknown')} (Frame {max_angle_timing1.get('next_transition_frame', 'N/A')})")
        
        print("  📊 Video 2 Loading Analysis:")
        print(f"    Total Loading Time: {video2_loading.get('total_loading_time', 'N/A')}s")
        print(f"    Loading-Rising Time: {video2_loading.get('loading_rising_time', 'N/A')}s")
        print(f"    Loading Frames: {video2_loading.get('loading_frames', 'N/A')} (Loading phase only)")
        print(f"    Loading-Rising Frames: {video2_loading.get('loading_rising_frames', 'N/A')} (Loading-Rising phase only)")
        print(f"    Combined Loading Frames: {video2_loading.get('total_loading_frames', 'N/A')} (Loading + Loading-Rising)")
        
        # Display max leg angles
        leg_angles2 = video2_loading.get('max_leg_angles', {})
        print(f"    Max Leg Angles:")
        print(f"      Left: {leg_angles2.get('left', {}).get('max_angle', 'Undefined')}°")
        print(f"      Right: {leg_angles2.get('right', {}).get('max_angle', 'Undefined')}°")
        
        # Display max upper body tilt
        upper_body_tilt2 = video2_loading.get('max_upper_body_tilt', {})
        print(f"    Max Upper Body Tilt: {upper_body_tilt2.get('max_tilt', 'Undefined')}°")
        
        # Display max angle to transition timing
        max_angle_timing2 = video2_loading.get('max_angle_to_transition', {})
        if max_angle_timing2:
            print(f"    Max Angle to Transition:")
            print(f"      Left Max: {max_angle_timing2.get('left_max_angle', 'Undefined')}° (Frame {max_angle_timing2.get('left_max_frame', 'N/A')})")
            print(f"      Right Max: {max_angle_timing2.get('right_max_angle', 'Undefined')}° (Frame {max_angle_timing2.get('right_max_frame', 'N/A')})")
            print(f"      Both Max Frame: {max_angle_timing2.get('both_max_frame', 'N/A')}")
            print(f"      Time to Transition: {max_angle_timing2.get('time_to_transition', 'Undefined'):.3f}s")
            print(f"      Next Transition: {max_angle_timing2.get('next_transition_phase', 'Unknown')} (Frame {max_angle_timing2.get('next_transition_frame', 'N/A')})")

    def _display_rising_analysis(self, rising_analysis: Dict):
        """Display rising analysis results."""
        video1_rising = rising_analysis.get('video1', {})
        video2_rising = rising_analysis.get('video2', {})

        if 'error' in video1_rising or 'error' in video2_rising:
            print("  ⚠️  Rising analysis errors:")
            if 'error' in video1_rising:
                print(f"    Video 1: {video1_rising['error']}")
            if 'error' in video2_rising:
                print(f"    Video 2: {video2_rising['error']}")
            return
        
        print("  📊 Video 1 Rising Analysis:")
        print(f"    Total Rising Time: {video1_rising.get('total_rising_time', 'N/A')}s")
        print(f"    Rising Frames: {video1_rising.get('rising_frames', 'N/A')}")
        print(f"    Loading-Rising Frames: {video1_rising.get('loading_rising_frames', 'N/A')}")
        print(f"    Combined Rising Frames: {video1_rising.get('total_rising_frames', 'N/A')}")
        
        # Windup trajectory analysis
        windup1 = video1_rising.get('windup_trajectory', {})
        if 'error' not in windup1:
            print(f"    Windup Trajectory:")
            print(f"      Dip Frame: {windup1.get('dip_frame', 'N/A')}")
            print(f"      Setup Frame: {windup1.get('setup_frame', 'N/A')}")
            print(f"      Trajectory Frames: {windup1.get('trajectory_frames', 'N/A')}")
            print(f"      Interpolated Frames: {windup1.get('interpolated_frames', 'N/A')}")
            print(f"      Trajectory Curvature: {windup1.get('trajectory_curvature', 'N/A'):.4f}")
            print(f"      Trajectory Path Length: {windup1.get('trajectory_path_length', 'N/A'):.4f}")
        
        # Jump analysis
        jump1 = video1_rising.get('jump_analysis', {})
        if 'error' not in jump1:
            print(f"    Jump Analysis:")
            print(f"      Max Jump Height: {jump1.get('max_jump_height', 'N/A'):.4f}")
            print(f"      Max Height Frame: {jump1.get('max_height_frame', 'N/A')}")
            print(f"      Max Height Time: {jump1.get('max_height_time', 'N/A'):.3f}s")
            print(f"      Setup Time: {jump1.get('setup_time', 'N/A'):.3f}s")
            print(f"      Relative Timing: {jump1.get('relative_timing', 'N/A'):.3f}s")
        
        # Body analysis
        body1 = video1_rising.get('body_analysis', {})
        if 'error' not in body1:
            print(f"    Body Analysis (at Max Jump Height):")
            print(f"      Body Tilt: {body1.get('body_tilt', 'N/A'):.2f}°")
            leg_angles1 = body1.get('leg_angles', {})
            print(f"      Left Thigh Angle: {leg_angles1.get('left_thigh_angle', 'N/A'):.2f}°")
            print(f"      Left Leg Angle: {leg_angles1.get('left_leg_angle', 'N/A'):.2f}°")
            print(f"      Right Thigh Angle: {leg_angles1.get('right_thigh_angle', 'N/A'):.2f}°")
            print(f"      Right Leg Angle: {leg_angles1.get('right_leg_angle', 'N/A'):.2f}°")
            print(f"      Max Jump Frame: {body1.get('max_jump_frame_index', 'N/A')}")
        
        # Timing analysis
        timing1 = video1_rising.get('timing_analysis', {})
        if 'error' not in timing1:
            print(f"    Timing Analysis:")
            print(f"      Windup Time: {timing1.get('windup_time', 'N/A'):.3f}s")
            print(f"      Total Rising Time: {timing1.get('total_rising_time', 'N/A'):.3f}s")
            print(f"      Windup Ratio: {timing1.get('windup_ratio', 'N/A'):.2f}")
        
        # Setup point analysis
        setup_point1 = video1_rising.get('setup_point_analysis', {})
        if 'error' not in setup_point1:
            print(f"    Setup Point Analysis:")
            print(f"      Setup Frame: {setup_point1.get('setup_frame_index', 'N/A')}")
            
            # Arm angles at setup point
            arm_angles1 = setup_point1.get('arm_angles', {})
            if arm_angles1:
                print(f"      Arm Angles:")
                print(f"        Shoulder-Elbow-Wrist: {arm_angles1.get('shoulder_elbow_wrist', 'N/A'):.2f}°")
                print(f"        Elbow-Shoulder-Hip: {arm_angles1.get('elbow_shoulder_hip', 'N/A'):.2f}°")
                print(f"        Torso Angle: {arm_angles1.get('torso_angle', 'N/A'):.2f}°")
                print(f"        Arm-Torso Angle: {arm_angles1.get('arm_torso_angle', 'N/A'):.2f}°")
            
            # Ball position relative to eyes at setup point
            ball_eye1 = setup_point1.get('ball_eye_position', {})
            if ball_eye1 and 'error' not in ball_eye1:
                print(f"      Ball Position (Relative to Eyes):")
                print(f"        Horizontal Distance: {ball_eye1.get('relative_x', 'N/A'):.4f}")
                print(f"        Vertical Distance: {ball_eye1.get('relative_y', 'N/A'):.4f}")
        
        print("  📊 Video 2 Rising Analysis:")
        print(f"    Total Rising Time: {video2_rising.get('total_rising_time', 'N/A')}s")
        print(f"    Rising Frames: {video2_rising.get('rising_frames', 'N/A')}")
        print(f"    Loading-Rising Frames: {video2_rising.get('loading_rising_frames', 'N/A')}")
        print(f"    Combined Rising Frames: {video2_rising.get('total_rising_frames', 'N/A')}")
        
        dip_point1 = video1_rising.get('dip_point_analysis', {})
        if 'error' not in dip_point1:
            print(f"    Dip Point Analysis:")
            print(f"      Dip Frame: {dip_point1.get('dip_frame_index', 'N/A')}")
            print(f"      Arm Angles:")
            print(f"        Shoulder-Elbow-Wrist: {dip_point1.get('dip_shoulder_elbow_wrist', 'N/A'):.2f}°")
            print(f"        Elbow-Shoulder-Hip: {dip_point1.get('dip_elbow_shoulder_hip', 'N/A'):.2f}°")
            print(f"        Torso Angle: {dip_point1.get('dip_torso_angle', 'N/A'):.2f}°")
            print(f"        Arm-Torso Angle: {dip_point1.get('dip_arm_torso_angle', 'N/A'):.2f}°")
            
            # Ball position relative to eyes at setup point
            ball_eye1 = setup_point1.get('dip_ball_eye_position', {})
            if ball_eye1 and 'error' not in ball_eye1:
                print(f"      Ball Position (Relative to Eyes):")
                print(f"        Horizontal Distance: {ball_eye1.get('relative_x', 'N/A'):.4f}")
                print(f"        Vertical Distance: {ball_eye1.get('relative_y', 'N/A'):.4f}")
        
        print("  📊 Video 2 Rising Analysis:")
        print(f"    Total Rising Time: {video2_rising.get('total_rising_time', 'N/A')}s")
        print(f"    Rising Frames: {video2_rising.get('rising_frames', 'N/A')}")
        print(f"    Loading-Rising Frames: {video2_rising.get('loading_rising_frames', 'N/A')}")
        print(f"    Combined Rising Frames: {video2_rising.get('total_rising_frames', 'N/A')}")

        # Windup trajectory analysis
        windup2 = video2_rising.get('windup_trajectory', {})
        if 'error' not in windup2:
            print(f"    Windup Trajectory:")
            print(f"      Dip Frame: {windup2.get('dip_frame', 'N/A')}")
            print(f"      Setup Frame: {windup2.get('setup_frame', 'N/A')}")
            print(f"      Trajectory Frames: {windup2.get('trajectory_frames', 'N/A')}")
            print(f"      Interpolated Frames: {windup2.get('interpolated_frames', 'N/A')}")
            print(f"      Trajectory Curvature: {windup2.get('trajectory_curvature', 'N/A'):.4f}")
            print(f"      Trajectory Path Length: {windup2.get('trajectory_path_length', 'N/A'):.4f}")
        
        # Jump analysis
        jump2 = video2_rising.get('jump_analysis', {})
        if 'error' not in jump2:
            print(f"    Jump Analysis:")
            print(f"      Max Jump Height: {jump2.get('max_jump_height', 'N/A'):.4f}")
            print(f"      Max Height Frame: {jump2.get('max_height_frame', 'N/A')}")
            print(f"      Max Height Time: {jump2.get('max_height_time', 'N/A'):.3f}s")
            print(f"      Setup Time: {jump2.get('setup_time', 'N/A'):.3f}s")
            print(f"      Relative Timing: {jump2.get('relative_timing', 'N/A'):.3f}s")
        
        # Body analysis
        body2 = video2_rising.get('body_analysis', {})
        if 'error' not in body2:
            print(f"    Body Analysis (at Max Jump Height):")
            print(f"      Body Tilt: {body2.get('body_tilt', 'N/A'):.2f}°")
            leg_angles2 = body2.get('leg_angles', {})
            print(f"      Left Thigh Angle: {leg_angles2.get('left_thigh_angle', 'N/A'):.2f}°")
            print(f"      Left Leg Angle: {leg_angles2.get('left_leg_angle', 'N/A'):.2f}°")
            print(f"      Right Thigh Angle: {leg_angles2.get('right_thigh_angle', 'N/A'):.2f}°")
            print(f"      Right Leg Angle: {leg_angles2.get('right_leg_angle', 'N/A'):.2f}°")
            print(f"      Max Jump Frame: {body2.get('max_jump_frame_index', 'N/A')}")
        
        # Timing analysis
        timing2 = video2_rising.get('timing_analysis', {})
        if 'error' not in timing2:
            print(f"    Timing Analysis:")
            print(f"      Windup Time: {timing2.get('windup_time', 'N/A'):.3f}s")
            print(f"      Total Rising Time: {timing2.get('total_rising_time', 'N/A'):.3f}s")
            print(f"      Windup Ratio: {timing2.get('windup_ratio', 'N/A'):.2f}")
        
        # Setup point analysis
        setup_point2 = video2_rising.get('setup_point_analysis', {})
        if 'error' not in setup_point2:
            print(f"    Setup Point Analysis:")
            print(f"      Setup Frame: {setup_point2.get('setup_frame_index', 'N/A')}")
            
            # Arm angles at setup point
            arm_angles2 = setup_point2.get('arm_angles', {})
            if arm_angles2:
                print(f"      Arm Angles:")
                print(f"        Shoulder-Elbow-Wrist: {arm_angles2.get('shoulder_elbow_wrist', 'N/A'):.2f}°")
                print(f"        Elbow-Shoulder-Hip: {arm_angles2.get('elbow_shoulder_hip', 'N/A'):.2f}°")
                print(f"        Torso Angle: {arm_angles2.get('torso_angle', 'N/A'):.2f}°")
                print(f"        Arm-Torso Angle: {arm_angles2.get('arm_torso_angle', 'N/A'):.2f}°")
            
            # Ball position relative to eyes at setup point
            ball_eye2 = setup_point2.get('ball_eye_position', {})
            if ball_eye2 and 'error' not in ball_eye2:
                print(f"      Ball Position (Relative to Eyes):")
                print(f"        Horizontal Distance: {ball_eye2.get('relative_x', 'N/A'):.4f}")
                print(f"        Vertical Distance: {ball_eye2.get('relative_y', 'N/A'):.4f}")
        dip_point2 = video2_rising.get('dip_point_analysis', {})
        if 'error' not in dip_point2:
            print(f"    Dip Point Analysis:")
            print(f"      Dip Frame: {dip_point2.get('dip_frame_index', 'N/A')}")
            print(f"      Arm Angles:")
            print(f"        Shoulder-Elbow-Wrist: {dip_point2.get('dip_shoulder_elbow_wrist', 'N/A'):.2f}°")
            print(f"        Elbow-Shoulder-Hip: {dip_point2.get('dip_elbow_shoulder_hip', 'N/A'):.2f}°")
            print(f"        Torso Angle: {dip_point2.get('dip_torso_angle', 'N/A'):.2f}°")
            print(f"        Arm-Torso Angle: {dip_point2.get('dip_arm_torso_angle', 'N/A'):.2f}°")
            
            # Ball position relative to eyes at setup point
            ball_eye1 = setup_point1.get('dip_ball_eye_position', {})
            if ball_eye1 and 'error' not in ball_eye1:
                print(f"      Ball Position (Relative to Eyes):")
                print(f"        Horizontal Distance: {ball_eye1.get('relative_x', 'N/A'):.4f}")
                print(f"        Vertical Distance: {ball_eye1.get('relative_y', 'N/A'):.4f}")

    def _display_release_analysis(self, release_analysis: Dict):
        """Display release analysis results."""
        video1_release = release_analysis.get('video1', {})
        video2_release = release_analysis.get('video2', {})
        
        if 'error' in video1_release or 'error' in video2_release:
            print("  ⚠️  Release analysis errors:")
            if 'error' in video1_release:
                print(f"    Video 1: {video1_release['error']}")
            if 'error' in video2_release:
                print(f"    Video 2: {video2_release['error']}")
            return
        
        print("  📊 Video 1 Release Analysis:")
        print(f"    Total Release Time: {video1_release.get('total_release_time', 'N/A')}s")
        print(f"    Release Frames: {video1_release.get('total_release_frames', 'N/A')}")
        
        # Arm angles analysis
        arm_angles1 = video1_release.get('arm_angles', {})
        if 'error' not in arm_angles1:
            print(f"    Arm Angles:")
            left_arm1 = arm_angles1.get('left_arm', {})
            right_arm1 = arm_angles1.get('right_arm', {})
            print(f"      Left Arm - Torso: {left_arm1.get('torso_angle', {}).get('average', 'N/A'):.2f}°")
            print(f"      Left Arm - Vertical: {left_arm1.get('vertical_angle', {}).get('average', 'N/A'):.2f}°")
            print(f"      Right Arm - Torso: {right_arm1.get('torso_angle', {}).get('average', 'N/A'):.2f}°")
            print(f"      Right Arm - Vertical: {right_arm1.get('vertical_angle', {}).get('average', 'N/A'):.2f}°")
        
        # Ball position analysis
        ball_position1 = video1_release.get('ball_position', {})
        if 'error' not in ball_position1:
            print(f"    Ball Position (Relative to Eyes):")
            print(f"      Average X: {ball_position1.get('average_relative_x', 'N/A'):.4f}")
            print(f"      Average Y: {ball_position1.get('average_relative_y', 'N/A'):.4f}")
            print(f"      X Range: {ball_position1.get('min_relative_x', 'N/A'):.4f} to {ball_position1.get('max_relative_x', 'N/A'):.4f}")
            print(f"      Y Range: {ball_position1.get('min_relative_y', 'N/A'):.4f} to {ball_position1.get('max_relative_y', 'N/A'):.4f}")
            
            # Ball vector analysis
            ball_vector1 = ball_position1.get('ball_vector', {})
            if ball_vector1:
                print(f"    Ball Vector:")
                print(f"      Average Magnitude: {ball_vector1.get('average_magnitude', 'N/A'):.4f}")
                print(f"      Average Angle: {ball_vector1.get('average_angle', 'N/A'):.2f}°")
                print(f"      Average Velocity X: {ball_vector1.get('average_velocity_x', 'N/A'):.4f}")
                print(f"      Average Velocity Y: {ball_vector1.get('average_velocity_y', 'N/A'):.4f}")
                print(f"      Magnitude Range: {ball_vector1.get('min_magnitude', 'N/A'):.4f} to {ball_vector1.get('max_magnitude', 'N/A'):.4f}")
                print(f"      Angle Range: {ball_vector1.get('min_angle', 'N/A'):.2f}° to {ball_vector1.get('max_angle', 'N/A'):.2f}°")
        
        # Body analysis
        body1 = video1_release.get('body_analysis', {})
        if 'error' not in body1:
            print(f"    Body Analysis:")
            body_tilt1 = body1.get('body_tilt', {})
            print(f"      Body Tilt: {body_tilt1.get('average', 'N/A'):.2f}°")
            leg_angles1 = body1.get('leg_angles', {})
            print(f"      Left Thigh Angle: {leg_angles1.get('left_thigh_angle', {}).get('average', 'N/A'):.2f}°")
            print(f"      Left Leg Angle: {leg_angles1.get('left_leg_angle', {}).get('average', 'N/A'):.2f}°")
            print(f"      Right Thigh Angle: {leg_angles1.get('right_thigh_angle', {}).get('average', 'N/A'):.2f}°")
            print(f"      Right Leg Angle: {leg_angles1.get('right_leg_angle', {}).get('average', 'N/A'):.2f}°")
        
        print("  📊 Video 2 Release Analysis:")
        print(f"    Total Release Time: {video2_release.get('total_release_time', 'N/A')}s")
        print(f"    Release Frames: {video2_release.get('total_release_frames', 'N/A')}")
        
        # Arm angles analysis
        arm_angles2 = video2_release.get('arm_angles', {})
        if 'error' not in arm_angles2:
            print(f"    Arm Angles:")
            left_arm2 = arm_angles2.get('left_arm', {})
            right_arm2 = arm_angles2.get('right_arm', {})
            print(f"      Left Arm - Torso: {left_arm2.get('torso_angle', {}).get('average', 'N/A'):.2f}°")
            print(f"      Left Arm - Vertical: {left_arm2.get('vertical_angle', {}).get('average', 'N/A'):.2f}°")
            print(f"      Right Arm - Torso: {right_arm2.get('torso_angle', {}).get('average', 'N/A'):.2f}°")
            print(f"      Right Arm - Vertical: {right_arm2.get('vertical_angle', {}).get('average', 'N/A'):.2f}°")
        
        # Ball position analysis
        ball_position2 = video2_release.get('ball_position', {})
        if 'error' not in ball_position2:
            print(f"    Ball Position (Relative to Eyes):")
            print(f"      Average X: {ball_position2.get('average_relative_x', 'N/A'):.4f}")
            print(f"      Average Y: {ball_position2.get('average_relative_y', 'N/A'):.4f}")
            print(f"      X Range: {ball_position2.get('min_relative_x', 'N/A'):.4f} to {ball_position2.get('max_relative_x', 'N/A'):.4f}")
            print(f"      Y Range: {ball_position2.get('min_relative_y', 'N/A'):.4f} to {ball_position2.get('max_relative_y', 'N/A'):.4f}")
            
            # Ball vector analysis
            ball_vector2 = ball_position2.get('ball_vector', {})
            if ball_vector2:
                print(f"    Ball Vector:")
                print(f"      Average Magnitude: {ball_vector2.get('average_magnitude', 'N/A'):.4f}")
                print(f"      Average Angle: {ball_vector2.get('average_angle', 'N/A'):.2f}°")
                print(f"      Average Velocity X: {ball_vector2.get('average_velocity_x', 'N/A'):.4f}")
                print(f"      Average Velocity Y: {ball_vector2.get('average_velocity_y', 'N/A'):.4f}")
                print(f"      Magnitude Range: {ball_vector2.get('min_magnitude', 'N/A'):.4f} to {ball_vector2.get('max_magnitude', 'N/A'):.4f}")
                print(f"      Angle Range: {ball_vector2.get('min_angle', 'N/A'):.2f}° to {ball_vector2.get('max_angle', 'N/A'):.2f}°")
        
        # Body analysis
        body2 = video2_release.get('body_analysis', {})
        if 'error' not in body2:
            print(f"    Body Analysis:")
            body_tilt2 = body2.get('body_tilt', {})
            print(f"      Body Tilt: {body_tilt2.get('average', 'N/A'):.2f}°")
            leg_angles2 = body2.get('leg_angles', {})
            print(f"      Left Thigh Angle: {leg_angles2.get('left_thigh_angle', {}).get('average', 'N/A'):.2f}°")
            print(f"      Left Leg Angle: {leg_angles2.get('left_leg_angle', {}).get('average', 'N/A'):.2f}°")
            print(f"      Right Thigh Angle: {leg_angles2.get('right_thigh_angle', {}).get('average', 'N/A'):.2f}°")
            print(f"      Right Leg Angle: {leg_angles2.get('right_leg_angle', {}).get('average', 'N/A'):.2f}°")

    def _display_follow_through_analysis(self, follow_through_analysis: Dict):
        """Display follow-through analysis results."""
        video1_follow_through = follow_through_analysis.get('video1', {})
        video2_follow_through = follow_through_analysis.get('video2', {})
        
        if 'error' in video1_follow_through or 'error' in video2_follow_through:
            print("  ⚠️  Follow-through analysis errors:")
            if 'error' in video1_follow_through:
                print(f"    Video 1: {video1_follow_through['error']}")
            if 'error' in video2_follow_through:
                print(f"    Video 2: {video2_follow_through['error']}")
            return
        
        print("  📊 Video 1 Follow-through Analysis:")
        print(f"    Total Follow-through Time: {video1_follow_through.get('total_follow_through_time', 'N/A')}s")
        print(f"    Follow-through Frames: {video1_follow_through.get('total_follow_through_frames', 'N/A')}")
        
        # Max elbow angle analysis
        max_elbow1 = video1_follow_through.get('max_elbow_angle_analysis', {})
        if 'error' not in max_elbow1:
            print(f"    Max Elbow Angle Analysis:")
            print(f"      Max Elbow Angle: {max_elbow1.get('max_elbow_angle', 'N/A'):.2f}°")
            print(f"      Max Elbow Frame Index: {max_elbow1.get('max_elbow_frame_idx', 'N/A')}")
            print(f"      Arm Angles Std: {max_elbow1.get('arm_angles_std', 'N/A'):.2f}°")
            print(f"      Body Angles Std: {max_elbow1.get('body_angles_std', 'N/A'):.2f}°")
            print(f"      Leg Angles Std: {max_elbow1.get('leg_angles_std', 'N/A'):.2f}°")
            print(f"      Overall Angles Std: {max_elbow1.get('overall_angles_std', 'N/A'):.2f}°")
        
        # Stability analysis
        stability1 = video1_follow_through.get('stability_analysis', {})
        if 'error' not in stability1:
            print(f"    Stability Analysis:")
            print(f"      Stability Threshold: {stability1.get('stability_threshold', 'N/A')}°")
            print(f"      Overall Stable Duration: {stability1.get('overall_stable_duration', 'N/A'):.3f}s")
            print(f"      Arm Stable Duration: {stability1.get('arm_stable_duration', 'N/A'):.3f}s")
            print(f"      Other Body Stable Duration: {stability1.get('other_body_stable_duration', 'N/A'):.3f}s")
            print(f"      Stability Percentage: {stability1.get('stability_percentage', 'N/A'):.1f}%")
            print(f"      Arm Stability Percentage: {stability1.get('arm_stability_percentage', 'N/A'):.1f}%")
            print(f"      Other Stability Percentage: {stability1.get('other_stability_percentage', 'N/A'):.1f}%")
        
        # Arm stability analysis
        arm_stability1 = video1_follow_through.get('arm_stability', {})
        if 'error' not in arm_stability1:
            print(f"    Arm Stability:")
            shoulder1 = arm_stability1.get('shoulder_angle', {})
            elbow1 = arm_stability1.get('elbow_angle', {})
            # wrist1 = arm_stability1.get('wrist_angle', {})
            print(f"      Shoulder Angle - Avg: {shoulder1.get('average', 'N/A'):.2f}°, Std: {shoulder1.get('std', 'N/A'):.2f}°")
            print(f"      Elbow Angle - Avg: {elbow1.get('average', 'N/A'):.2f}°, Std: {elbow1.get('std', 'N/A'):.2f}°")
            # print(f"      Wrist Angle - Avg: {wrist1.get('average', 'N/A'):.2f}°, Std: {wrist1.get('std', 'N/A'):.2f}°")
        
        # Overall stability analysis
        overall_stability1 = video1_follow_through.get('overall_stability', {})
        if 'error' not in overall_stability1:
            print(f"    Overall Stability:")
            hip1 = overall_stability1.get('hip_angle', {})
            knee1 = overall_stability1.get('knee_angle', {})
            # ankle1 = overall_stability1.get('ankle_angle', {})
            torso1 = overall_stability1.get('torso_angle', {})
            print(f"      Hip Angle - Avg: {hip1.get('average', 'N/A'):.2f}°, Std: {hip1.get('std', 'N/A'):.2f}°")
            print(f"      Knee Angle - Avg: {knee1.get('average', 'N/A'):.2f}°, Std: {knee1.get('std', 'N/A'):.2f}°")
            # print(f"      Ankle Angle - Avg: {ankle1.get('average', 'N/A'):.2f}°, Std: {ankle1.get('std', 'N/A'):.2f}°")
            print(f"      Torso Angle - Avg: {torso1.get('average', 'N/A'):.2f}°, Std: {torso1.get('std', 'N/A'):.2f}°")
        
        print("  📊 Video 2 Follow-through Analysis:")
        print(f"    Total Follow-through Time: {video2_follow_through.get('total_follow_through_time', 'N/A')}s")
        print(f"    Follow-through Frames: {video2_follow_through.get('total_follow_through_frames', 'N/A')}")
        
        # Max elbow angle analysis
        max_elbow2 = video2_follow_through.get('max_elbow_angle_analysis', {})
        if 'error' not in max_elbow2:
            print(f"    Max Elbow Angle Analysis:")
            print(f"      Max Elbow Angle: {max_elbow2.get('max_elbow_angle', 'N/A'):.2f}°")
            print(f"      Max Elbow Frame Index: {max_elbow2.get('max_elbow_frame_idx', 'N/A')}")
            print(f"      Arm Angles Std: {max_elbow2.get('arm_angles_std', 'N/A'):.2f}°")
            print(f"      Body Angles Std: {max_elbow2.get('body_angles_std', 'N/A'):.2f}°")
            print(f"      Leg Angles Std: {max_elbow2.get('leg_angles_std', 'N/A'):.2f}°")
            print(f"      Overall Angles Std: {max_elbow2.get('overall_angles_std', 'N/A'):.2f}°")
        
        # Stability analysis
        stability2 = video2_follow_through.get('stability_analysis', {})
        if 'error' not in stability2:
            print(f"    Stability Analysis:")
            print(f"      Stability Threshold: {stability2.get('stability_threshold', 'N/A')}°")
            print(f"      Overall Stable Duration: {stability2.get('overall_stable_duration', 'N/A'):.3f}s")
            print(f"      Arm Stable Duration: {stability2.get('arm_stable_duration', 'N/A'):.3f}s")
            print(f"      Other Body Stable Duration: {stability2.get('other_body_stable_duration', 'N/A'):.3f}s")
            print(f"      Stability Percentage: {stability2.get('stability_percentage', 'N/A'):.1f}%")
            print(f"      Arm Stability Percentage: {stability2.get('arm_stability_percentage', 'N/A'):.1f}%")
            print(f"      Other Stability Percentage: {stability2.get('other_stability_percentage', 'N/A'):.1f}%")
        
        # Arm stability analysis
        arm_stability2 = video2_follow_through.get('arm_stability', {})
        if 'error' not in arm_stability2:
            print(f"    Arm Stability:")
            shoulder2 = arm_stability2.get('shoulder_angle', {})
            elbow2 = arm_stability2.get('elbow_angle', {})
            # wrist2 = arm_stability2.get('wrist_angle', {})
            print(f"      Shoulder Angle - Avg: {shoulder2.get('average', 'N/A'):.2f}°, Std: {shoulder2.get('std', 'N/A'):.2f}°")
            print(f"      Elbow Angle - Avg: {elbow2.get('average', 'N/A'):.2f}°, Std: {elbow2.get('std', 'N/A'):.2f}°")
            # print(f"      Wrist Angle - Avg: {wrist2.get('average', 'N/A'):.2f}°, Std: {wrist2.get('std', 'N/A'):.2f}°")
        
        # Overall stability analysis
        overall_stability2 = video2_follow_through.get('overall_stability', {})
        if 'error' not in overall_stability2:
            print(f"    Overall Stability:")
            hip2 = overall_stability2.get('hip_angle', {})
            knee2 = overall_stability2.get('knee_angle', {})
            # ankle2 = overall_stability2.get('ankle_angle', {})
            torso2 = overall_stability2.get('torso_angle', {})
            print(f"      Hip Angle - Avg: {hip2.get('average', 'N/A'):.2f}°, Std: {hip2.get('std', 'N/A'):.2f}°")
            print(f"      Knee Angle - Avg: {knee2.get('average', 'N/A'):.2f}°, Std: {knee2.get('std', 'N/A'):.2f}°")
            # print(f"      Ankle Angle - Avg: {ankle2.get('average', 'N/A'):.2f}°, Std: {ankle2.get('std', 'N/A'):.2f}°")
            print(f"      Torso Angle - Avg: {torso2.get('average', 'N/A'):.2f}°, Std: {torso2.get('std', 'N/A'):.2f}°")

    def _display_landing_analysis(self, landing_analysis: Dict):
        """Display landing analysis results."""
        video1_landing = landing_analysis.get('video1', {})
        video2_landing = landing_analysis.get('video2', {})
        
        if 'error' in video1_landing or 'error' in video2_landing:
            print("  ⚠️  Landing analysis errors:")
            if 'error' in video1_landing:
                print(f"    Video 1: {video1_landing['error']}")
            if 'error' in video2_landing:
                print(f"    Video 2: {video2_landing['error']}")
            return
        
        print("  📊 Video 1 Landing Analysis:")
        print(f"    Follow-through Frames: {video1_landing.get('follow_through_frames_count', 'N/A')}")
        print(f"    Post Follow-through Frames: {video1_landing.get('post_follow_through_frames_count', 'N/A')}")
        
        # Landing detection
        landing_detection1 = video1_landing.get('landing_detection', {})
        if 'error' not in landing_detection1:
            print(f"    Landing Detection:")
            print(f"      Landing Detected: {landing_detection1.get('landing_detected', 'N/A')}")
            print(f"      Stable Landing: {landing_detection1.get('stable_landing', 'N/A')}")
            print(f"      Landing Frame Index: {landing_detection1.get('landing_frame_idx', 'N/A')}")
            print(f"      Landing Threshold: {landing_detection1.get('landing_threshold', 'N/A')}")
            print(f"      Stability Threshold: {landing_detection1.get('stability_threshold', 'N/A')}")
        
        # Landing position analysis
        landing_position1 = video1_landing.get('landing_position_analysis', {})
        if 'error' not in landing_position1:
            print(f"    Landing Position Analysis:")
            left_foot1 = landing_position1.get('left_foot', {})
            right_foot1 = landing_position1.get('right_foot', {})
            print(f"      Left Foot - Avg Distance from Setup: {left_foot1.get('average_distance_from_setup', 'N/A'):.4f}")
            print(f"      Left Foot - Min Distance: {left_foot1.get('min_distance_from_setup', 'N/A'):.4f}")
            print(f"      Left Foot - Max Distance: {left_foot1.get('max_distance_from_setup', 'N/A'):.4f}")
            print(f"      Left Foot - Std Distance: {left_foot1.get('std_distance_from_setup', 'N/A'):.4f}")
            print(f"      Right Foot - Avg Distance from Setup: {right_foot1.get('average_distance_from_setup', 'N/A'):.4f}")
            print(f"      Right Foot - Min Distance: {right_foot1.get('min_distance_from_setup', 'N/A'):.4f}")
            print(f"      Right Foot - Max Distance: {right_foot1.get('max_distance_from_setup', 'N/A'):.4f}")
            print(f"      Right Foot - Std Distance: {right_foot1.get('std_distance_from_setup', 'N/A'):.4f}")
        
        # Landing torso analysis
        landing_torso1 = video1_landing.get('landing_torso_analysis', {})
        if 'error' not in landing_torso1:
            print(f"    Landing Torso Analysis:")
            print(f"      Average Torso Angle: {landing_torso1.get('average_torso_angle', 'N/A'):.2f}°")
            print(f"      Std Torso Angle: {landing_torso1.get('std_torso_angle', 'N/A'):.2f}°")
            print(f"      Min Torso Angle: {landing_torso1.get('min_torso_angle', 'N/A'):.2f}°")
            print(f"      Max Torso Angle: {landing_torso1.get('max_torso_angle', 'N/A'):.2f}°")
        
        # Landing timing
        landing_timing1 = video1_landing.get('landing_timing', {})
        if 'error' not in landing_timing1:
            print(f"    Landing Timing:")
            print(f"      Follow-through Duration: {landing_timing1.get('follow_through_duration', 'N/A'):.3f}s")
            print(f"      Post Follow-through Duration: {landing_timing1.get('post_follow_through_duration', 'N/A'):.3f}s")
            print(f"      Total Analysis Duration: {landing_timing1.get('total_analysis_duration', 'N/A'):.3f}s")
        
        print("  📊 Video 2 Landing Analysis:")
        print(f"    Follow-through Frames: {video2_landing.get('follow_through_frames_count', 'N/A')}")
        print(f"    Post Follow-through Frames: {video2_landing.get('post_follow_through_frames_count', 'N/A')}")
        
        # Landing detection
        landing_detection2 = video2_landing.get('landing_detection', {})
        if 'error' not in landing_detection2:
            print(f"    Landing Detection:")
            print(f"      Landing Detected: {landing_detection2.get('landing_detected', 'N/A')}")
            print(f"      Stable Landing: {landing_detection2.get('stable_landing', 'N/A')}")
            print(f"      Landing Frame Index: {landing_detection2.get('landing_frame_idx', 'N/A')}")
            print(f"      Landing Threshold: {landing_detection2.get('landing_threshold', 'N/A')}")
            print(f"      Stability Threshold: {landing_detection2.get('stability_threshold', 'N/A')}")
        
        # Landing position analysis
        landing_position2 = video2_landing.get('landing_position_analysis', {})
        if 'error' not in landing_position2:
            print(f"    Landing Position Analysis:")
            left_foot2 = landing_position2.get('left_foot', {})
            right_foot2 = landing_position2.get('right_foot', {})
            print(f"      Left Foot - Avg Distance from Setup: {left_foot2.get('average_distance_from_setup', 'N/A'):.4f}")
            print(f"      Left Foot - Min Distance: {left_foot2.get('min_distance_from_setup', 'N/A'):.4f}")
            print(f"      Left Foot - Max Distance: {left_foot2.get('max_distance_from_setup', 'N/A'):.4f}")
            print(f"      Left Foot - Std Distance: {left_foot2.get('std_distance_from_setup', 'N/A'):.4f}")
            print(f"      Right Foot - Avg Distance from Setup: {right_foot2.get('average_distance_from_setup', 'N/A'):.4f}")
            print(f"      Right Foot - Min Distance: {right_foot2.get('min_distance_from_setup', 'N/A'):.4f}")
            print(f"      Right Foot - Max Distance: {right_foot2.get('max_distance_from_setup', 'N/A'):.4f}")
            print(f"      Right Foot - Std Distance: {right_foot2.get('std_distance_from_setup', 'N/A'):.4f}")
        
        # Landing torso analysis
        landing_torso2 = video2_landing.get('landing_torso_analysis', {})
        if 'error' not in landing_torso2:
            print(f"    Landing Torso Analysis:")
            print(f"      Average Torso Angle: {landing_torso2.get('average_torso_angle', 'N/A'):.2f}°")
            print(f"      Std Torso Angle: {landing_torso2.get('std_torso_angle', 'N/A'):.2f}°")
            print(f"      Min Torso Angle: {landing_torso2.get('min_torso_angle', 'N/A'):.2f}°")
            print(f"      Max Torso Angle: {landing_torso2.get('max_torso_angle', 'N/A'):.2f}°")
        
        # Landing timing
        landing_timing2 = video2_landing.get('landing_timing', {})
        if 'error' not in landing_timing2:
            print(f"    Landing Timing:")
            print(f"      Follow-through Duration: {landing_timing2.get('follow_through_duration', 'N/A'):.3f}s")
            print(f"      Post Follow-through Duration: {landing_timing2.get('post_follow_through_duration', 'N/A'):.3f}s")
            print(f"      Total Analysis Duration: {landing_timing2.get('total_analysis_duration', 'N/A'):.3f}s")

    def _display_interpretation_results(self, interpretation: Dict):
        """Display interpreted analysis results."""
        print("\n🧠 INTERPRETED ANALYSIS RESULTS:")
        
        # Display phase transition analysis first
        if 'phase_transition_analysis' in interpretation:
            self._display_phase_transition_analysis(interpretation['phase_transition_analysis'])
        
        text_analysis = interpretation.get('text_analysis', {})
        key_insights = interpretation.get('key_insights', [])
        recommendations = interpretation.get('recommendations', [])
        
        # Display phase-by-phase interpretation
        for phase, analysis in text_analysis.items():
            print(f"\n📋 {phase.upper()} PHASE:")
            
            differences = analysis.get('differences', [])
            if differences:
                print("   Key Differences:")
                for diff in differences:
                    print(f"   • {diff}")
            
            insights = analysis.get('insights', [])
            if insights:
                print("   Insights:")
                for insight in insights:
                    print(f"   • {insight}")
        
        # Display overall insights
        if key_insights:
            print(f"\n🔍 OVERALL INSIGHTS:")
            for insight in key_insights:
                print(f"   • {insight}")
        
        # Display recommendations
        if recommendations:
            print(f"\n💡 COACHING RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        # Generate LLM prompt
        llm_prompt = self.analysis_interpreter.generate_llm_prompt(interpretation)
        print(f"\n🤖 LLM PROMPT GENERATED:")
        print("   (Ready for LLM analysis)")
        print(f"   Prompt length: {len(llm_prompt)} characters")
        
        # Save LLM prompt to file
        prompt_filename = f"llm_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        prompt_path = os.path.join(self.comparison_results_dir, prompt_filename)
        self.prompt_file_name= prompt_filename
        try:
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(llm_prompt)
            print(f"   📄 LLM prompt saved: {prompt_filename}")
        except Exception as e:
            print(f"   ❌ Error saving LLM prompt: {e}")

    def run_comparison(self) -> bool:
        """
        Run the complete comparison pipeline
        
        Returns:
            True if successful, False otherwise
        """
        print("🏀 Basketball Shooting Form Comparison Pipeline")
        print("=" * 60)
        
        # Step 1: Select videos
        video1_path, video2_path = self.select_videos()
        if not video1_path or not video2_path:
            return False
        
        # Set video paths for metadata tracking
        self.video1_path = video1_path
        self.video2_path = video2_path
        
        # Step 2: Process videos
        print("\n🔄 STEP 2: Processing Videos")
        print("=" * 50)
        
        self.video1_data = self.process_video_data(video1_path)
        if not self.video1_data:
            print("❌ Failed to process first video")
            return False
            
        self.video2_data = self.process_video_data(video2_path)
        if not self.video2_data:
            print("❌ Failed to process second video")
            return False
        
        # Ensure metadata is set
        if not self.video1_metadata:
            self.video1_metadata = self.video1_data.get('metadata', {})
        if not self.video2_metadata:
            self.video2_metadata = self.video2_data.get('metadata', {})
        
        # Step 3: Select shots for comparison
        selected_shot1, selected_shot2 = self.select_shots(self.video1_data, self.video2_data)
        
        # Check if we have valid shot selections
        shots1 = self.video1_data.get('metadata', {}).get('shots', [])
        shots2 = self.video2_data.get('metadata', {}).get('shots', [])
        
        if len(shots1) == 0 and len(shots2) == 0:
            print("❌ No shots detected in either video. Cannot perform comparison.")
            return False
        elif selected_shot1 is None and selected_shot2 is None:
            print("✅ All shots selected for comparison.")
        elif selected_shot1 is None or selected_shot2 is None:
            print("❌ Please select specific shots for comparison or 'all' for all shots.")
            return False
        else:
            print(f"✅ Selected specific shots: Video 1 - {selected_shot1}, Video 2 - {selected_shot2}")
        
        # Step 4: Perform comparison
        comparison_results = self.perform_comparison(selected_shot1, selected_shot2)
        if not comparison_results:
            return False
        
        # Step 5: Save results
        if not self.save_comparison_results():
            return False
        
        # Step 6: Print summary
        self.print_comparison_summary()
        
        # Step 7: Create visualization (optional)
        print("\n🎬 STEP 7: Visualization (Optional)")
        print("=" * 50)
        
        # Ask user if they want to create visualizations
        while True:
            create_viz = input("Do you want to create comparison visualizations? (y/n): ").lower().strip()
            if create_viz in ['y', 'yes', 'n', 'no']:
                break
            print("Please enter 'y' for yes or 'n' for no.")
        
        if create_viz in ['y', 'yes']:
            try:
                visualizer = ShootingComparisonVisualizer()
                video_info = {
                    'video1_path': self.video1_path,
                    'video2_path': self.video2_path,
                    'video1_fps': self.video1_metadata.get('fps', 30.0) if self.video1_metadata else 30.0,
                    'video2_fps': self.video2_metadata.get('fps', 30.0) if self.video2_metadata else 30.0,
                    'video1_frames': len(self.video1_data.get('frames', [])),
                    'video2_frames': len(self.video2_data.get('frames', []))
                }
                
                # Create comprehensive visualizations
                output_videos = visualizer.create_comprehensive_visualizations(
                    self.video1_path, self.video2_path,
                    self.video1_data, self.video2_data,
                    self.comparison_results, video_info
                )
                
                if output_videos:
                    print(f"\n✅ Created {len(output_videos)} visualization videos:")
                    for video_type, video_path in output_videos.items():
                        print(f"   📹 {video_type}: {os.path.basename(video_path)}")
                else:
                    print("❌ No visualization videos were created")
                    
            except Exception as e:
                print(f"❌ Error creating visualizations: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⏭️  Skipping visualization creation.")
        
        print("\n🎉 Comparison pipeline completed successfully!")
        return True
    
    def select_shots(self, video1_data: Dict, video2_data: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        Select specific shots for comparison
        
        Args:
            video1_data: First video data
            video2_data: Second video data
            
        Returns:
            Tuple of (selected_shot1, selected_shot2) or (None, None) for all shots
        """
        print("\n🎯 STEP 2: Select Shots for Comparison")
        print("=" * 50)
        
        # Get shots from both videos
        shots1 = video1_data.get('metadata', {}).get('shots', {})
        shots2 = video2_data.get('metadata', {}).get('shots', {})
        
        shot_count1 = len(shots1)
        shot_count2 = len(shots2)
        
        print(f"📹 Video 1 ({os.path.basename(self.video1_path)}): {shot_count1} shots")
        print(f"📹 Video 2 ({os.path.basename(self.video2_path)}): {shot_count2} shots")
        
        # Ask user for comparison type
        print("\n🔍 Comparison Options:")
        print("1. Compare all shots (overall comparison)")
        print("2. Compare specific shots (individual shot comparison)")
        
        while True:
            try:
                choice = input("\nSelect option (1 or 2): ").strip()
                if choice == "1":
                    print("✅ Selected: Compare all shots")
                    return None, None  # None means all shots
                elif choice == "2":
                    print("✅ Selected: Compare specific shots")
                    return self._select_specific_shots(shots1, shots2)
                else:
                    print("❌ Invalid choice. Please enter 1 or 2.")
            except KeyboardInterrupt:
                print("\n❌ Selection cancelled.")
                return None, None
    
    def _select_specific_shots(self, shots1: Dict, shots2: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        Select specific shots from each video
        
        Args:
            shots1: Shots from first video
            shots2: Shots from second video
            
        Returns:
            Tuple of (selected_shot1, selected_shot2)
        """
        print("\n📋 Available Shots:")
        
        # Display shots from video 1
        print(f"\n📹 Video 1 ({os.path.basename(self.video1_path)}):")
        if isinstance(shots1, list):
            for i, shot_info in enumerate(shots1):
                if isinstance(shot_info, dict):
                    start_frame = shot_info.get('start_frame', 'N/A')
                    end_frame = shot_info.get('end_frame', 'N/A')
                    fixed_torso = shot_info.get('fixed_torso', 'N/A')
                    print(f"   shot{i+1}: Frames {start_frame}-{end_frame}, Torso: {fixed_torso}")
                else:
                    print(f"   shot{i+1}: {shot_info}")
        else:
            for shot_id, shot_info in shots1.items():
                start_frame = shot_info.get('start_frame', 'N/A')
                end_frame = shot_info.get('end_frame', 'N/A')
                fixed_torso = shot_info.get('fixed_torso', 'N/A')
                print(f"   {shot_id}: Frames {start_frame}-{end_frame}, Torso: {fixed_torso}")
        
        # Display shots from video 2
        print(f"\n📹 Video 2 ({os.path.basename(self.video2_path)}):")
        if isinstance(shots2, list):
            for i, shot_info in enumerate(shots2):
                if isinstance(shot_info, dict):
                    start_frame = shot_info.get('start_frame', 'N/A')
                    end_frame = shot_info.get('end_frame', 'N/A')
                    fixed_torso = shot_info.get('fixed_torso', 'N/A')
                    print(f"   shot{i+1}: Frames {start_frame}-{end_frame}, Torso: {fixed_torso}")
                else:
                    print(f"   shot{i+1}: {shot_info}")
        else:
            for shot_id, shot_info in shots2.items():
                start_frame = shot_info.get('start_frame', 'N/A')
                end_frame = shot_info.get('end_frame', 'N/A')
                fixed_torso = shot_info.get('fixed_torso', 'N/A')
                print(f"   {shot_id}: Frames {start_frame}-{end_frame}, Torso: {fixed_torso}")
        
        # Select shot from video 1
        while True:
            try:
                if len(shots1) == 0:
                    print("⚠️  No shots detected in Video 1")
                    selected_shot1 = None
                    break
                elif len(shots1) == 1:
                    # Only one shot available, auto-select
                    selected_shot1 = "shot1"
                    print(f"✅ Auto-selected: shot1 (only shot available)")
                    break
                else:
                    shot1_choice = input(f"\nSelect shot from Video 1 (1-{len(shots1)} or 'all'): ").strip()
                    if shot1_choice.lower() == 'all':
                        selected_shot1 = None
                        break
                    elif shot1_choice.isdigit() and 1 <= int(shot1_choice) <= len(shots1):
                        selected_shot1 = f"shot{int(shot1_choice)}"
                        break
                    else:
                        print(f"❌ Invalid choice. Please enter 1-{len(shots1)} or 'all'.")
            except KeyboardInterrupt:
                print("\n❌ Selection cancelled.")
                return None, None
        
        # Select shot from video 2
        while True:
            try:
                if len(shots2) == 0:
                    print("⚠️  No shots detected in Video 2")
                    selected_shot2 = None
                    break
                elif len(shots2) == 1:
                    # Only one shot available, auto-select
                    selected_shot2 = "shot1"
                    print(f"✅ Auto-selected: shot1 (only shot available)")
                    break
                else:
                    shot2_choice = input(f"Select shot from Video 2 (1-{len(shots2)} or 'all'): ").strip()
                    if shot2_choice.lower() == 'all':
                        selected_shot2 = None
                        break
                    elif shot2_choice.isdigit() and 1 <= int(shot2_choice) <= len(shots2):
                        selected_shot2 = f"shot{int(shot2_choice)}"
                        break
                    else:
                        print(f"❌ Invalid choice. Please enter 1-{len(shots2)} or 'all'.")
            except KeyboardInterrupt:
                print("\n❌ Selection cancelled.")
                return None, None
        
        print(f"\n✅ Selected:")
        print(f"   Video 1: {selected_shot1 if selected_shot1 else 'All shots'}")
        print(f"   Video 2: {selected_shot2 if selected_shot2 else 'All shots'}")
        
        return selected_shot1, selected_shot2
    
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


def main():
    """Main function to run the shooting comparison pipeline"""
    try:
        pipeline = ShootingComparisonPipeline()
        success = pipeline.run_comparison()
        
        if success:
            print("\n✅ Comparison pipeline completed successfully!")
        else:
            print("\n❌ Comparison pipeline failed!")
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()