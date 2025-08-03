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


class ShootingComparisonPipeline:
    """Pipeline for comparing basketball shooting forms between two videos"""
    
    def __init__(self):
        """Initialize the comparison pipeline"""
        self.video_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "video")
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "results")
        self.comparison_results_dir = os.path.join(os.path.dirname(__file__), "results")
        
        # Create comparison results directory if it doesn't exist
        os.makedirs(self.comparison_results_dir, exist_ok=True)
        
        # Initialize components
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
        Select two videos for comparison using GUI
        
        Returns:
            Tuple of (video1_path, video2_path)
        """
        print("🎬 STEP 1: Select Videos for Comparison")
        print("=" * 50)
        
        # Hide main tkinter window
        root = tk.Tk()
        root.withdraw()
        
        try:
            # Select first video
            print("📹 Select the first video (Reference):")
            video1_path = filedialog.askopenfilename(
                title="Select First Video (Reference)",
                initialdir=self.video_dir,
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ]
            )
            
            if not video1_path:
                print("❌ No first video selected.")
                return None, None
                
            print(f"✅ First video selected: {os.path.basename(video1_path)}")
            
            # Select second video
            print("\n📹 Select the second video (Comparison):")
            video2_path = filedialog.askopenfilename(
                title="Select Second Video (Comparison)",
                initialdir=self.video_dir,
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ]
            )
            
            if not video2_path:
                print("❌ No second video selected.")
                return None, None
                
            print(f"✅ Second video selected: {os.path.basename(video2_path)}")
            
            # Verify videos are different
            if video1_path == video2_path:
                print("⚠️ Same video selected for both. Please select different videos.")
                messagebox.showwarning("Warning", "Please select different videos for comparison.")
                return None, None
                
            self.video1_path = video1_path
            self.video2_path = video2_path
            
            return video1_path, video2_path
            
        except Exception as e:
            print(f"❌ Error selecting videos: {e}")
            return None, None
        finally:
            root.destroy()
    
    def process_video_data(self, video_path: str) -> Optional[Dict]:
        """
        Load existing processed video data from JSON file
        
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
    
    def perform_comparison(self) -> Optional[Dict]:
        """
        Perform DTW-based comparison between the two videos
        
        Returns:
            Comparison results dictionary or None if failed
        """
        if not self.video1_data or not self.video2_data:
            print("❌ Video data not available for comparison")
            return None
            
        print("\n🔄 STEP 3: Performing DTW Comparison")
        print("=" * 50)
        
        # Save video data to temporary files for DTW processor
        temp_file1 = os.path.join(self.comparison_results_dir, "temp_video1.json")
        temp_file2 = os.path.join(self.comparison_results_dir, "temp_video2.json")
        
        try:
            with open(temp_file1, 'w', encoding='utf-8') as f:
                json.dump(self.video1_data, f, indent=2)
            with open(temp_file2, 'w', encoding='utf-8') as f:
                json.dump(self.video2_data, f, indent=2)
            
            # Get selected hand from metadata
            selected_hand = self.video1_data.get('metadata', {}).get('hand', 'right')
            
            # Perform various DTW comparisons
            comparison_results = {}
            
            print("📊 Performing coordinate-based overall comparison...")
            coord_overall = self.dtw_processor.analyze_overall_phases_coordinate(
                temp_file1, temp_file2, selected_hand
            )
            comparison_results['coordinate_overall'] = coord_overall
            
            print("📊 Performing feature-based overall comparison...")
            feature_overall = self.dtw_processor.analyze_overall_phases_feature(
                temp_file1, temp_file2, selected_hand
            )
            comparison_results['feature_overall'] = feature_overall
            
            print("📊 Performing loading phases comparison...")
            loading = self.dtw_processor.analyze_loading_phases(
                temp_file1, temp_file2, selected_hand
            )
            comparison_results['loading'] = loading
            
            print("📊 Performing rising phases comparison...")
            rising = self.dtw_processor.analyze_rising_phases(
                temp_file1, temp_file2, selected_hand
            )
            comparison_results['rising'] = rising
            
            print("📊 Performing release phase comparison...")
            release = self.dtw_processor.analyze_release_phases(
                temp_file1, temp_file2, selected_hand
            )
            comparison_results['release'] = release
            
            print("📊 Performing follow-through phase comparison...")
            follow_through = self.dtw_processor.analyze_follow_through_phases(
                temp_file1, temp_file2, selected_hand
            )
            comparison_results['follow_through'] = follow_through
            
            # Add metadata
            comparison_results['metadata'] = {
                'video1_path': self.video1_path,
                'video2_path': self.video2_path,
                'video1_name': os.path.basename(self.video1_path),
                'video2_name': os.path.basename(self.video2_path),
                'selected_hand': selected_hand,
                'comparison_date': datetime.now().isoformat(),
                'video1_frames': len(self.video1_data.get('frames', [])),
                'video2_frames': len(self.video2_data.get('frames', []))
            }
            
            # Extract phase statistics
            video1_phases = self.extract_phase_data(self.video1_data)
            video2_phases = self.extract_phase_data(self.video2_data)
            
            comparison_results['phase_statistics'] = {
                'video1_phases': {phase: len(frames) for phase, frames in video1_phases.items()},
                'video2_phases': {phase: len(frames) for phase, frames in video2_phases.items()}
            }
            
            self.comparison_results = comparison_results
            
            print("✅ DTW comparison completed successfully!")
            return comparison_results
            
        except Exception as e:
            print(f"❌ Error during comparison: {e}")
            return None
        finally:
            # Clean up temporary files
            for temp_file in [temp_file1, temp_file2]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
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
        """Print a comprehensive comparison summary with time-based analysis"""
        if not self.comparison_results:
            print("❌ No comparison results available")
            return
            
        print("\n📋 COMPREHENSIVE COMPARISON ANALYSIS")
        print("=" * 60)
        
        # Generate comprehensive report
        if self.video1_metadata and self.video2_metadata:
            report = self.analysis_utils.create_comprehensive_report(
                self.comparison_results, self.video1_metadata, self.video2_metadata
            )
        else:
            # Fallback if metadata is not available
            report = self.analysis_utils.create_comprehensive_report(
                self.comparison_results, {}, {}
            )
        
        metadata = self.comparison_results.get('metadata', {})
        video_info = report['video_info']
        
        # Basic video information
        print(f"📹 Video 1 (Reference): {metadata.get('video1_name', 'Unknown')}")
        print(f"📹 Video 2 (Comparison): {metadata.get('video2_name', 'Unknown')}")
        print(f"🖐 Selected Hand: {metadata.get('selected_hand', 'Unknown')}")
        print(f"📊 Video 1: {metadata.get('video1_frames', 0)} frames @ {video_info['video1_fps']:.1f}fps")
        print(f"📊 Video 2: {metadata.get('video2_frames', 0)} frames @ {video_info['video2_fps']:.1f}fps")
        
        # Enhanced DTW results with time analysis
        print("\n🔍 DTW SIMILARITY ANALYSIS:")
        for phase in ['coordinate_overall', 'feature_overall', 'loading', 'rising', 'release', 'follow_through']:
            if phase in report['dtw_analysis']:
                analysis = report['dtw_analysis'][phase]
                print(analysis['formatted_display'])
        
        # Timing Analysis
        print("\n⏱️  TIMING ANALYSIS (Duration Comparison):")
        timing_analysis = report['timing_analysis']
        for phase, timing in timing_analysis.items():
            if timing['speed_ratio'] > 0:
                v1_time = timing['video1_duration']
                v2_time = timing['video2_duration']
                ratio = timing['speed_ratio']
                
                if timing['interpretation'] == 'similar_timing':
                    status = "✅ Perfect"
                elif 'slower' in timing['interpretation']:
                    status = f"🟡 V2 slower ({ratio:.1f}x)"
                elif 'faster' in timing['interpretation']:
                    status = f"🟠 V2 faster ({1/ratio:.1f}x)"
                else:
                    status = "🔴 Different"
                
                print(f"  • {phase.title()}: {v1_time:.2f}s vs {v2_time:.2f}s {status}")
            else:
                print(f"  • {phase.title()}: Missing in one video")
        
        # Coaching Insights
        insights = report['coaching_insights']
        if insights['strengths']:
            print("\n💪 STRENGTHS:")
            for strength in insights['strengths']:
                print(f"  ✅ {strength}")
        
        if insights['improvement_areas']:
            print("\n🎯 IMPROVEMENT AREAS:")
            for area in insights['improvement_areas']:
                print(f"  🔧 {area}")
        
        if insights['timing_insights']:
            print("\n⏰ TIMING INSIGHTS:")
            for insight in insights['timing_insights']:
                print(f"  📊 {insight}")
        
        # Summary verdict
        summary = report['summary']
        print(f"\n🏆 OVERALL ASSESSMENT:")
        
        if summary['best_phases']:
            best = ', '.join([p.title() for p in summary['best_phases']])
            print(f"  🥇 Best matching phases: {best}")
        
        if summary['worst_phases']:
            worst = ', '.join([p.title() for p in summary['worst_phases']])
            print(f"  🎯 Phases needing work: {worst}")
        
        timing_verdict = summary.get('timing_verdict', '')
        if timing_verdict == 'excellent_timing_consistency':
            print(f"  ⏱️  Timing: Excellent consistency across phases")
        elif timing_verdict == 'good_timing_consistency':
            print(f"  ⏱️  Timing: Good consistency, minor adjustments needed")
        elif timing_verdict == 'needs_timing_work':
            print(f"  ⏱️  Timing: Significant timing differences detected")
        
        # Phase statistics (legacy format)
        phase_stats = self.comparison_results.get('phase_statistics', {})
        video1_phases = phase_stats.get('video1_phases', {})
        video2_phases = phase_stats.get('video2_phases', {})
        
        print("\n📈 PHASE FRAME DISTRIBUTION:")
        all_phases = set(video1_phases.keys()) | set(video2_phases.keys())
        for phase in sorted(all_phases):
            v1_count = video1_phases.get(phase, 0)
            v2_count = video2_phases.get(phase, 0)
            v1_time = self.analysis_utils.frames_to_seconds(v1_count, video_info['video1_fps'])
            v2_time = self.analysis_utils.frames_to_seconds(v2_count, video_info['video2_fps'])
            print(f"  • {phase}: V1={v1_count}f({v1_time:.2f}s), V2={v2_count}f({v2_time:.2f}s)")
    
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
        
        # Step 3: Perform comparison
        comparison_results = self.perform_comparison()
        if not comparison_results:
            return False
        
        # Step 4: Save results
        if not self.save_comparison_results():
            return False
        
        # Step 5: Print summary
        self.print_comparison_summary()
        
        # Step 6: Create visualization
        print("\n🎬 STEP 6: Creating Comparison Visualization")
        print("=" * 50)
        
        try:
            visualizer = ShootingComparisonVisualizer()
            
            # Get the most recent results file
            video1_name = os.path.splitext(os.path.basename(self.video1_path))[0]
            video2_name = os.path.splitext(os.path.basename(self.video2_path))[0]
            
            # Find the most recent comparison results file
            results_files = [f for f in os.listdir(self.comparison_results_dir) 
                           if f.startswith(f"comparison_{video1_name}_vs_{video2_name}") and f.endswith('.json')]
            
            if results_files:
                # Sort by modification time and get the most recent
                results_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.comparison_results_dir, x)), reverse=True)
                latest_results_file = os.path.join(self.comparison_results_dir, results_files[0])
                
                print(f"📊 Using results file: {results_files[0]}")
                
                # Load and create visualization
                if visualizer.load_comparison_results(latest_results_file):
                    output_video_path = visualizer.create_comparison_video()
                    print(f"✅ Comparison visualization created: {os.path.basename(output_video_path)}")
                else:
                    print("❌ Failed to load comparison results for visualization")
            else:
                print("❌ No comparison results file found for visualization")
                
        except Exception as e:
            print(f"⚠️  Visualization creation failed (continuing anyway): {e}")
        
        print("\n🎉 Shooting form comparison completed successfully!")
        return True


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