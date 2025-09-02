"""
Enhanced Shooting Comparison Pipeline

Extends existing pipeline with DTW analysis while maintaining all existing functionality.
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np

from .shooting_comparison_pipeline import ShootingComparisonPipeline
from .analysis_interpreter import AnalysisInterpreter
from .dtw_interpreter_extension import DTWInterpreterExtension

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class EnhancedShootingComparisonPipeline(ShootingComparisonPipeline):
    """
    Enhanced pipeline that adds DTW analysis to existing comparison workflow.
    
    Maintains full compatibility with existing pipeline while adding DTW capabilities.
    """
    
    def __init__(self):
        super().__init__()  # Initialize existing pipeline completely
        self.dtw_extension = DTWInterpreterExtension()
        
        # Keep existing interpreter - just extend it
        self.existing_interpreter = AnalysisInterpreter()
        
        print("🔄 Enhanced Shooting Comparison Pipeline initialized")
        print("   📊 Existing phase analysis: ✅")
        print("   🎯 DTW motion analysis: ✅")
    
    def run_comparison(self, video1_path: str, video2_path: str, 
                      save_results: bool = True, include_dtw: bool = True, 
                      create_visualizations: bool = True, enable_shot_selection: bool = True) -> Dict:
        """
        Run comparison with optional DTW analysis and visualizations.
        
        Args:
            video1_path: Path to first video
            video2_path: Path to second video  
            save_results: Whether to save results
            include_dtw: Whether to include DTW analysis (default: True)
            create_visualizations: Whether to create DTW visualizations (default: True)
            enable_shot_selection: Whether to enable shot selection (default: True)
            
        Returns:
            Comparison results with optional DTW enhancement
        """
        print(f"\n🏀 Starting Enhanced Shooting Comparison")
        print("=" * 60)
        print(f"📹 Video 1: {os.path.basename(video1_path)}")
        print(f"📹 Video 2: {os.path.basename(video2_path)}")
        print(f"🎯 DTW Analysis: {'Enabled' if include_dtw else 'Disabled'}")
        print(f"🎨 Visualizations: {'Enabled' if create_visualizations else 'Disabled'}")
        print(f"🎯 Shot Selection: {'Enabled' if enable_shot_selection else 'Disabled'}")
        print("=" * 60)
        
        # Run existing comparison pipeline exactly as before
        print("\n🔄 Phase 1: Running existing phase-based comparison analysis...")
        try:
            # Use existing pipeline to get phase analysis
            existing_pipeline = ShootingComparisonPipeline()
            
            # Set up the pipeline properly
            existing_pipeline.video1_path = video1_path
            existing_pipeline.video2_path = video2_path
            
            # Try to process video data first
            print("   📹 Processing video data...")
            existing_pipeline.video1_data = existing_pipeline.process_video_data(video1_path)
            existing_pipeline.video2_data = existing_pipeline.process_video_data(video2_path)
            
            # If video data processing failed, try to load from other sources
            if not existing_pipeline.video1_data or not existing_pipeline.video2_data:
                print("   ⚠️ Video data not found in standard location, trying alternative sources...")
                
                # Try to load from basketball_shooting_analyzer results
                base_name1 = self._get_base_name(video1_path)
                base_name2 = self._get_base_name(video2_path)
                
                # Check for analyzer results
                analyzer_result1 = f"data/results/{base_name1}_normalized_output.json"
                analyzer_result2 = f"data/results/{base_name2}_normalized_output.json"
                
                if os.path.exists(analyzer_result1) and os.path.exists(analyzer_result2):
                    print("   📄 Found analyzer results, loading...")
                    with open(analyzer_result1, 'r') as f:
                        existing_pipeline.video1_data = json.load(f)
                    with open(analyzer_result2, 'r') as f:
                        existing_pipeline.video2_data = json.load(f)
                    
                    # Set metadata
                    existing_pipeline.video1_metadata = existing_pipeline.video1_data.get('metadata', {})
                    existing_pipeline.video2_metadata = existing_pipeline.video2_data.get('metadata', {})
                    
                    print(f"   ✅ Loaded video data from analyzer results")
                else:
                    print("❌ Failed to find video data for analysis")
                    print(f"   🔍 Expected files:")
                    print(f"      - {analyzer_result1}")
                    print(f"      - {analyzer_result2}")
                    print(f"   💡 Please run the integrated pipeline first to process these videos.")
                    return {'error': 'Video data not found for analysis'}
            
            # Set metadata for analyzers
            existing_pipeline.video1_metadata = existing_pipeline.video1_data.get('metadata', {})
            existing_pipeline.video2_metadata = existing_pipeline.video2_data.get('metadata', {})
            
            print(f"   ✅ Video 1: {len(existing_pipeline.video1_data.get('frames', []))} frames")
            print(f"   ✅ Video 2: {len(existing_pipeline.video2_data.get('frames', []))} frames")
            
            # Handle shot selection if enabled
            selected_shot1 = None
            selected_shot2 = None
            
            if enable_shot_selection:
                print("\n🎯 Shot Selection for Enhanced Analysis")
                print("=" * 50)
                
                # Check if shots are available
                shots1 = existing_pipeline.video1_data.get('metadata', {}).get('shots', [])
                shots2 = existing_pipeline.video2_data.get('metadata', {}).get('shots', [])
                
                shot_count1 = len(shots1)
                shot_count2 = len(shots2)
                
                print(f"📹 Video 1: {shot_count1} shots detected")
                print(f"📹 Video 2: {shot_count2} shots detected")
                
                if shot_count1 > 0 or shot_count2 > 0:
                    # Ask user for shot selection
                    print("\n🔍 Shot Selection Options:")
                    print("1. Analyze all shots (integrated analysis)")
                    print("2. Select specific shots for comparison")
                    
                    while True:
                        try:
                            choice = input("\nSelect option (1 or 2): ").strip()
                            if choice == "1":
                                print("✅ Selected: Analyze all shots (integrated)")
                                selected_shot1 = None
                                selected_shot2 = None
                                break
                            elif choice == "2":
                                print("✅ Selected: Select specific shots")
                                selected_shot1, selected_shot2 = self._select_specific_shots_for_enhanced(
                                    existing_pipeline.video1_data, existing_pipeline.video2_data
                                )
                                break
                            else:
                                print("❌ Invalid choice. Please enter 1 or 2.")
                        except KeyboardInterrupt:
                            print("\n❌ Selection cancelled.")
                            return {'error': 'Shot selection cancelled'}
                else:
                    print("⚠️ No shots detected, proceeding with full video analysis")
                    selected_shot1 = None
                    selected_shot2 = None
            else:
                print("🎯 Shot selection disabled - using full video analysis")
                selected_shot1 = None
                selected_shot2 = None
            
            # Run phase analysis with selected shots
            existing_results = existing_pipeline.perform_comparison(selected_shot1, selected_shot2)
            
            if not existing_results:
                print("❌ Phase analysis failed")
                return {'error': 'Phase analysis failed'}
                
            print("✅ Phase-based analysis completed")
            
        except Exception as e:
            print(f"❌ Phase analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Phase analysis failed: {str(e)}'}
        
        # Run existing interpretation exactly as before
        print("\n🔄 Phase 2: Interpreting phase-based analysis...")
        try:
            # Debug: Check what's in existing_results
            print(f"   📊 Available keys in existing_results: {list(existing_results.keys())}")
            if 'setup_analysis' in existing_results:
                print("   ✅ setup_analysis found")
            if 'loading_analysis' in existing_results:
                print("   ✅ loading_analysis found")
            if 'rising_analysis' in existing_results:
                print("   ✅ rising_analysis found")
            if 'release_analysis' in existing_results:
                print("   ✅ release_analysis found")
            if 'follow_through_analysis' in existing_results:
                print("   ✅ follow_through_analysis found")
            if 'landing_analysis' in existing_results:
                print("   ✅ landing_analysis found")
            
            existing_interpretation = self.existing_interpreter.interpret_comparison_results(existing_results)
            
            # Debug: Check what's in interpretation
            print(f"   📊 Interpretation keys: {list(existing_interpretation.keys())}")
            if 'text_analysis' in existing_interpretation:
                text_analysis = existing_interpretation['text_analysis']
                print(f"   📝 Text analysis phases: {list(text_analysis.keys())}")
                for phase, analysis in text_analysis.items():
                    differences = analysis.get('differences', [])
                    print(f"      • {phase}: {len(differences)} differences")
                    if differences:
                        print(f"         - {differences[0]}")
            
            print("✅ Phase-based interpretation completed")
        except Exception as e:
            print(f"⚠️ Interpretation failed: {e}")
            existing_interpretation = {'error': f'Interpretation failed: {str(e)}'}
        
        # If DTW not requested, return existing results with interpretation
        if not include_dtw:
            print("\n📋 DTW analysis skipped - returning standard results")
            final_results = existing_results.copy()
            final_results['interpretation'] = existing_interpretation
            final_results['metadata']['analysis_type'] = 'standard'
            
            if save_results:
                self._save_results(final_results, video1_path, video2_path, suffix='_standard')
            
            return final_results
        
        # Add DTW analysis
        print("\n🔄 Phase 3: Adding DTW motion analysis...")
        try:
            # Use video data from existing pipeline
            video1_data = existing_pipeline.video1_data
            video2_data = existing_pipeline.video2_data
            
            if video1_data and video2_data:
                selected_hand = existing_results.get('metadata', {}).get('selected_hand', 'right')
                print(f"   🤚 Using {selected_hand} hand analysis")
                print(f"   📊 Video 1 frames: {len(video1_data.get('frames', []))}")
                print(f"   📊 Video 2 frames: {len(video2_data.get('frames', []))}")
                
                # Add video data to existing results for DTW analysis
                # Keep full data for DTW analysis
                existing_results['video1_data'] = video1_data
                existing_results['video2_data'] = video2_data
                
                # Extend existing interpretation with DTW
                enhanced_interpretation = self.dtw_extension.extend_existing_interpretation(
                    existing_interpretation, existing_results, video1_data, video2_data, selected_hand
                )
                
                # Get DTW analysis results from enhanced interpretation
                dtw_analysis = enhanced_interpretation.get('dtw_analysis', {})
                
                # Ensure existing text analysis is preserved
                if 'text_analysis' in existing_interpretation:
                    enhanced_interpretation['text_analysis'] = existing_interpretation['text_analysis']
                    print("✅ Preserved existing text analysis")
                
                # Ensure existing key insights are preserved
                if 'key_insights' in existing_interpretation:
                    if 'key_insights' not in enhanced_interpretation:
                        enhanced_interpretation['key_insights'] = []
                    enhanced_interpretation['key_insights'].extend(existing_interpretation['key_insights'])
                    print("✅ Preserved existing key insights")
                
                # Combine results
                final_results = existing_results.copy()
                final_results['interpretation'] = enhanced_interpretation
                final_results['metadata']['dtw_analysis_included'] = True
                final_results['metadata']['analysis_type'] = 'enhanced'
                
                # Add DTW analysis results to final_results
                if dtw_analysis:
                    final_results['dtw_analysis'] = dtw_analysis
                    # Update overall similarity from DTW analysis
                    if 'overall_similarity' in dtw_analysis:
                        final_results['overall_similarity'] = dtw_analysis['overall_similarity']
                        final_results['grade'] = dtw_analysis.get('grade', 'N/A')
                        final_results['confidence'] = dtw_analysis.get('metadata', {}).get('analysis_confidence', 'Unknown')
                        print(f"✅ Updated overall similarity to {dtw_analysis['overall_similarity']:.1f}% from DTW analysis")
                
                # Remove frame data from final results to reduce file size
                if 'video1_data' in final_results and 'frames' in final_results['video1_data']:
                    final_results['video1_data']['frames'] = []
                    print("✅ Removed video1 frame data from final results")
                if 'video2_data' in final_results and 'frames' in final_results['video2_data']:
                    final_results['video2_data']['frames'] = []
                    print("✅ Removed video2 frame data from final results")
                
                print("✅ DTW analysis completed and integrated successfully")
                
                # Create DTW visualizations if requested
                if create_visualizations:
                    print("\n🎨 Phase 4: Creating DTW visualizations...")
                    print(f"   🔍 Debug: create_visualizations = {create_visualizations}")
                    
                    try:
                        # Try to import visualizer
                        try:
                            from .dtw_analysis.dtw_visualizer import DTWVisualizer
                        except ImportError:
                            from dtw_analysis.dtw_visualizer import DTWVisualizer
                        
                        print("   ✅ DTWVisualizer imported successfully")
                        
                        # Check if we have the required DTW results
                        dtw_motion_analysis = enhanced_interpretation.get('dtw_motion_analysis', {})
                        print(f"   📊 DTW motion analysis keys: {list(dtw_motion_analysis.keys())}")
                        
                        visualizer = DTWVisualizer()
                        print("   ✅ DTWVisualizer instance created")
                        
                        # Use the complete DTW results instead of just motion analysis
                        dtw_results_for_viz = {
                            'dtw_analysis': dtw_motion_analysis  # Wrap in expected structure
                        }
                        
                        print("   🎨 Starting comprehensive DTW report creation...")
                        visualization_files = visualizer.create_comprehensive_dtw_report(
                            dtw_results_for_viz, 
                            video1_data, video2_data, video1_path, video2_path
                        )
                        
                        if visualization_files:
                            final_results['metadata']['visualizations'] = visualization_files
                            print(f"   ✅ Created {len(visualization_files)} DTW visualization files")
                            for viz_type, file_path in visualization_files.items():
                                if file_path:
                                    print(f"      📁 {viz_type}: {file_path}")
                        else:
                            print("   ⚠️ No visualization files were created")
                            
                    except ImportError as import_error:
                        print(f"   ❌ Import error: {import_error}")
                        final_results['metadata']['visualization_error'] = f'Import error: {str(import_error)}'
                    except Exception as viz_error:
                        print(f"   ❌ Visualization creation failed: {viz_error}")
                        print(f"   🔍 Error type: {type(viz_error).__name__}")
                        import traceback
                        print(f"   📜 Traceback: {traceback.format_exc()}")
                        final_results['metadata']['visualization_error'] = str(viz_error)
                else:
                    print("\n🎨 Phase 4: DTW visualizations skipped (disabled)")
            else:
                print("⚠️ Could not load normalized data for DTW analysis")
                print("   📋 Falling back to existing analysis only")
                
                final_results = existing_results.copy()
                final_results['interpretation'] = existing_interpretation
                final_results['metadata']['dtw_analysis_included'] = False
                final_results['metadata']['analysis_type'] = 'standard_fallback'
                final_results['metadata']['dtw_error'] = 'Normalized data not available'
                
        except Exception as e:
            print(f"❌ DTW analysis failed: {e}")
            print("   📋 Falling back to existing analysis")
            
            final_results = existing_results.copy()
            final_results['interpretation'] = existing_interpretation
            final_results['metadata']['dtw_analysis_included'] = False
            final_results['metadata']['analysis_type'] = 'standard_fallback'
            final_results['metadata']['dtw_error'] = str(e)
        
        # Add final metadata
        final_results['metadata']['pipeline_version'] = 'enhanced_v1.0'
        final_results['metadata']['analysis_timestamp'] = datetime.now().isoformat()
        
        # Save results
        if save_results:
            dtw_included = final_results['metadata'].get('dtw_analysis_included', False)
            suffix = '_enhanced' if dtw_included else '_standard'
            self._save_results(final_results, video1_path, video2_path, suffix)
        
        # Print summary
        self._print_analysis_summary(final_results)
        
        return final_results
    
    def _load_normalized_video_data(self, video_path: str) -> Optional[Dict]:
        """Load normalized video data for DTW analysis"""
        try:
            print(f"   📂 Loading normalized data for {os.path.basename(video_path)}...")
            
            # Try to get data from existing analyzer if available
            if hasattr(self, 'analyzer') and self.analyzer:
                # Check if analyzer has the data we need
                if hasattr(self.analyzer, 'normalized_data') and self.analyzer.normalized_data:
                    # Convert analyzer's normalized data to the format expected by DTW
                    normalized_frames = []
                    
                    for frame_data in self.analyzer.normalized_data:
                        # Convert to expected format
                        normalized_frame = {
                            'frame_index': frame_data.get('frame_index', 0),
                            'phase': frame_data.get('phase', 'General'),
                            'normalized_pose': self._convert_pose_format(frame_data),
                            'ball_info': frame_data.get('ball_info', {})
                        }
                        normalized_frames.append(normalized_frame)
                    
                    video_data = {
                        'frames': normalized_frames,
                        'metadata': {
                            'fps': getattr(self.analyzer, 'video_fps', 30.0),
                            'total_frames': len(normalized_frames)
                        }
                    }
                    
                    print(f"   ✅ Loaded {len(normalized_frames)} normalized frames")
                    return video_data
            
            # Fallback: try to load from file system
            base_name = self._get_base_name(video_path)
            potential_paths = [
                f"data/analyzed_data/{base_name}_normalized.json",
                f"data/extracted_data/{base_name}_pose_normalized.json",
                f"shooting_comparison/results/{base_name}_analysis.json"
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    print(f"   📄 Found data file: {path}")
                    with open(path, 'r') as f:
                        data = json.load(f)
                        if 'frames' in data or 'normalized_data' in data:
                            print(f"   ✅ Successfully loaded from {path}")
                            return data
            
            print(f"   ⚠️ No normalized data found for {base_name}")
            return None
            
        except Exception as e:
            print(f"   ❌ Error loading normalized data: {e}")
            return None
    
    def _convert_pose_format(self, frame_data: Dict) -> Dict:
        """Convert frame data to normalized pose format expected by DTW"""
        # This method converts the analyzer's frame format to DTW expected format
        normalized_pose = {}
        
        # Copy pose keypoints if available
        if 'pose' in frame_data:
            pose = frame_data['pose']
            for keypoint, data in pose.items():
                if isinstance(data, dict) and 'x' in data and 'y' in data:
                    normalized_pose[keypoint] = {
                        'x': float(data['x']),
                        'y': float(data['y']),
                        'confidence': data.get('confidence', 1.0)
                    }
        
        # Handle different pose data formats
        for key in ['normalized_pose', 'pose_data']:
            if key in frame_data:
                pose_data = frame_data[key]
                if isinstance(pose_data, dict):
                    for keypoint, data in pose_data.items():
                        if isinstance(data, dict) and 'x' in data and 'y' in data:
                            normalized_pose[keypoint] = {
                                'x': float(data['x']),
                                'y': float(data['y']),
                                'confidence': data.get('confidence', 1.0)
                            }
        
        return normalized_pose
    
    def _save_results(self, results: Dict, video1_path: str, video2_path: str, suffix: str = ''):
        """Save results with appropriate suffix"""
        try:
            # Create results directory if it doesn't exist
            results_dir = os.path.join("shooting_comparison", "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name1 = self._get_base_name(video1_path)
            base_name2 = self._get_base_name(video2_path)
            
            filename = f"comparison_{base_name1}_vs_{base_name2}{suffix}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            # Save results
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
            
            print(f"\n💾 Results saved: {filename}")
            
            # Also save a summary for quick reference
            summary_filename = f"summary_{base_name1}_vs_{base_name2}{suffix}_{timestamp}.json"
            summary_filepath = os.path.join(results_dir, summary_filename)
            
            summary = self._create_results_summary(results)
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
            
            print(f"📋 Summary saved: {summary_filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def _create_results_summary(self, results: Dict) -> Dict:
        """Create a concise summary of the analysis results"""
        summary = {
            'metadata': results.get('metadata', {}),
            'analysis_type': results.get('metadata', {}).get('analysis_type', 'unknown'),
            'dtw_analysis_included': results.get('metadata', {}).get('dtw_analysis_included', False)
        }
        
        # Add interpretation summary if available
        interpretation = results.get('interpretation', {})
        
        if 'dtw_motion_analysis' in interpretation:
            dtw_analysis = interpretation['dtw_motion_analysis']
            overall_motion = dtw_analysis.get('overall_motion_similarity', {})
            
            summary['dtw_summary'] = {
                'overall_similarity': overall_motion.get('score', 0),
                'grade': overall_motion.get('grade', 'N/A'),
                'confidence': overall_motion.get('confidence', 'Unknown')
            }
            
            # Feature breakdown summary
            feature_breakdown = dtw_analysis.get('feature_breakdown', {})
            summary['feature_similarities'] = {}
            
            for feature, data in feature_breakdown.items():
                if isinstance(data, dict) and 'similarity_score' in data:
                    summary['feature_similarities'][data.get('name', feature)] = data['similarity_score']
        
        # Add key insights
        if 'key_insights' in interpretation:
            summary['key_insights'] = interpretation['key_insights'][:5]  # Top 5 insights
        
        return summary
    
    def _print_analysis_summary(self, results: Dict):
        """Print a summary of the analysis results"""
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)
        
        metadata = results.get('metadata', {})
        interpretation = results.get('interpretation', {})
        
        print(f"📋 Analysis Type: {metadata.get('analysis_type', 'Unknown')}")
        print(f"🎯 DTW Analysis: {'✅ Included' if metadata.get('dtw_analysis_included') else '❌ Not included'}")
        
        # Display existing text analysis
        text_analysis = interpretation.get('text_analysis', {})
        if text_analysis:
            print("\n📝 Phase-based Analysis:")
            for phase, analysis in text_analysis.items():
                differences = analysis.get('differences', [])
                if differences:
                    print(f"   🔸 {phase.upper()}:")
                    for diff in differences[:2]:  # Show top 2 differences per phase
                        print(f"      • {diff}")
        
        # Check for DTW analysis results
        dtw_analysis = results.get('dtw_analysis', {})
        if dtw_analysis and 'overall_similarity' in dtw_analysis:
            overall_similarity = dtw_analysis['overall_similarity']
            grade = dtw_analysis.get('grade', 'N/A')
            confidence = dtw_analysis.get('metadata', {}).get('analysis_confidence', 'Unknown')
            
            print(f"\n📊 Overall Similarity: {overall_similarity:.1f}%")
            print(f"🎖️ Grade: {grade}")
            print(f"🎯 Confidence: {confidence}")
            
            # Feature similarities
            feature_similarities = dtw_analysis.get('feature_similarities', {})
            if feature_similarities:
                print("\n📈 Feature Similarities:")
                for feature, similarity in feature_similarities.items():
                    print(f"   🔸 {feature}: {similarity:.1f}%")
            
            # Phase similarities
            phase_similarities = dtw_analysis.get('phase_similarities', {})
            if phase_similarities:
                print("\n📊 Phase Similarities:")
                for phase, phase_data in phase_similarities.items():
                    if isinstance(phase_data, dict) and 'similarity' in phase_data:
                        similarity = phase_data['similarity']
                        print(f"   🔸 {phase}: {similarity:.1f}%")
        
        # Key insights
        key_insights = interpretation.get('key_insights', [])
        if key_insights:
            print("\n💡 Key Insights:")
            for i, insight in enumerate(key_insights[:3], 1):  # Show top 3
                print(f"   {i}. {insight}")
        
        print("=" * 60)
        print("✅ Analysis completed successfully!")
    
    def _process_video_data(self, video_path: str, video_label: str) -> Optional[Dict]:
        """Process video data by loading existing analysis results"""
        try:
            print(f"   📹 Loading {video_label}: {os.path.basename(video_path)}")
            
            # Try to find existing analysis results
            base_name = self._get_base_name(video_path)
            analysis_file = f"data/results/{base_name}_normalized_output.json"
            
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r') as f:
                    video_data = json.load(f)
                print(f"   ✅ Loaded existing analysis: {base_name}")
                return video_data
            else:
                print(f"   ❌ No analysis found: {analysis_file}")
                return None
                
        except Exception as e:
            print(f"   ❌ Error loading {video_label}: {e}")
            return None
    
    def _get_base_name(self, video_path: str) -> str:
        """Get base name from video path"""
        return os.path.splitext(os.path.basename(video_path))[0]
    
    def _resolve_video_path(self, video_input: str) -> str:
        """Resolve video input to full path"""
        # If it's already a full path and exists
        if os.path.exists(video_input):
            return video_input
        
        # If it's just a filename, look in video directories
        video_dirs = [
            "data/video/Standard",
            "data/video/EdgeCase", 
            "data/video/Bakke",
            "data/video/test/clips"
        ]
        
        for video_dir in video_dirs:
            full_path = os.path.join(video_dir, video_input)
            if os.path.exists(full_path):
                return full_path
        
        # If it's a number, try to match with available videos
        try:
            video_index = int(video_input) - 1
            available_videos = self.list_available_videos(show_output=False)
            if 0 <= video_index < len(available_videos):
                return available_videos[video_index]
        except ValueError:
            pass
        
        return None
    
    def list_videos_by_folder(self):
        """Organize videos by folders and return structured data"""
        video_dirs = {
            "Standard": "data/video/Standard",
            "EdgeCase": "data/video/EdgeCase", 
            "Bakke": "data/video/Bakke",
            "Test Clips": "data/video/test/clips"
        }
        
        videos_by_folder = {}
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        
        for folder_name, video_dir in video_dirs.items():
            if os.path.exists(video_dir):
                folder_videos = []
                for ext in video_extensions:
                    # Search for both lowercase and uppercase extensions
                    folder_videos.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
                    folder_videos.extend(glob.glob(os.path.join(video_dir, f"*{ext.upper()}")))
                
                if folder_videos:
                    # Remove duplicates and sort
                    folder_videos = list(set(folder_videos))
                    folder_videos.sort()
                    videos_by_folder[folder_name] = folder_videos
        
        return videos_by_folder
    
    def select_folder_and_videos(self):
        """Interactive folder and video selection"""
        videos_by_folder = self.list_videos_by_folder()
        
        if not videos_by_folder:
            print("❌ No videos found in any folder")
            return None, None
        
        # Display folder selection
        print("\n📁 Available video folders:")
        print("=" * 50)
        folder_names = list(videos_by_folder.keys())
        for i, folder in enumerate(folder_names, 1):
            video_count = len(videos_by_folder[folder])
            print(f"  [{i}] {folder} ({video_count} videos)")
        
        # Get folder selection
        while True:
            try:
                folder_choice = input(f"\n📂 Select folder (1-{len(folder_names)}): ").strip()
                folder_idx = int(folder_choice) - 1
                if 0 <= folder_idx < len(folder_names):
                    selected_folder = folder_names[folder_idx]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(folder_names)}")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Display videos in selected folder
        folder_videos = videos_by_folder[selected_folder]
        print(f"\n📹 Videos in {selected_folder} folder:")
        print("=" * 50)
        for i, video in enumerate(folder_videos, 1):
            print(f"  [{i}] {os.path.basename(video)}")
        
        # Get two video selections
        selected_videos = []
        for video_num in [1, 2]:
            while True:
                try:
                    video_choice = input(f"\n📹 Select video {video_num} (1-{len(folder_videos)}): ").strip()
                    video_idx = int(video_choice) - 1
                    if 0 <= video_idx < len(folder_videos):
                        selected_video = folder_videos[video_idx]
                        if selected_video not in selected_videos:
                            selected_videos.append(selected_video)
                            print(f"✅ Selected: {os.path.basename(selected_video)}")
                            break
                        else:
                            print("❌ This video is already selected. Please choose a different video.")
                    else:
                        print(f"❌ Please enter a number between 1 and {len(folder_videos)}")
                except ValueError:
                    print("❌ Please enter a valid number")
        
        return selected_videos[0], selected_videos[1]
    
    def list_available_videos(self, show_output=True):
        """List available videos with DTW capability info (legacy method)"""
        videos_by_folder = self.list_videos_by_folder()
        
        all_videos = []
        for folder_videos in videos_by_folder.values():
            all_videos.extend(folder_videos)
        
        if show_output:
            print(f"\n📹 Found {len(all_videos)} available videos across {len(videos_by_folder)} folders")
            
            print("\n🎯 Enhanced Pipeline Features:")
            print("   📊 Phase-based analysis: ✅")  
            print("   🎯 DTW motion analysis: ✅")
            print("   📋 Integrated interpretation: ✅")
            print("   💾 Enhanced result format: ✅")
        
        return all_videos

    def _select_specific_shots_for_enhanced(self, video1_data: Dict, video2_data: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        Select specific shots for enhanced analysis
        
        Args:
            video1_data: First video data
            video2_data: Second video data
            
        Returns:
            Tuple of (selected_shot1, selected_shot2)
        """
        print("\n📋 Available Shots:")
        
        # Get shots from both videos
        shots1 = video1_data.get('metadata', {}).get('shots', [])
        shots2 = video2_data.get('metadata', {}).get('shots', [])
        
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
        
        print(f"\n✅ Selected for Enhanced Analysis:")
        print(f"   Video 1: {selected_shot1 if selected_shot1 else 'All shots'}")
        print(f"   Video 2: {selected_shot2 if selected_shot2 else 'All shots'}")
        
        return selected_shot1, selected_shot2


# Utility function for easy pipeline usage
def run_enhanced_comparison(video1_path: str, video2_path: str, 
                          save_results: bool = True, include_dtw: bool = True) -> Dict:
    """
    Convenience function to run enhanced shooting comparison.
    
    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        save_results: Whether to save results to file
        include_dtw: Whether to include DTW analysis
        
    Returns:
        Enhanced comparison results
    """
    pipeline = EnhancedShootingComparisonPipeline()
    return pipeline.run_comparison(video1_path, video2_path, save_results, include_dtw)


if __name__ == "__main__":
    print("🏀 Enhanced Shooting Comparison Pipeline")
    print("="*60)
    
    # Create pipeline instance
    pipeline = EnhancedShootingComparisonPipeline()
    
    # Show pipeline features
    pipeline.list_available_videos()
    
    # Interactive folder and video selection
    print("\n🎯 Select videos for DTW analysis:")
    video1_path, video2_path = pipeline.select_folder_and_videos()
    
    if not video1_path or not video2_path:
        print("❌ Video selection cancelled or failed")
        exit(1)
    
    print(f"\n📋 Selected videos:")
    print(f"   📹 Video 1: {os.path.basename(video1_path)}")
    print(f"   📹 Video 2: {os.path.basename(video2_path)}")
    
    # Ask about DTW analysis
    dtw_choice = input("\n🎯 Include DTW analysis? (y/n, default: y): ").strip().lower()
    include_dtw = dtw_choice != 'n'
    
    # Ask about visualizations
    viz_choice = input("🎨 Create DTW visualizations? (y/n, default: y): ").strip().lower()
    create_visualizations = viz_choice != 'n'
    
    # Ask about shot selection
    shot_selection_choice = input("🎯 Enable shot selection for enhanced analysis? (y/n, default: y): ").strip().lower()
    enable_shot_selection = shot_selection_choice != 'n'
    
    # Run enhanced comparison
    print(f"\n🚀 Starting enhanced comparison...")
    results = pipeline.run_comparison(video1_path, video2_path, 
                                    include_dtw=include_dtw, 
                                    create_visualizations=create_visualizations,
                                    enable_shot_selection=enable_shot_selection)
    
    if 'error' not in results:
        print(f"\n🎉 Analysis completed successfully!")
        dtw_included = results.get('metadata', {}).get('dtw_analysis_included', False)
        analysis_type = results.get('metadata', {}).get('analysis_type', 'unknown')
        print(f"📊 Analysis type: {analysis_type}")
        print(f"🎯 DTW analysis: {'✅ Included' if dtw_included else '❌ Not included'}")
        print(f"🎯 Shot selection: {'✅ Enabled' if enable_shot_selection else '❌ Disabled'}")
    else:
        print(f"❌ Analysis failed: {results['error']}")
        
    print("\n🏀 Enhanced Pipeline execution completed!")