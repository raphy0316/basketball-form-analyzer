"""
Basketball Shooting Analysis Utilities

This module provides utility functions for analyzing and interpreting
basketball shooting comparison results, including DTW analysis,
time-based calculations, and result interpretation.
"""

import json
import math
from typing import Dict, List, Tuple, Any
import numpy as np


class ShootingAnalysisUtils:
    """Utility class for basketball shooting analysis calculations"""
    
    def __init__(self):
        # DTW distance interpretation thresholds
        self.dtw_thresholds = {
            'very_similar': 100,
            'slightly_different': 500, 
            'moderately_different': 1000,
            'very_different': float('inf')
        }
        
        # Standard basketball video FPS (can be overridden)
        self.default_fps = 30.0
    
    def load_video_metadata(self, json_path: str) -> Dict:
        """
        Load video metadata including FPS from JSON file
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            Dictionary containing metadata
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                return data.get('metadata', {})
        except Exception as e:
            print(f"⚠️  Error loading metadata from {json_path}: {e}")
            return {}
    
    def frames_to_seconds(self, frames: int, fps: float = None) -> float:
        """
        Convert frame count to seconds
        
        Args:
            frames: Number of frames
            fps: Frames per second (uses default if None)
            
        Returns:
            Duration in seconds
        """
        if fps is None:
            fps = self.default_fps
        return frames / fps if fps > 0 else 0.0
    
    def calculate_phase_durations(self, phase_frames: Dict, fps: float = None) -> Dict:
        """
        Calculate phase durations in seconds
        
        Args:
            phase_frames: Dictionary of {phase_name: frame_count}
            fps: Frames per second
            
        Returns:
            Dictionary of {phase_name: duration_in_seconds}
        """
        if fps is None:
            fps = self.default_fps
            
        durations = {}
        for phase, frame_count in phase_frames.items():
            durations[phase] = self.frames_to_seconds(frame_count, fps)
        
        return durations
    
    def interpret_dtw_distance(self, distance: float) -> str:
        """
        Interpret DTW distance value
        
        Args:
            distance: DTW distance value
            
        Returns:
            String interpretation of the distance
        """
        if distance == 0.0:
            return "identical"
        elif distance <= self.dtw_thresholds['very_similar']:
            return "very_similar"
        elif distance <= self.dtw_thresholds['slightly_different']:
            return "slightly_different"
        elif distance <= self.dtw_thresholds['moderately_different']:
            return "moderately_different"
        else:
            return "very_different"
    
    def get_interpretation_emoji(self, interpretation: str) -> str:
        """Get emoji for interpretation level"""
        emoji_map = {
            'identical': '🎯',
            'very_similar': '✅', 
            'slightly_different': '🟡',
            'moderately_different': '🟠',
            'very_different': '🔴'
        }
        return emoji_map.get(interpretation, '❓')
    
    def calculate_timing_comparison(self, video1_phases: Dict, video2_phases: Dict, 
                                  fps1: float = None, fps2: float = None) -> Dict:
        """
        Compare timing between two videos
        
        Args:
            video1_phases: Phase frame counts for video 1
            video2_phases: Phase frame counts for video 2  
            fps1: FPS for video 1
            fps2: FPS for video 2
            
        Returns:
            Dictionary with timing comparison results
        """
        if fps1 is None:
            fps1 = self.default_fps
        if fps2 is None:
            fps2 = self.default_fps
            
        durations1 = self.calculate_phase_durations(video1_phases, fps1)
        durations2 = self.calculate_phase_durations(video2_phases, fps2)
        
        comparison = {}
        all_phases = set(durations1.keys()) | set(durations2.keys())
        
        for phase in all_phases:
            dur1 = durations1.get(phase, 0.0)
            dur2 = durations2.get(phase, 0.0)
            
            if dur1 > 0 and dur2 > 0:
                ratio = dur2 / dur1
                time_diff = abs(dur2 - dur1)
                
                comparison[phase] = {
                    'video1_duration': dur1,
                    'video2_duration': dur2,
                    'time_difference': time_diff,
                    'speed_ratio': ratio,  # >1 means video2 is slower
                    'interpretation': self._interpret_timing_difference(ratio)
                }
            else:
                comparison[phase] = {
                    'video1_duration': dur1,
                    'video2_duration': dur2,
                    'time_difference': abs(dur2 - dur1),
                    'speed_ratio': 0.0,
                    'interpretation': 'missing_phase'
                }
        
        return comparison
    
    def _interpret_timing_difference(self, ratio: float) -> str:
        """Interpret timing ratio between two videos"""
        if 0.9 <= ratio <= 1.1:
            return "similar_timing"
        elif ratio > 1.2:
            return "much_slower"
        elif ratio > 1.1:
            return "slightly_slower"
        elif ratio < 0.8:
            return "much_faster"
        elif ratio < 0.9:
            return "slightly_faster"
        else:
            return "different_timing"
    
    def calculate_dtw_efficiency(self, dtw_distance: float, frame_count1: int, frame_count2: int) -> float:
        """
        Calculate DTW efficiency (normalized by frame count)
        
        Args:
            dtw_distance: DTW distance
            frame_count1: Frame count for video 1
            frame_count2: Frame count for video 2
            
        Returns:
            Normalized DTW distance per frame
        """
        avg_frames = (frame_count1 + frame_count2) / 2
        if avg_frames > 0:
            return dtw_distance / avg_frames
        return float('inf')
    
    def generate_coaching_insights(self, comparison_results: Dict, 
                                 timing_analysis: Dict) -> Dict:
        """
        Generate coaching insights based on comparison results
        
        Args:
            comparison_results: DTW comparison results
            timing_analysis: Timing comparison results
            
        Returns:
            Dictionary with coaching insights
        """
        insights = {
            'strengths': [],
            'improvement_areas': [],
            'timing_insights': [],
            'technical_recommendations': []
        }
        
        # Analyze DTW results for technical insights
        for phase, result in comparison_results.items():
            if phase in ['coordinate_overall', 'feature_overall']:
                continue
                
            if 'dtw_distance' in result:
                distance = result['dtw_distance']
                interpretation = self.interpret_dtw_distance(distance)
                
                if interpretation in ['identical', 'very_similar']:
                    insights['strengths'].append(f"{phase.title()} technique is excellent - maintain this form")
                elif interpretation in ['moderately_different', 'very_different']:
                    insights['improvement_areas'].append(f"{phase.title()} needs attention - work on consistency")
        
        # Analyze timing insights
        for phase, timing in timing_analysis.items():
            interp = timing['interpretation']
            ratio = timing['speed_ratio']
            
            if interp == 'much_slower':
                insights['timing_insights'].append(f"{phase.title()}: Video2 is {ratio:.1f}x slower - consider speeding up")
            elif interp == 'much_faster':
                insights['timing_insights'].append(f"{phase.title()}: Video2 is {1/ratio:.1f}x faster - may need more control")
            elif interp == 'similar_timing':
                insights['timing_insights'].append(f"{phase.title()}: Excellent timing consistency")
        
        return insights
    
    def format_dtw_result_enhanced(self, result: Dict, phase_name: str, 
                                 fps1: float = None, fps2: float = None) -> str:
        """
        Format DTW result with enhanced information including time analysis
        
        Args:
            result: DTW result dictionary
            phase_name: Name of the phase
            fps1: FPS for video 1
            fps2: FPS for video 2
            
        Returns:
            Formatted string with enhanced information
        """
        if fps1 is None:
            fps1 = self.default_fps
        if fps2 is None:
            fps2 = self.default_fps
            
        if 'error' in result:
            frames1 = result.get('frames1', 0)
            frames2 = result.get('frames2', 0)
            time1 = self.frames_to_seconds(frames1, fps1)
            time2 = self.frames_to_seconds(frames2, fps2)
            return f"  • {phase_name}: No frames (V1:{time1:.2f}s, V2:{time2:.2f}s)"
        
        distance = result.get('dtw_distance', result.get('distance', 'N/A'))
        if isinstance(distance, (int, float)) and distance != float('inf'):
            frames1 = result.get('frames1', result.get('sequence1_length', 0))
            frames2 = result.get('frames2', result.get('sequence2_length', 0))
            time1 = self.frames_to_seconds(frames1, fps1)
            time2 = self.frames_to_seconds(frames2, fps2)
            
            interpretation = self.interpret_dtw_distance(distance)
            emoji = self.get_interpretation_emoji(interpretation)
            efficiency = self.calculate_dtw_efficiency(distance, frames1, frames2)
            
            return f"  • {phase_name}: {distance:.2f} {emoji} (V1:{time1:.2f}s, V2:{time2:.2f}s, Eff:{efficiency:.2f})"
        else:
            return f"  • {phase_name}: N/A"
    
    def create_comprehensive_report(self, comparison_results: Dict, 
                                  video1_metadata: Dict, video2_metadata: Dict) -> Dict:
        """
        Create a comprehensive analysis report
        
        Args:
            comparison_results: DTW comparison results
            video1_metadata: Metadata for video 1
            video2_metadata: Metadata for video 2
            
        Returns:
            Comprehensive report dictionary
        """
        fps1 = video1_metadata.get('fps', self.default_fps)
        fps2 = video2_metadata.get('fps', self.default_fps)
        
        # Extract phase statistics
        phase_stats = comparison_results.get('phase_statistics', {})
        video1_phases = phase_stats.get('video1_phases', {})
        video2_phases = phase_stats.get('video2_phases', {})
        
        # Calculate timing analysis
        timing_analysis = self.calculate_timing_comparison(video1_phases, video2_phases, fps1, fps2)
        
        # Generate insights
        insights = self.generate_coaching_insights(comparison_results, timing_analysis)
        
        # Create comprehensive report
        report = {
            'video_info': {
                'video1_fps': fps1,
                'video2_fps': fps2,
                'video1_total_frames': video1_metadata.get('total_frames', 0),
                'video2_total_frames': video2_metadata.get('total_frames', 0)
            },
            'dtw_analysis': {},
            'timing_analysis': timing_analysis,
            'coaching_insights': insights,
            'summary': self._create_summary(comparison_results, timing_analysis)
        }
        
        # Process DTW results with enhanced formatting
        for phase in ['coordinate_overall', 'feature_overall', 'loading', 'rising', 'release', 'follow_through']:
            if phase in comparison_results:
                result = comparison_results[phase]
                report['dtw_analysis'][phase] = {
                    'raw_result': result,
                    'formatted_display': self.format_dtw_result_enhanced(result, phase.replace('_', ' ').title(), fps1, fps2),
                    'interpretation': self.interpret_dtw_distance(result.get('dtw_distance', float('inf')))
                }
        
        return report
    
    def _create_summary(self, comparison_results: Dict, timing_analysis: Dict) -> Dict:
        """Create a summary of the analysis"""
        summary = {
            'best_phases': [],
            'worst_phases': [],
            'timing_verdict': '',
            'overall_similarity': ''
        }
        
        # Find best and worst phases based on DTW distance
        phase_scores = []
        for phase in ['loading', 'rising', 'release', 'follow_through']:
            if phase in comparison_results:
                result = comparison_results[phase]
                distance = result.get('dtw_distance', float('inf'))
                if distance != float('inf'):
                    phase_scores.append((phase, distance))
        
        if phase_scores:
            phase_scores.sort(key=lambda x: x[1])
            summary['best_phases'] = [phase for phase, _ in phase_scores[:2]]
            summary['worst_phases'] = [phase for phase, _ in phase_scores[-2:]]
        
        # Overall timing verdict
        timing_similarities = sum(1 for timing in timing_analysis.values() 
                                if timing['interpretation'] == 'similar_timing')
        total_phases = len(timing_analysis)
        
        if total_phases > 0:
            if timing_similarities / total_phases >= 0.7:
                summary['timing_verdict'] = 'excellent_timing_consistency'
            elif timing_similarities / total_phases >= 0.5:
                summary['timing_verdict'] = 'good_timing_consistency'
            else:
                summary['timing_verdict'] = 'needs_timing_work'
        
        return summary

def get_analysis_utils() -> ShootingAnalysisUtils:
    """Get a singleton instance of ShootingAnalysisUtils"""
    return ShootingAnalysisUtils()