"""
Analysis Interpreter for Basketball Shooting Comparison

This module takes raw comparison results and generates meaningful text interpretations
that can be sent to LLM for further analysis and coaching recommendations.
"""

import json
from typing import Dict, List, Tuple, Optional
import numpy as np


class AnalysisInterpreter:
    """
    Interprets comparison results and generates meaningful text analysis.
    
    Converts raw numerical data into human-readable insights and comparisons
    that can be used by LLM for coaching recommendations.
    """
    
    def __init__(self):
        self.interpretation_results = {}
    
    def interpret_comparison_results(self, comparison_results: Dict) -> Dict:
        """
        Interpret comparison results and generate text analysis.
        
        Args:
            comparison_results: Raw comparison data from pipeline
            
        Returns:
            Dictionary containing both raw data and interpreted text
        """
        interpretation = {
            'text_analysis': {},
            'key_insights': [],
            'recommendations': []
        }
        
        # Interpret phase transitions first
        if 'metadata' in comparison_results:
            metadata = comparison_results['metadata']
            video1_transitions = metadata.get('video1_phase_transitions', [])
            video2_transitions = metadata.get('video2_phase_transitions', [])
            
            interpretation['phase_transition_analysis'] = self._interpret_phase_transitions(
                video1_transitions, video2_transitions
            )
        
        # Interpret each phase
        if 'setup_analysis' in comparison_results:
            interpretation['text_analysis']['setup'] = self._interpret_setup_analysis(
                comparison_results['setup_analysis']
            )
        
        if 'loading_analysis' in comparison_results:
            interpretation['text_analysis']['loading'] = self._interpret_loading_analysis(
                comparison_results['loading_analysis']
            )
        
        if 'rising_analysis' in comparison_results:
            interpretation['text_analysis']['rising'] = self._interpret_rising_analysis(
                comparison_results['rising_analysis']
            )
        
        if 'release_analysis' in comparison_results:
            interpretation['text_analysis']['release'] = self._interpret_release_analysis(
                comparison_results['release_analysis']
            )
        
        if 'follow_through_analysis' in comparison_results:
            interpretation['text_analysis']['follow_through'] = self._interpret_follow_through_analysis(
                comparison_results['follow_through_analysis']
            )
        
        if 'landing_analysis' in comparison_results:
            interpretation['text_analysis']['landing'] = self._interpret_landing_analysis(
                comparison_results['landing_analysis']
            )
        
        # Generate overall insights and recommendations (DISABLED)
        # interpretation['key_insights'] = self._generate_key_insights(interpretation['text_analysis'])
        # interpretation['recommendations'] = self._generate_recommendations(interpretation['text_analysis'])
        
        return interpretation
    
    def _interpret_phase_transitions(self, video1_transitions: List[str], video2_transitions: List[str]) -> Dict:
        """
        Interpret phase transition patterns.
        
        Args:
            video1_transitions: Phase transition sequence for video 1
            video2_transitions: Phase transition sequence for video 2
            
        Returns:
            Dictionary containing phase transition analysis
        """
        interpretation = {
            'video1_pattern': self._analyze_phase_pattern(video1_transitions),
            'video2_pattern': self._analyze_phase_pattern(video2_transitions),
            'comparison': self._compare_phase_patterns(video1_transitions, video2_transitions)
        }
        
        return interpretation
    
    def _analyze_phase_pattern(self, transitions: List[str]) -> Dict:
        """
        Analyze a single video's phase transition pattern.
        
        Args:
            transitions: List of phase transitions
            
        Returns:
            Dictionary containing pattern analysis
        """
        analysis = {
            'pattern': 'continuous_loading_into_rising',
            'description': '',
            'issues': []
        }
        
        # Check for Loading-Rising pattern
        loading_rising_count = transitions.count('Loading-Rising')
        loading_count = transitions.count('Loading')
        rising_count = transitions.count('Rising')
        
        # Pattern 1: Loading-Rising appears multiple times
        if loading_rising_count > 1:
            analysis['pattern'] = 'repeated_loading_rising'
            analysis['issues'].append('Loading-Rising phase repeats multiple times')
        # Pattern 2: Rising appears without Loading-Rising
        elif rising_count > 0 and loading_rising_count == 0:
            analysis['pattern'] = 'rising_without_loading_rising'
            analysis['issues'].append('Rising phase occurs without Loading-Rising')
        # Pattern 3: Loading-Rising appears after Rising
        elif 'Rising' in transitions and 'Loading-Rising' in transitions and transitions.index('Loading-Rising') > transitions.index('Rising'):
            analysis['pattern'] = 'loading_rising_after_rising'
            analysis['issues'].append('Loading-Rising phase occurs after Rising')
        # Pattern 4: Continuous Loading into Rising pattern (Loading-Rising appears once, before Rising)
        elif loading_rising_count == 1 and rising_count > 0 and transitions.index('Loading-Rising') < transitions.index('Rising'):
            analysis['pattern'] = 'continuous_loading_into_rising'
        # Pattern 5: No Loading-Rising at all
        elif loading_rising_count == 0:
            analysis['pattern'] = 'no_loading_rising'
        
        # Assign description based on pattern
        pattern_descriptions = {
            'continuous_loading_into_rising': 'Loading-Rising occurs once before Rising, indicating smooth transition between loading and rising phases',
            'repeated_loading_rising': 'Loading-Rising phase repeats multiple times, indicating unstable phase transitions',
            'rising_without_loading_rising': 'Rising phase occurs without Loading-Rising, indicating minimal loading phase',
            'loading_rising_after_rising': 'Loading-Rising phase occurs after Rising, indicating improper loading during rising',
            'no_loading_rising': 'No Loading-Rising phase detected, indicating separate Loading and Rising phases',
        }
        analysis['description'] = pattern_descriptions.get(analysis['pattern'], '')
        return analysis
    
    def _compare_phase_patterns(self, video1_transitions: List[str], video2_transitions: List[str]) -> Dict:
        """
        Compare phase transition patterns between two videos.
        
        Args:
            video1_transitions: Phase transitions for video 1
            video2_transitions: Phase transitions for video 2
            
        Returns:
            Dictionary containing comparison analysis
        """
        comparison = {
            'similarity': 'similar',
            'differences': [],
            'recommendations': []
        }
        
        # Check if both videos have the same pattern
        video1_pattern = self._analyze_phase_pattern(video1_transitions)
        video2_pattern = self._analyze_phase_pattern(video2_transitions)
        
        if video1_pattern['pattern'] == video2_pattern['pattern']:
            comparison['similarity'] = 'similar'
            comparison['differences'].append('Both videos show similar phase transition patterns')
        else:
            comparison['similarity'] = 'different'
            comparison['differences'].append(f'Video 1 shows {video1_pattern["pattern"]} pattern')
            comparison['differences'].append(f'Video 2 shows {video2_pattern["pattern"]} pattern')
        
        # Add specific pattern descriptions
        comparison['video1_description'] = video1_pattern['description']
        comparison['video2_description'] = video2_pattern['description']
        
        # Add recommendations based on patterns
        if video1_pattern['pattern'] != 'continuous_loading_into_rising' or video2_pattern['pattern'] != 'continuous_loading_into_rising':
            comparison['recommendations'].append('Review phase detection settings for more stable transitions')
        
        return comparison
    
    def _interpret_setup_analysis(self, setup_analysis: Dict) -> Dict:
        """Interpret set-up phase analysis."""
        video1 = setup_analysis.get('video1', {})
        video2 = setup_analysis.get('video2', {})
        
        interpretation = {
            'comparisons': {},
            'differences': []
            # 'insights': [] # Removed
        }
        
        # Compare hip-knee-ankle angles
        angles1 = video1.get('hip_knee_ankle_angles', {})
        angles2 = video2.get('hip_knee_ankle_angles', {})
        
        if angles1 and angles2:
            left_avg1 = angles1.get('left', {}).get('average', 0)
            right_avg1 = angles1.get('right', {}).get('average', 0)
            left_avg2 = angles2.get('left', {}).get('average', 0)
            right_avg2 = angles2.get('right', {}).get('average', 0)
            
            # Check stability first (std > 5 degrees indicates instability)
            left_std1 = angles1.get('left', {}).get('std', 0)
            left_std2 = angles2.get('left', {}).get('std', 0)
            right_std1 = angles1.get('right', {}).get('std', 0)
            right_std2 = angles2.get('right', {}).get('std', 0)
            
            # Handle 'Undefined' values
            def safe_float(value):
                if value == 'Undefined' or value is None:
                    return 0.0
                return float(value)
            
            left_std1 = safe_float(left_std1)
            left_std2 = safe_float(left_std2)
            right_std1 = safe_float(right_std1)
            right_std2 = safe_float(right_std2)
            
            # Check for instability (DISABLED)
            # if left_std1 > 5:
            #     interpretation['differences'].append(
            #         f"Video 1 left leg shows high variability (std dev {left_std1:.1f}°)"
            #     )
            # if left_std2 > 5:
            #     interpretation['differences'].append(
            #         f"Video 2 left leg shows high variability (std dev {left_std2:.1f}°)"
            #     )
            # if right_std1 > 5:
            #     interpretation['differences'].append(
            #         f"Video 1 right leg shows high variability (std dev {right_std1:.1f}°)"
            #     )
            # if right_std2 > 5:
            #     interpretation['differences'].append(
            #         f"Video 2 right leg shows high variability (std dev {right_std2:.1f}°)"
            #     )
            
            # Only compare if both videos are stable (DISABLED - always compare)
            # if left_std1 <= 5 and left_std2 <= 5 and right_std1 <= 5 and right_std2 <= 5:
            
                        # Analyze differences (always compare now)
            left_diff = abs(left_avg1 - left_avg2)
            right_diff = abs(right_avg1 - right_avg2)
            
            if left_diff > 10:
                # Determine difference level
                if left_diff <= 15:
                    level = "low"
                elif left_diff <= 25:
                    level = "medium"
                elif left_diff <= 35:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has more bent leg (Video 1 is reference)
                if left_avg1 < left_avg2:
                    interpretation['differences'].append(
                        f"Left leg shows {level} difference in bending - Video 2 bends less ({left_diff:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Left leg shows {level} difference in bending - Video 2 bends more ({left_diff:.1f}°)"
                    )
            
            if right_diff > 10:
                # Determine difference level
                if right_diff <= 15:
                    level = "low"
                elif right_diff <= 25:
                    level = "medium"
                elif right_diff <= 35:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has more bent leg (Video 1 is reference)
                if right_avg1 < right_avg2:
                    interpretation['differences'].append(
                        f"Right leg shows {level} difference in bending - Video 2 bends less ({right_diff:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Right leg shows {level} difference in bending - Video 2 bends more ({right_diff:.1f}°)"
                    )
            
            # Determine which form is more stable (DISABLED)
            # if left_std1 < left_std2 and right_std1 < right_std2:
            #     interpretation['insights'].append("Video 1 shows more consistent leg positioning")
            # elif left_std2 < left_std1 and right_std2 < right_std1:
            #     interpretation['insights'].append("Video 2 shows more consistent leg positioning")
        
        # Compare ball-hip distances
        ball_dist1 = video1.get('ball_hip_distances', {})
        ball_dist2 = video2.get('ball_hip_distances', {})
        
        if ball_dist1 and ball_dist2:
            vert_avg1 = ball_dist1.get('average_vertical', 0)
            vert_avg2 = ball_dist2.get('average_vertical', 0)
            horiz_avg1 = ball_dist1.get('average_horizontal', 0)
            horiz_avg2 = ball_dist2.get('average_horizontal', 0)
            
            # Check stability (std > 0.05 for vertical, std > 0.05 * aspect_ratio for horizontal) (DISABLED)
            vert_std1 = ball_dist1.get('std_vertical', 0)
            vert_std2 = ball_dist2.get('std_vertical', 0)
            horiz_std1 = ball_dist1.get('std_horizontal', 0)
            horiz_std2 = ball_dist2.get('std_horizontal', 0)
            
            # Handle 'Undefined' values
            def safe_float(value):
                if value == 'Undefined' or value is None:
                    return 0.0
                return float(value)
            
            vert_std1 = safe_float(vert_std1)
            vert_std2 = safe_float(vert_std2)
            horiz_std1 = safe_float(horiz_std1)
            horiz_std2 = safe_float(horiz_std2)
            
            # Assume 16:9 aspect ratio for horizontal threshold (0.05 * 16/9 ≈ 0.089)
            horiz_threshold = 0.089
            
            # Check for instability (DISABLED)
            # if vert_std1 > 0.05:
            #     interpretation['differences'].append(
            #         f"Video 1 ball vertical position shows high variability (std dev {vert_std1:.3f})"
            #     )
            # if vert_std2 > 0.05:
            #     interpretation['differences'].append(
            #         f"Video 2 ball vertical position shows high variability (std dev {vert_std2:.3f})"
            #     )
            # if horiz_std1 > horiz_threshold:
            #     interpretation['differences'].append(
            #         f"Video 1 ball horizontal position shows high variability (std dev {horiz_std1:.3f})"
            #     )
            # if horiz_std2 > horiz_threshold:
            #     interpretation['differences'].append(
            #         f"Video 2 ball horizontal position shows high variability (std dev {horiz_std2:.3f})"
            #     )
            
            # Only compare if both videos are stable (DISABLED - always compare)
            # if vert_std1 <= 0.05 and vert_std2 <= 0.05 and horiz_std1 <= horiz_threshold and horiz_std2 <= horiz_threshold:
            
            # Always compare now
            vert_diff = abs(vert_avg1 - vert_avg2)
            horiz_diff = abs(horiz_avg1 - horiz_avg2)

            if vert_diff > 0.1:
                # Determine difference level
                if vert_diff <= 0.15:
                    level = "low"
                elif vert_diff <= 0.25:
                    level = "medium"
                elif vert_diff <= 0.35:
                    level = "high"
                else:
                    level = "very high"
                # Determine which video has ball higher/lower relative to hip (Video 1 is reference)
                if vert_avg1 > vert_avg2:
                    interpretation['differences'].append(
                        f"Ball vertical position shows {level} difference - Video 2 has ball lower ({vert_diff:.3f})"
                    )
                else:
                    interpretation['differences'].append(
                        f"Ball vertical position shows {level} difference - Video 2 has ball higher ({vert_diff:.3f})"
                    )

            if horiz_diff > 0.1:
                # Determine difference level
                if horiz_diff <= 0.15:
                    level = "low"
                elif horiz_diff <= 0.25:
                    level = "medium"
                elif horiz_diff <= 0.35:
                    level = "high"
                else:
                    level = "very high"
                # Determine which video has ball closer/further from body horizontally (Video 1 is reference)
                if horiz_avg1 > horiz_avg2:
                    interpretation['differences'].append(
                        f"Ball horizontal position shows {level} difference - Video 2 has ball closer to body ({horiz_diff:.3f})"
                    )
                else:
                    interpretation['differences'].append(
                        f"Ball horizontal position shows {level} difference - Video 2 has ball further from body ({horiz_diff:.3f})"
                    )
        
        # Compare stance width (distance between feet)
        foot_pos1 = video1.get('foot_positions', {})
        foot_pos2 = video2.get('foot_positions', {})
        
        if foot_pos1 and foot_pos2:
            left_foot1 = foot_pos1.get('left_foot', {})
            left_foot2 = foot_pos2.get('left_foot', {})
            right_foot1 = foot_pos1.get('right_foot', {})
            right_foot2 = foot_pos2.get('right_foot', {})
            
            # Calculate stance width for both videos
            if left_foot1 and right_foot1 and left_foot2 and right_foot2:
                # Get foot positions
                left_x1 = left_foot1.get('average_x', 0)
                right_x1 = right_foot1.get('average_x', 0)
                left_x2 = left_foot2.get('average_x', 0)
                right_x2 = right_foot2.get('average_x', 0)
                
                # Calculate stance width (distance between feet)
                stance_width1 = abs(left_x1 - right_x1)
                stance_width2 = abs(left_x2 - right_x2)
                
                # Check stability for both feet
                left_std_x1 = left_foot1.get('std_x', 0)
                left_std_x2 = left_foot2.get('std_x', 0)
                right_std_x1 = right_foot1.get('std_x', 0)
                right_std_x2 = right_foot2.get('std_x', 0)
                
                # Handle 'Undefined' values
                def safe_float(value):
                    if value == 'Undefined' or value is None:
                        return 0.0
                    return float(value)
                
                left_std_x1 = safe_float(left_std_x1)
                left_std_x2 = safe_float(left_std_x2)
                right_std_x1 = safe_float(right_std_x1)
                right_std_x2 = safe_float(right_std_x2)
                
                # Assume 16:9 aspect ratio for horizontal threshold
                horiz_threshold = 0.089
                
                # Check for instability in either foot (DISABLED)
                # if left_std_x1 > horiz_threshold or right_std_x1 > horiz_threshold:
                #     interpretation['differences'].append(
                #         f"Video 1 stance width shows high variability (std dev {max(left_std_x1, right_std_x1):.3f})"
                #     )
                # if left_std_x2 > horiz_threshold or right_std_x2 > horiz_threshold:
                #     interpretation['differences'].append(
                #         f"Video 2 stance width shows high variability (std dev {max(left_std_x2, right_std_x2):.3f})"
                #     )
                
                # Only compare if both videos are stable (DISABLED - always compare)
                # if (left_std_x1 <= horiz_threshold and right_std_x1 <= horiz_threshold and
                #     left_std_x2 <= horiz_threshold and right_std_x2 <= horiz_threshold):
                
                # Always compare now
                stance_diff = abs(stance_width1 - stance_width2)

                if stance_diff > 0.05:  # Lower threshold for stance width
                    # Determine difference level
                    if stance_diff <= 0.15:
                        level = "low"
                    elif stance_diff <= 0.25:
                        level = "medium"
                    elif stance_diff <= 0.35:
                        level = "high"
                    else:
                        level = "very high"
                    # Determine which video has wider stance (Video 1 is reference)
                    if stance_width1 > stance_width2:
                        interpretation['differences'].append(
                            f"Stance width shows {level} difference - Video 2 has narrower stance ({stance_diff:.3f})"
                        )
                    else:
                        interpretation['differences'].append(
                            f"Stance width shows {level} difference - Video 2 has wider stance ({stance_diff:.3f})"
                        )
                    
                    # Compare stance consistency (REMOVED)
                    # total_std1 = (left_std_x1 + right_std_x1) / 2
                    # total_std2 = (left_std_x2 + right_std_x2) / 2
                    # if total_std1 < total_std2:
                    #     interpretation['insights'].append("Video 1 shows more consistent stance positioning")
                    # elif total_std2 < total_std1:
                    #     interpretation['insights'].append("Video 2 shows more consistent stance positioning")
        
        # Compare shoulder tilt (upper body lean)
        shoulder_tilt1 = video1.get('shoulder_tilt', {})
        shoulder_tilt2 = video2.get('shoulder_tilt', {})
        
        if shoulder_tilt1 and shoulder_tilt2:
            tilt_avg1 = shoulder_tilt1.get('average', 0)
            tilt_avg2 = shoulder_tilt2.get('average', 0)
            
            # Check stability
            tilt_std1 = shoulder_tilt1.get('std', 0)
            tilt_std2 = shoulder_tilt2.get('std', 0)
            
            # Handle 'Undefined' values
            def safe_float(value):
                if value == 'Undefined' or value is None:
                    return 0.0
                return float(value)
            
            tilt_std1 = safe_float(tilt_std1)
            tilt_std2 = safe_float(tilt_std2)
            
            # Check for instability (std > 5 degrees) (DISABLED)
            # if tilt_std1 > 5:
            #     interpretation['differences'].append(
            #         f"Video 1 upper body lean shows high variability (std dev {tilt_std1:.1f}°)"
            #     )
            # if tilt_std2 > 5:
            #     interpretation['differences'].append(
            #         f"Video 2 upper body lean shows high variability (std dev {tilt_std2:.1f}°)"
            #     )
            
            # Only compare if both videos are stable (DISABLED - always compare)
            # if tilt_std1 <= 5 and tilt_std2 <= 5:
            
            # Always compare now
            tilt_diff = abs(tilt_avg1 - tilt_avg2)
                
            if tilt_diff > 5:  # Threshold for significant difference
                # Determine difference level
                if tilt_diff <= 10:
                    level = "low"
                elif tilt_diff <= 15:
                    level = "medium"
                elif tilt_diff <= 20:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video leans more (Video 1 is reference)
                if tilt_avg1 > tilt_avg2:
                    interpretation['differences'].append(
                        f"Upper body lean shows {level} difference - Video 2 leans less forward ({tilt_diff:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Upper body lean shows {level} difference - Video 2 leans more forward ({tilt_diff:.1f}°)"
                    )
        

        
        return interpretation
    
    def _interpret_loading_analysis(self, loading_analysis: Dict) -> Dict:
        """Interpret loading phase analysis."""
        video1 = loading_analysis.get('video1', {})
        video2 = loading_analysis.get('video2', {})
        
        interpretation = {
            'comparisons': {},
            'differences': []
            # 'insights': [] # Removed
        }
        

        
        # Compare max leg angles and asymmetric loading
        max_angles1 = video1.get('max_leg_angles', {})
        max_angles2 = video2.get('max_leg_angles', {})
        
        if max_angles1 and max_angles2:
            left_max1 = max_angles1.get('left', {}).get('max_angle', 0)
            right_max1 = max_angles1.get('right', {}).get('max_angle', 0)
            left_max2 = max_angles2.get('left', {}).get('max_angle', 0)
            right_max2 = max_angles2.get('right', {}).get('max_angle', 0)
            
            # Analyze asymmetric loading within each video
            asym1 = abs(left_max1 - right_max1)
            asym2 = abs(left_max2 - right_max2)
            
            # Determine asymmetric loading levels
            def get_asym_level(diff):
                if diff <= 10:
                    return "low"
                elif diff <= 20:
                    return "medium"
                elif diff <= 30:
                    return "high"
                else:
                    return "very high"
            
            asym1_level = get_asym_level(asym1)
            asym2_level = get_asym_level(asym2)
            
            # Compare asymmetric loading between videos
            asym_diff = abs(asym1 - asym2)
            if asym_diff > 5:
                if asym_diff <= 10:
                    asym_diff_level = "low"
                elif asym_diff <= 20:
                    asym_diff_level = "medium"
                elif asym_diff <= 30:
                    asym_diff_level = "high"
                else:
                    asym_diff_level = "very high"
                
                # Determine which leg each video favors
                def determine_leg_favor(left_angle, right_angle):
                    if left_angle < right_angle:  # Smaller angle = more bent = more weight
                        return "left"
                    else:
                        return "right"
                
                video1_favor = determine_leg_favor(left_max1, right_max1)
                video2_favor = determine_leg_favor(left_max2, right_max2)
                
                # Compare the favoring patterns
                if video1_favor == video2_favor:
                    # Both favor the same leg, but to different degrees
                    if asym1 > asym2:
                        interpretation['differences'].append(
                            f"Asymmetric loading shows {asym_diff_level} difference - Video 2 has more balanced loading on {video1_favor} side ({asym_diff:.1f}°)"
                        )
                    else:
                        interpretation['differences'].append(
                            f"Asymmetric loading shows {asym_diff_level} difference - Video 2 has more extreme loading on {video1_favor} side ({asym_diff:.1f}°)"
                        )
                else:
                    # Different legs are favored
                    interpretation['differences'].append(
                        f"Asymmetric loading shows {asym_diff_level} difference - Video 2 shifts weight to {video2_favor} side while Video 1 favors {video1_favor} side ({asym_diff:.1f}°)"
                    )
            
            # Analyze overall loading depth differences
            left_depth_diff = abs(left_max1 - left_max2)
            right_depth_diff = abs(right_max1 - right_max2)
            
            # Compare left leg depth
            if left_depth_diff > 15:
                # Determine difference level
                if left_depth_diff <= 20:
                    level = "low"
                elif left_depth_diff <= 30:
                    level = "medium"
                elif left_depth_diff <= 40:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video bends more (smaller angle = more bent)
                if left_max1 < left_max2:
                    interpretation['differences'].append(
                        f"Left leg loading depth shows {level} difference - Video 2 bends less ({left_depth_diff:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Left leg loading depth shows {level} difference - Video 2 bends more ({left_depth_diff:.1f}°)"
                    )
            
            # Compare right leg depth
            if right_depth_diff > 15:
                # Determine difference level
                if right_depth_diff <= 20:
                    level = "low"
                elif right_depth_diff <= 30:
                    level = "medium"
                elif right_depth_diff <= 40:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video bends more (smaller angle = more bent)
                if right_max1 < right_max2:
                    interpretation['differences'].append(
                        f"Right leg loading depth shows {level} difference - Video 2 bends less ({right_depth_diff:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Right leg loading depth shows {level} difference - Video 2 bends more ({right_depth_diff:.1f}°)"
                    )
        
        # Compare max angle timing and transition timing
        transition1 = video1.get('max_angle_to_transition', {})
        transition2 = video2.get('max_angle_to_transition', {})
        
        if transition1 and transition2:
            # Compare time to reach max angle
            left_max_frame1 = transition1.get('left_max_frame', 0)
            right_max_frame1 = transition1.get('right_max_frame', 0)
            both_max_frame1 = transition1.get('both_max_frame', 0)
            
            left_max_frame2 = transition2.get('left_max_frame', 0)
            right_max_frame2 = transition2.get('right_max_frame', 0)
            both_max_frame2 = transition2.get('both_max_frame', 0)
            
            # Calculate time to max angle (assuming 30fps for conversion)
            fps1 = video1.get('fps', 30.0)
            fps2 = video2.get('fps', 30.0)
            
            time_to_max1 = both_max_frame1 / fps1
            time_to_max2 = both_max_frame2 / fps2
            
            if abs(time_to_max1 - time_to_max2) > 0.1:
                max_timing_diff = abs(time_to_max1 - time_to_max2)
                # Determine difference level
                if max_timing_diff <= 0.2:
                    level = "low"
                elif max_timing_diff <= 0.4:
                    level = "medium"
                elif max_timing_diff <= 0.6:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video reaches max faster
                if time_to_max1 < time_to_max2:
                    interpretation['differences'].append(
                        f"Time to reach max angle shows {level} difference - Video 2 is slower ({max_timing_diff:.2f}s)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Time to reach max angle shows {level} difference - Video 2 is faster ({max_timing_diff:.2f}s)"
                    )
            
            # Compare transition timing (time from max to next phase)
            time_to_trans1 = transition1.get('time_to_transition', 0)
            time_to_trans2 = transition2.get('time_to_transition', 0)
            
            if abs(time_to_trans1 - time_to_trans2) > 0.1:
                timing_diff = abs(time_to_trans1 - time_to_trans2)
                # Determine difference level
                if timing_diff <= 0.3:
                    level = "low"
                elif timing_diff <= 0.5:
                    level = "medium"
                elif timing_diff <= 0.8:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has faster transition (Video 1 is reference)
                if time_to_trans1 < time_to_trans2:
                    interpretation['differences'].append(
                        f"Transition timing shows {level} difference - Video 2 is slower ({timing_diff:.2f}s)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Transition timing shows {level} difference - Video 2 is faster ({timing_diff:.2f}s)"
                    )
        

        
        return interpretation
    
    def _interpret_rising_analysis(self, rising_analysis: Dict) -> Dict:
        """Interpret rising phase analysis."""
        video1 = rising_analysis.get('video1', {})
        video2 = rising_analysis.get('video2', {})
        
        interpretation = {
            'comparisons': {},
            'differences': []
            # 'insights': [] # Removed
        }
        
        # Compare jump heights
        jump1 = video1.get('jump_analysis', {})
        jump2 = video2.get('jump_analysis', {})
        
        if jump1 and jump2:
            height1 = jump1.get('max_jump_height', 0)
            height2 = jump2.get('max_jump_height', 0)
            has_jump1 = jump1.get('has_significant_jump', True)
            has_jump2 = jump2.get('has_significant_jump', True)
            
            # Check if both videos have significant jumps
            if has_jump1 and has_jump2:
                # Both videos have significant jumps, compare heights
                if abs(height1 - height2) > 0.02:
                    height_diff = abs(height1 - height2)
                    # Determine difference level
                    if height_diff <= 0.03:
                        level = "low"
                    elif height_diff <= 0.05:
                        level = "medium"
                    elif height_diff <= 0.08:
                        level = "high"
                    else:
                        level = "very high"
                    
                    # Determine which video has higher jump (Video 1 is reference)
                    if height1 > height2:
                        interpretation['differences'].append(
                            f"Jump height shows {level} difference - Video 2 has lower jump ({height_diff:.3f})"
                        )
                    else:
                        interpretation['differences'].append(
                            f"Jump height shows {level} difference - Video 2 has higher jump ({height_diff:.3f})"
                        )
                
                # Determine which has higher jump (REMOVED)
                # if height1 > height2:
                #     interpretation['insights'].append("Video 1 achieves higher jump")
                # elif height2 > height1:
                #     interpretation['insights'].append("Video 2 achieves higher jump")
            
            elif not has_jump1 and not has_jump2:
                # Both videos don't have significant jumps
                interpretation['differences'].append(
                    "Both videos show minimal jumping - similar no-jump patterns"
                )
            
            elif has_jump1 and not has_jump2:
                # Video 1 jumps, Video 2 doesn't
                interpretation['differences'].append(
                    f"Video 2 does not jump while Video 1 shows significant jump ({height1:.3f})"
                )
            
            elif not has_jump1 and has_jump2:
                # Video 2 jumps, Video 1 doesn't
                interpretation['differences'].append(
                    f"Video 2 jumps while Video 1 shows minimal jump ({height2:.3f})"
                )
        
        # Compare ball position relative to hip at rising start
        ball_position1 = video1.get('ball_position_analysis', {})
        ball_position2 = video2.get('ball_position_analysis', {})
        
        if ball_position1 and ball_position2 and 'error' not in ball_position1 and 'error' not in ball_position2:
            # Compare ball height relative to hip at rising start
            ball_height1 = ball_position1.get('ball_height_relative_to_hip', 0)
            ball_height2 = ball_position2.get('ball_height_relative_to_hip', 0)
            
            if abs(ball_height1 - ball_height2) > 0.02:  # 2cm threshold
                ball_height_diff = abs(ball_height1 - ball_height2)
                # Determine difference level
                if ball_height_diff <= 0.05:
                    level = "low"
                elif ball_height_diff <= 0.1:
                    level = "medium"
                elif ball_height_diff <= 0.15:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has ball higher relative to hip (Video 1 is reference)
                if ball_height1 > ball_height2:
                    interpretation['differences'].append(
                        f"Ball height relative to hip at rising start shows {level} difference - Video 2 has ball lower ({ball_height_diff:.3f})"
                    )
                else:
                    interpretation['differences'].append(
                        f"Ball height relative to hip at rising start shows {level} difference - Video 2 has ball higher ({ball_height_diff:.3f})"
                    )
            
            # Compare ball horizontal position relative to hip at rising start
            ball_horizontal1 = ball_position1.get('ball_horizontal_relative_to_hip', 0)
            ball_horizontal2 = ball_position2.get('ball_horizontal_relative_to_hip', 0)
            
            if abs(ball_horizontal1 - ball_horizontal2) > 0.02:  # 2cm threshold
                ball_horizontal_diff = abs(ball_horizontal1 - ball_horizontal2)
                # Determine difference level
                if ball_horizontal_diff <= 0.05:
                    level = "low"
                elif ball_horizontal_diff <= 0.1:
                    level = "medium"
                elif ball_horizontal_diff <= 0.15:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has ball more forward relative to hip (Video 1 is reference)
                if ball_horizontal1 > ball_horizontal2:
                    interpretation['differences'].append(
                        f"Ball horizontal position relative to hip at rising start shows {level} difference - Video 2 has ball more backward ({ball_horizontal_diff:.3f})"
                    )
                else:
                    interpretation['differences'].append(
                        f"Ball horizontal position relative to hip at rising start shows {level} difference - Video 2 has ball more forward ({ball_horizontal_diff:.3f})"
                    )
        
        # Compare timing
        timing1 = video1.get('timing_analysis', {})
        timing2 = video2.get('timing_analysis', {})
        
        if timing1 and timing2:
            windup1 = timing1.get('windup_time', 0)
            windup2 = timing2.get('windup_time', 0)
            
            if abs(windup1 - windup2) > 0.1:
                windup_diff = abs(windup1 - windup2)
                # Determine difference level
                if windup_diff <= 0.3:
                    level = "low"
                elif windup_diff <= 0.5:
                    level = "medium"
                elif windup_diff <= 0.8:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has faster windup (Video 1 is reference)
                if windup1 < windup2:
                    interpretation['differences'].append(
                        f"Windup timing shows {level} difference - Video 2 is slower ({windup_diff:.2f}s)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Windup timing shows {level} difference - Video 2 is faster ({windup_diff:.2f}s)"
                    )

        # Compare windup trajectory curvature and path length
        windup1 = video1.get('windup_trajectory', {})
        windup2 = video2.get('windup_trajectory', {})

        if windup1 and windup2 and 'error' not in windup1 and 'error' not in windup2:
            # Compare trajectory curvature
            curvature1 = windup1.get('trajectory_curvature', 0)
            curvature2 = windup2.get('trajectory_curvature', 0)
            
            if abs(curvature1 - curvature2) > 0.001:  # Small threshold for curvature
                curvature_diff = abs(curvature1 - curvature2)
                # Determine difference level
                if curvature_diff <= 0.002:
                    level = "low"
                elif curvature_diff <= 0.005:
                    level = "medium"
                elif curvature_diff <= 0.01:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has more curved trajectory (Video 1 is reference)
                if curvature1 > curvature2:
                    interpretation['differences'].append(
                        f"Windup trajectory curvature shows {level} difference - Video 2 has straighter path ({curvature_diff:.4f})"
                    )
                else:
                    interpretation['differences'].append(
                        f"Windup trajectory curvature shows {level} difference - Video 2 has more curved path ({curvature_diff:.4f})"
                    )
            
            # Compare trajectory path length
            path_length1 = windup1.get('trajectory_path_length', 0)
            path_length2 = windup2.get('trajectory_path_length', 0)
            
            if abs(path_length1 - path_length2) > 0.01:  # Small threshold for path length
                path_length_diff = abs(path_length1 - path_length2)
                # Determine difference level
                if path_length_diff <= 0.02:
                    level = "low"
                elif path_length_diff <= 0.05:
                    level = "medium"
                elif path_length_diff <= 0.1:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has longer path (Video 1 is reference)
                if path_length1 > path_length2:
                    interpretation['differences'].append(
                        f"Windup trajectory path length shows {level} difference - Video 2 has shorter path ({path_length_diff:.4f})"
                    )
                else:
                    interpretation['differences'].append(
                        f"Windup trajectory path length shows {level} difference - Video 2 has longer path ({path_length_diff:.4f})"
                    )
        
        # Compare max jump timing and setup timing relative to max jump
        jump1 = video1.get('jump_analysis', {})
        jump2 = video2.get('jump_analysis', {})
        
        if jump1 and jump2:
            # Compare time to reach max jump height
            max_time1 = jump1.get('max_height_time', 0)
            max_time2 = jump2.get('max_height_time', 0)
            
            if abs(max_time1 - max_time2) > 0.1:
                max_timing_diff = abs(max_time1 - max_time2)
                # Determine difference level
                if max_timing_diff <= 0.2:
                    level = "low"
                elif max_timing_diff <= 0.4:
                    level = "medium"
                elif max_timing_diff <= 0.6:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video reaches max jump faster (Video 1 is reference)
                if max_time1 < max_time2:
                    interpretation['differences'].append(
                        f"Time to reach max jump height shows {level} difference - Video 2 is slower ({max_timing_diff:.2f}s)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Time to reach max jump height shows {level} difference - Video 2 is faster ({max_timing_diff:.2f}s)"
                    )
        
        # Compare dip point analysis
        dip1 = video1.get('dip_point_analysis', {})
        dip2 = video2.get('dip_point_analysis', {})
        
        if dip1 and dip2 and 'error' not in dip1 and 'error' not in dip2:
            # Compare arm angles at setup point

                # Compare shoulder-elbow-wrist angle
            sew1 = dip1.get('dip_shoulder_elbow_wrist', 0)
            sew2 = dip2.get('dip_shoulder_elbow_wrist', 0)

            if abs(sew1 - sew2) > 5:  # 5 degree threshold
                sew_diff = abs(sew1 - sew2)
                # Determine difference level
                if sew_diff <= 10:
                    level = "low"
                elif sew_diff <= 20:
                    level = "medium"
                elif sew_diff <= 30:
                    level = "high"
                else:
                    level = "very high"

                # Determine which video has larger angle (Video 1 is reference)
                if sew1 > sew2:
                    interpretation['differences'].append(
                        f"Dip point shoulder-elbow-wrist angle shows {level} difference - Video 2 has smaller angle ({sew_diff:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Dip point shoulder-elbow-wrist angle shows {level} difference - Video 2 has larger angle ({sew_diff:.1f}°)"
                    )

            # Compare arm-torso angle
            at1 = dip1.get('dip_arm_torso_angle', 0)
            at2 = dip2.get('dip_arm_torso_angle', 0)
            
            if abs(at1 - at2) > 5:  # 5 degree threshold
                at_diff = abs(at1 - at2)
                # Determine difference level
                if at_diff <= 10:
                    level = "low"
                elif at_diff <= 20:
                    level = "medium"
                elif at_diff <= 30:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has larger arm-torso angle (Video 1 is reference)
                if at1 > at2:
                    interpretation['differences'].append(
                        f"Dip point arm-torso angle shows {level} difference - Video 2 has smaller angle ({at_diff:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Dip point arm-torso angle shows {level} difference - Video 2 has larger angle ({at_diff:.1f}°)"
                    )

        # Compare setup point analysis
        setup1 = video1.get('setup_point_analysis', {})
        setup2 = video2.get('setup_point_analysis', {})
        
        if setup1 and setup2 and 'error' not in setup1 and 'error' not in setup2:
            # Compare arm angles at setup point
            arm_angles1 = setup1.get('arm_angles', {})
            arm_angles2 = setup2.get('arm_angles', {})
            
            if arm_angles1 and arm_angles2:
                # Compare shoulder-elbow-wrist angle
                sew1 = arm_angles1.get('shoulder_elbow_wrist', 0)
                sew2 = arm_angles2.get('shoulder_elbow_wrist', 0)
                
                if abs(sew1 - sew2) > 5:  # 5 degree threshold
                    sew_diff = abs(sew1 - sew2)
                    # Determine difference level
                    if sew_diff <= 10:
                        level = "low"
                    elif sew_diff <= 20:
                        level = "medium"
                    elif sew_diff <= 30:
                        level = "high"
                    else:
                        level = "very high"
                    
                    # Determine which video has larger angle (Video 1 is reference)
                    if sew1 > sew2:
                        interpretation['differences'].append(
                            f"Setup point shoulder-elbow-wrist angle shows {level} difference - Video 2 has smaller angle ({sew_diff:.1f}°)"
                        )
                    else:
                        interpretation['differences'].append(
                            f"Setup point shoulder-elbow-wrist angle shows {level} difference - Video 2 has larger angle ({sew_diff:.1f}°)"
                        )
                
                # Compare arm-torso angle
                at1 = arm_angles1.get('arm_torso_angle', 0)
                at2 = arm_angles2.get('arm_torso_angle', 0)
                
                if abs(at1 - at2) > 5:  # 5 degree threshold
                    at_diff = abs(at1 - at2)
                    # Determine difference level
                    if at_diff <= 10:
                        level = "low"
                    elif at_diff <= 20:
                        level = "medium"
                    elif at_diff <= 30:
                        level = "high"
                    else:
                        level = "very high"
                    
                    # Determine which video has larger arm-torso angle (Video 1 is reference)
                    if at1 > at2:
                        interpretation['differences'].append(
                            f"Setup point arm-torso angle shows {level} difference - Video 2 has smaller angle ({at_diff:.1f}°)"
                        )
                    else:
                        interpretation['differences'].append(
                            f"Setup point arm-torso angle shows {level} difference - Video 2 has larger angle ({at_diff:.1f}°)"
                        )

            # Compare ball position relative to eyes at setup point
            ball_eye1 = setup1.get('ball_eye_position', {})
            ball_eye2 = setup2.get('ball_eye_position', {})
            
            if ball_eye1 and ball_eye2 and 'error' not in ball_eye1 and 'error' not in ball_eye2:
                # Compare horizontal distance
                rel_x1 = ball_eye1.get('relative_x', 0)
                rel_x2 = ball_eye2.get('relative_x', 0)
                
                if abs(rel_x1 - rel_x2) > 0.02:  # 2cm threshold
                    rel_x_diff = abs(rel_x1 - rel_x2)
                    # Determine difference level
                    if rel_x_diff <= 0.05:
                        level = "low"
                    elif rel_x_diff <= 0.1:
                        level = "medium"
                    elif rel_x_diff <= 0.15:
                        level = "high"
                    else:
                        level = "very high"
                    
                    # Determine which video has ball more to the right relative to eyes (Video 1 is reference)
                    if rel_x1 > rel_x2:
                        interpretation['differences'].append(
                            f"Setup point ball horizontal position relative to eyes shows {level} difference - Video 2 has ball more to the left ({rel_x_diff:.3f})"
                        )
                    else:
                        interpretation['differences'].append(
                            f"Setup point ball horizontal position relative to eyes shows {level} difference - Video 2 has ball more to the right ({rel_x_diff:.3f})"
                        )

                # Compare vertical distance
                rel_y1 = ball_eye1.get('relative_y', 0)
                rel_y2 = ball_eye2.get('relative_y', 0)
                
                if abs(rel_y1 - rel_y2) > 0.02:  # 2cm threshold
                    rel_y_diff = abs(rel_y1 - rel_y2)
                    # Determine difference level
                    if rel_y_diff <= 0.05:
                        level = "low"
                    elif rel_y_diff <= 0.1:
                        level = "medium"
                    elif rel_y_diff <= 0.15:
                        level = "high"
                    else:
                        level = "very high"
                    
                    # Determine which video has ball higher relative to eyes (Video 1 is reference)
                    if rel_y1 > rel_y2:
                        interpretation['differences'].append(
                            f"Setup point ball vertical position relative to eyes shows {level} difference - Video 2 has ball lower ({rel_y_diff:.3f})"
                        )
                    else:
                        interpretation['differences'].append(
                            f"Setup point ball vertical position relative to eyes shows {level} difference - Video 2 has ball higher ({rel_y_diff:.3f})"
                        )

        return interpretation
    
    def _interpret_release_analysis(self, release_analysis: Dict) -> Dict:
        """Interpret release phase analysis."""
        video1 = release_analysis.get('video1', {})
        video2 = release_analysis.get('video2', {})
        
        interpretation = {
            'comparisons': {},
            'differences': []
            # 'insights': [] # Removed
        }
        
        # Compare release timing relative to max jump
        timing1 = video1.get('release_timing', {})
        timing2 = video2.get('release_timing', {})
        
        if timing1 and timing2:
            relative_timing1 = timing1.get('relative_timing', 0)
            relative_timing2 = timing2.get('relative_timing', 0)
            
            if abs(relative_timing1 - relative_timing2) > 0.1:
                relative_timing_diff = abs(relative_timing1 - relative_timing2)
                # Determine difference level
                if relative_timing_diff <= 0.2:
                    level = "low"
                elif relative_timing_diff <= 0.4:
                    level = "medium"
                elif relative_timing_diff <= 0.6:
                    level = "high"
                else:
                    level = "very high"
                
                # Show actual timing differences (negative = before max jump, positive = after max jump)
                timing_diff_video1 = relative_timing1
                timing_diff_video2 = relative_timing2
                
                interpretation['differences'].append(
                    f"Release timing relative to max jump shows {level} difference - Video 1: {timing_diff_video1:.2f}s, Video 2: {timing_diff_video2:.2f}s"
                )
        
        # Compare body angles at release
        body1 = video1.get('body_analysis', {})
        body2 = video2.get('body_analysis', {})
        
        if body1 and body2:
            # Compare body tilt (upper body lean)
            tilt1 = body1.get('body_tilt', {}).get('average', 0)
            tilt2 = body2.get('body_tilt', {}).get('average', 0)
            
            if abs(tilt1 - tilt2) > 10:
                tilt_diff = abs(tilt1 - tilt2)
                # Determine difference level
                if tilt_diff <= 15:
                    level = "low"
                elif tilt_diff <= 25:
                    level = "medium"
                elif tilt_diff <= 35:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has more tilt (Video 1 is reference)
                if tilt1 > tilt2:
                    interpretation['differences'].append(
                        f"Upper body lean at release shows {level} difference - Video 2 leans less forward ({tilt_diff:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Upper body lean at release shows {level} difference - Video 2 leans more forward ({tilt_diff:.1f}°)"
                    )
            
            # Compare leg angles at release
            leg_angles1 = body1.get('leg_angles', {})
            leg_angles2 = body2.get('leg_angles', {})
            
            if leg_angles1 and leg_angles2:
                # Compare left thigh angle (hip-knee)
                left_thigh1 = leg_angles1.get('left_thigh_angle', {}).get('average', 0)
                left_thigh2 = leg_angles2.get('left_thigh_angle', {}).get('average', 0)
                
                if abs(left_thigh1 - left_thigh2) > 10:
                    thigh_diff = abs(left_thigh1 - left_thigh2)
                    # Determine difference level
                    if thigh_diff <= 15:
                        level = "low"
                    elif thigh_diff <= 25:
                        level = "medium"
                    elif thigh_diff <= 35:
                        level = "high"
                    else:
                        level = "very high"
                    
                    # Determine which video has more bent left thigh (Video 1 is reference)
                    if left_thigh1 < left_thigh2:  # Smaller angle = more bent
                        interpretation['differences'].append(
                            f"Left thigh angle at release shows {level} difference - Video 2 bends less ({thigh_diff:.1f}°)"
                        )
                    else:
                        interpretation['differences'].append(
                            f"Left thigh angle at release shows {level} difference - Video 2 bends more ({thigh_diff:.1f}°)"
                        )
                
                # Compare right thigh angle (hip-knee)
                right_thigh1 = leg_angles1.get('right_thigh_angle', {}).get('average', 0)
                right_thigh2 = leg_angles2.get('right_thigh_angle', {}).get('average', 0)
                
                if abs(right_thigh1 - right_thigh2) > 10:
                    thigh_diff = abs(right_thigh1 - right_thigh2)
                    # Determine difference level
                    if thigh_diff <= 15:
                        level = "low"
                    elif thigh_diff <= 25:
                        level = "medium"
                    elif thigh_diff <= 35:
                        level = "high"
                    else:
                        level = "very high"
                    
                    # Determine which video has more bent right thigh (Video 1 is reference)
                    if right_thigh1 < right_thigh2:  # Smaller angle = more bent
                        interpretation['differences'].append(
                            f"Right thigh angle at release shows {level} difference - Video 2 bends less ({thigh_diff:.1f}°)"
                        )
                    else:
                        interpretation['differences'].append(
                            f"Right thigh angle at release shows {level} difference - Video 2 bends more ({thigh_diff:.1f}°)"
                        )
                
                # Compare left leg angle (hip-knee-ankle)
                left_leg1 = leg_angles1.get('left_leg_angle', {}).get('average', 0)
                left_leg2 = leg_angles2.get('left_leg_angle', {}).get('average', 0)
                
                if abs(left_leg1 - left_leg2) > 10:
                    leg_diff = abs(left_leg1 - left_leg2)
                    # Determine difference level
                    if leg_diff <= 15:
                        level = "low"
                    elif leg_diff <= 25:
                        level = "medium"
                    elif leg_diff <= 35:
                        level = "high"
                    else:
                        level = "very high"
                    
                    # Determine which video has more bent left leg (Video 1 is reference)
                    if left_leg1 < left_leg2:  # Smaller angle = more bent
                        interpretation['differences'].append(
                            f"Left leg angle at release shows {level} difference - Video 2 bends less ({leg_diff:.1f}°)"
                        )
                    else:
                        interpretation['differences'].append(
                            f"Left leg angle at release shows {level} difference - Video 2 bends more ({leg_diff:.1f}°)"
                        )
                
                # Compare right leg angle (hip-knee-ankle)
                right_leg1 = leg_angles1.get('right_leg_angle', {}).get('average', 0)
                right_leg2 = leg_angles2.get('right_leg_angle', {}).get('average', 0)
                
                if abs(right_leg1 - right_leg2) > 10:
                    leg_diff = abs(right_leg1 - right_leg2)
                    # Determine difference level
                    if leg_diff <= 15:
                        level = "low"
                    elif leg_diff <= 25:
                        level = "medium"
                    elif leg_diff <= 35:
                        level = "high"
                    else:
                        level = "very high"
                    
                    # Determine which video has more bent right leg (Video 1 is reference)
                    if right_leg1 < right_leg2:  # Smaller angle = more bent
                        interpretation['differences'].append(
                            f"Right leg angle at release shows {level} difference - Video 2 bends less ({leg_diff:.1f}°)"
                        )
                    else:
                        interpretation['differences'].append(
                            f"Right leg angle at release shows {level} difference - Video 2 bends more ({leg_diff:.1f}°)"
                        )
            
            # Compare additional angles (REMOVED - duplicates of body_tilt and leg_angles)
            # Upper body angle - same as body_tilt
            # Waist angle - same as body_tilt  
            # Thigh angle - average of left_thigh_angle and right_thigh_angle
            
            # Shoulder-elbow-wrist angle
            shoulder_elbow_wrist1 = body1.get('shoulder_elbow_wrist_angle', {}).get('average', 0)
            shoulder_elbow_wrist2 = body2.get('shoulder_elbow_wrist_angle', {}).get('average', 0)
            
            if abs(shoulder_elbow_wrist1 - shoulder_elbow_wrist2) > 10:
                angle_diff = abs(shoulder_elbow_wrist1 - shoulder_elbow_wrist2)
                # Determine difference level
                if angle_diff <= 15:
                    level = "low"
                elif angle_diff <= 25:
                    level = "medium"
                elif angle_diff <= 35:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has larger arm angle (Video 1 is reference)
                if shoulder_elbow_wrist1 > shoulder_elbow_wrist2:
                    interpretation['differences'].append(
                        f"Shoulder-elbow-wrist angle at release shows {level} difference - Video 2 has smaller angle ({angle_diff:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Shoulder-elbow-wrist angle at release shows {level} difference - Video 2 has larger angle ({angle_diff:.1f}°)"
                    )
            
            # Wrist-shoulder-hip angle
            wrist_shoulder_hip1 = body1.get('wrist_shoulder_hip_angle', {}).get('average', 0)
            wrist_shoulder_hip2 = body2.get('wrist_shoulder_hip_angle', {}).get('average', 0)
            
            if abs(wrist_shoulder_hip1 - wrist_shoulder_hip2) > 10:
                angle_diff = abs(wrist_shoulder_hip1 - wrist_shoulder_hip2)
                # Determine difference level
                if angle_diff <= 15:
                    level = "low"
                elif angle_diff <= 25:
                    level = "medium"
                elif angle_diff <= 35:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has larger angle (Video 1 is reference)
                if wrist_shoulder_hip1 > wrist_shoulder_hip2:
                    interpretation['differences'].append(
                        f"Wrist-shoulder-hip angle at release shows {level} difference - Video 2 has smaller angle ({angle_diff:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Wrist-shoulder-hip angle at release shows {level} difference - Video 2 has larger angle ({angle_diff:.1f}°)"
                    )
        
        # Compare arm angles
        arm_angles1 = video1.get('arm_angles', {})
        arm_angles2 = video2.get('arm_angles', {})
        
        if arm_angles1 and arm_angles2:
            right_arm1 = arm_angles1.get('right_arm', {})
            right_arm2 = arm_angles2.get('right_arm', {})
            
            if right_arm1 and right_arm2:
                torso_angle1 = right_arm1.get('torso_angle', {}).get('average', 0)
                torso_angle2 = right_arm2.get('torso_angle', {}).get('average', 0)
                
                if abs(torso_angle1 - torso_angle2) > 5:
                    arm_diff = abs(torso_angle1 - torso_angle2)
                    # Determine difference level
                    if arm_diff <= 10:
                        level = "low"
                    elif arm_diff <= 20:
                        level = "medium"
                    elif arm_diff <= 30:
                        level = "high"
                    else:
                        level = "very high"
                    
                    # Determine which video has larger arm angle (Video 1 is reference)
                    if torso_angle1 > torso_angle2:
                        interpretation['differences'].append(
                            f"Right arm angle shows {level} difference - Video 2 has smaller angle ({arm_diff:.1f}°)"
                        )
                    else:
                        interpretation['differences'].append(
                            f"Right arm angle shows {level} difference - Video 2 has larger angle ({arm_diff:.1f}°)"
                        )
        
        # Compare ball position
        ball_pos1 = video1.get('ball_position', {})
        ball_pos2 = video2.get('ball_position', {})
        
        if ball_pos1 and ball_pos2:
            rel_x1 = ball_pos1.get('average_relative_x', 0)
            rel_x2 = ball_pos2.get('average_relative_x', 0)
            rel_y1 = ball_pos1.get('average_relative_y', 0)
            rel_y2 = ball_pos2.get('average_relative_y', 0)
            
            if abs(rel_x1 - rel_x2) > 0.1:
                ball_x_diff = abs(rel_x1 - rel_x2)
                # Determine difference level
                if ball_x_diff <= 0.15:
                    level = "low"
                elif ball_x_diff <= 0.25:
                    level = "medium"
                elif ball_x_diff <= 0.35:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has ball further/closer horizontally (Video 1 is reference)
                if rel_x1 > rel_x2:
                    interpretation['differences'].append(
                        f"Ball horizontal position shows {level} difference - Video 2 has ball closer ({ball_x_diff:.3f})"
                    )
                else:
                    interpretation['differences'].append(
                        f"Ball horizontal position shows {level} difference - Video 2 has ball further ({ball_x_diff:.3f})"
                    )
            
            if abs(rel_y1 - rel_y2) > 0.1:
                ball_y_diff = abs(rel_y1 - rel_y2)
                # Determine difference level
                if ball_y_diff <= 0.15:
                    level = "low"
                elif ball_y_diff <= 0.25:
                    level = "medium"
                elif ball_y_diff <= 0.35:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has ball higher/lower vertically (Video 1 is reference)
                if rel_y1 > rel_y2:
                    interpretation['differences'].append(
                        f"Ball vertical position shows {level} difference - Video 2 has ball lower ({ball_y_diff:.3f})"
                    )
                else:
                    interpretation['differences'].append(
                        f"Ball vertical position shows {level} difference - Video 2 has ball higher ({ball_y_diff:.3f})"
                    )
        
        # Compare ball vector at release
        ball_vector1 = ball_pos1.get('ball_vector', {}) if ball_pos1 else {}
        ball_vector2 = ball_pos2.get('ball_vector', {}) if ball_pos2 else {}
        
        if ball_vector1 and ball_vector2:
            # Compare ball vector magnitude
            magnitude1 = ball_vector1.get('average_magnitude', 0)
            magnitude2 = ball_vector2.get('average_magnitude', 0)
            
            if abs(magnitude1 - magnitude2) > 0.01:
                magnitude_diff = abs(magnitude1 - magnitude2)
                # Determine difference level
                if magnitude_diff <= 0.02:
                    level = "low"
                elif magnitude_diff <= 0.05:
                    level = "medium"
                elif magnitude_diff <= 0.1:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has larger ball velocity (Video 1 is reference)
                if magnitude1 > magnitude2:
                    interpretation['differences'].append(
                        f"Ball velocity magnitude shows {level} difference - Video 2 has slower ball ({magnitude_diff:.4f})"
                    )
                else:
                    interpretation['differences'].append(
                        f"Ball velocity magnitude shows {level} difference - Video 2 has faster ball ({magnitude_diff:.4f})"
                    )
            
            # Compare ball vector angle
            angle1 = ball_vector1.get('average_angle', 0)
            angle2 = ball_vector2.get('average_angle', 0)
            
            if abs(angle1 - angle2) > 5:
                angle_diff = abs(angle1 - angle2)
                # Determine difference level
                if angle_diff <= 10:
                    level = "low"
                elif angle_diff <= 20:
                    level = "medium"
                elif angle_diff <= 30:
                    level = "high"
                else:
                    level = "very high"
                
                # Determine which video has higher/lower ball trajectory (Video 1 is reference)
                if angle1 > angle2:
                    interpretation['differences'].append(
                        f"Ball trajectory angle shows {level} difference - Video 2 has lower trajectory ({angle_diff:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Ball trajectory angle shows {level} difference - Video 2 has higher trajectory ({angle_diff:.1f}°)"
                    )
        

        
        return interpretation
    
    def _interpret_follow_through_analysis(self, follow_through_analysis: Dict) -> Dict:
        """
        Interpret follow-through analysis results.
        
        Args:
            follow_through_analysis: Dictionary containing follow-through analysis results
            
        Returns:
            Dictionary containing interpreted follow-through analysis
        """
        interpretation = {
            'differences': []
            # 'insights': [] # Removed
        }
        
        video1 = follow_through_analysis.get('video1', {})
        video2 = follow_through_analysis.get('video2', {})
        
        if 'error' in video1 or 'error' in video2:
            if 'error' in video1:
                interpretation['differences'].append(f"Video 1 follow-through analysis error: {video1['error']}")
            if 'error' in video2:
                interpretation['differences'].append(f"Video 2 follow-through analysis error: {video2['error']}")
            return interpretation
        
        # Compare stability analysis
        stability1 = video1.get('stability_analysis', {})
        stability2 = video2.get('stability_analysis', {})
        
        if 'error' not in stability1 and 'error' not in stability2:
            # Compare overall stable duration
            overall_stable1 = stability1.get('overall_stable_duration', 0)
            overall_stable2 = stability2.get('overall_stable_duration', 0)
            
            if overall_stable1 > 0 and overall_stable2 > 0:
                stable_diff = abs(overall_stable1 - overall_stable2)
                level = self._get_difference_level(stable_diff, 0.1, 0.2)
                
                if overall_stable1 > overall_stable2:
                    interpretation['differences'].append(
                        f"Overall form stability shows {level} difference - Video 1 maintains stable form longer ({overall_stable1:.3f}s vs {overall_stable2:.3f}s)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Overall form stability shows {level} difference - Video 2 maintains stable form longer ({overall_stable2:.3f}s vs {overall_stable1:.3f}s)"
                    )
            
            # Compare arm stable duration
            arm_stable1 = stability1.get('arm_stable_duration', 0)
            arm_stable2 = stability2.get('arm_stable_duration', 0)
            
            if arm_stable1 > 0 and arm_stable2 > 0:
                arm_stable_diff = abs(arm_stable1 - arm_stable2)
                level = self._get_difference_level(arm_stable_diff, 0.1, 0.2)
                
                if arm_stable1 > arm_stable2:
                    interpretation['differences'].append(
                        f"Arm stability shows {level} difference - Video 1 maintains stable arm position longer ({arm_stable1:.3f}s vs {arm_stable2:.3f}s)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Arm stability shows {level} difference - Video 2 maintains stable arm position longer ({arm_stable2:.3f}s vs {arm_stable1:.3f}s)"
                    )
            
            # Compare other body stable duration
            other_stable1 = stability1.get('other_body_stable_duration', 0)
            other_stable2 = stability2.get('other_body_stable_duration', 0)
            
            if other_stable1 > 0 and other_stable2 > 0:
                other_stable_diff = abs(other_stable1 - other_stable2)
                level = self._get_difference_level(other_stable_diff, 0.1, 0.2)
                
                if other_stable1 > other_stable2:
                    interpretation['differences'].append(
                        f"Body stability shows {level} difference - Video 1 maintains stable body position longer ({other_stable1:.3f}s vs {other_stable2:.3f}s)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Body stability shows {level} difference - Video 2 maintains stable body position longer ({other_stable2:.3f}s vs {other_stable1:.3f}s)"
                    )
        
        # Compare max elbow angle analysis
        max_elbow1 = video1.get('max_elbow_angle_analysis', {})
        max_elbow2 = video2.get('max_elbow_angle_analysis', {})
        
        if 'error' not in max_elbow1 and 'error' not in max_elbow2:
            # Compare max elbow angle
            max_angle1 = max_elbow1.get('max_elbow_angle', 0)
            max_angle2 = max_elbow2.get('max_elbow_angle', 0)
            
            if max_angle1 > 0 and max_angle2 > 0:
                angle_diff = abs(max_angle1 - max_angle2)
                level = self._get_difference_level(angle_diff, 5.0, 10.0)
                
                if max_angle1 > max_angle2:
                    interpretation['differences'].append(
                        f"Maximum elbow angle shows {level} difference - Video 1 has higher max angle ({max_angle1:.1f}° vs {max_angle2:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Maximum elbow angle shows {level} difference - Video 2 has higher max angle ({max_angle2:.1f}° vs {max_angle1:.1f}°)"
                    )
            
            # Compare overall angles standard deviation at max elbow point
            overall_std1 = max_elbow1.get('overall_angles_std', 0)
            overall_std2 = max_elbow2.get('overall_angles_std', 0)
            
            if overall_std1 > 0 and overall_std2 > 0:
                std_diff = abs(overall_std1 - overall_std2)
                level = self._get_difference_level(std_diff, 2.0, 5.0)
                
                if overall_std1 < overall_std2:
                    interpretation['differences'].append(
                        f"Form consistency at max elbow shows {level} difference - Video 1 has more consistent angles (std: {overall_std1:.1f}° vs {overall_std2:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Form consistency at max elbow shows {level} difference - Video 2 has more consistent angles (std: {overall_std2:.1f}° vs {overall_std1:.1f}°)"
                    )
        
        # Compare arm stability statistics
        arm_stability1 = video1.get('arm_stability', {})
        arm_stability2 = video2.get('arm_stability', {})
        
        if 'error' not in arm_stability1 and 'error' not in arm_stability2:
            # Compare elbow angle standard deviation
            elbow_std1 = arm_stability1.get('elbow_angle', {}).get('std', 0)
            elbow_std2 = arm_stability2.get('elbow_angle', {}).get('std', 0)
            
            if elbow_std1 > 0 and elbow_std2 > 0:
                elbow_std_diff = abs(elbow_std1 - elbow_std2)
                level = self._get_difference_level(elbow_std_diff, 2.0, 5.0)
                
                if elbow_std1 < elbow_std2:
                    interpretation['differences'].append(
                        f"Elbow angle consistency shows {level} difference - Video 1 has more stable elbow (std: {elbow_std1:.1f}° vs {elbow_std2:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Elbow angle consistency shows {level} difference - Video 2 has more stable elbow (std: {elbow_std2:.1f}° vs {elbow_std1:.1f}°)"
                    )
            
            # Compare shoulder angle standard deviation
            shoulder_std1 = arm_stability1.get('shoulder_angle', {}).get('std', 0)
            shoulder_std2 = arm_stability2.get('shoulder_angle', {}).get('std', 0)
            
            if shoulder_std1 > 0 and shoulder_std2 > 0:
                shoulder_std_diff = abs(shoulder_std1 - shoulder_std2)
                level = self._get_difference_level(shoulder_std_diff, 2.0, 5.0)
                
                if shoulder_std1 < shoulder_std2:
                    interpretation['differences'].append(
                        f"Shoulder angle consistency shows {level} difference - Video 1 has more stable shoulder (std: {shoulder_std1:.1f}° vs {shoulder_std2:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Shoulder angle consistency shows {level} difference - Video 2 has more stable shoulder (std: {shoulder_std2:.1f}° vs {shoulder_std1:.1f}°)"
                    )
        
        # Generate insights (REMOVED)
        # if not interpretation['differences']:
        #     interpretation['insights'].append("Both forms show similar follow-through stability and consistency")
        # else:
        #     interpretation['insights'].append("Follow-through phase reveals differences in form maintenance and stability")
        
        return interpretation
    
    def _interpret_landing_analysis(self, landing_analysis: Dict) -> Dict:
        """
        Interpret landing analysis results.
        
        Args:
            landing_analysis: Dictionary containing landing analysis results
            
        Returns:
            Dictionary containing interpreted landing analysis
        """
        interpretation = {
            'differences': []
            # 'insights': [] # Removed
        }
        
        video1 = landing_analysis.get('video1', {})
        video2 = landing_analysis.get('video2', {})
        
        if 'error' in video1 or 'error' in video2:
            if 'error' in video1:
                interpretation['differences'].append(f"Video 1 landing analysis error: {video1['error']}")
            if 'error' in video2:
                interpretation['differences'].append(f"Video 2 landing analysis error: {video2['error']}")
            return interpretation
        
        # Compare landing detection
        landing_detection1 = video1.get('landing_detection', {})
        landing_detection2 = video2.get('landing_detection', {})
        
        if 'error' not in landing_detection1 and 'error' not in landing_detection2:
            # Compare landing detection
            landing_detected1 = landing_detection1.get('landing_detected', False)
            landing_detected2 = landing_detection2.get('landing_detected', False)
            
            if landing_detected1 and landing_detected2:
                # Both videos have landing detected
                stable_landing1 = landing_detection1.get('stable_landing', False)
                stable_landing2 = landing_detection2.get('stable_landing', False)
                
                if stable_landing1 and stable_landing2:
                    interpretation['differences'].append("Both videos show stable landing after follow-through")
                elif stable_landing1 and not stable_landing2:
                    interpretation['differences'].append("Video 1 shows stable landing while Video 2 shows unstable landing")
                elif not stable_landing1 and stable_landing2:
                    interpretation['differences'].append("Video 2 shows stable landing while Video 1 shows unstable landing")
                else:
                    interpretation['differences'].append("Both videos show unstable landing after follow-through")
            
            elif landing_detected1 and not landing_detected2:
                interpretation['differences'].append("Video 1 shows landing while Video 2 does not land after follow-through")
            
            elif not landing_detected1 and landing_detected2:
                interpretation['differences'].append("Video 2 shows landing while Video 1 does not land after follow-through")
            
            else:
                interpretation['differences'].append("Neither video shows landing after follow-through")
        
        # Compare landing position analysis (REMOVED)
        # landing_position1 = video1.get('landing_position_analysis', {})
        # landing_position2 = video2.get('landing_position_analysis', {})
        # 
        # if 'error' not in landing_position1 and 'error' not in landing_position2:
        #     # Compare left foot landing position
        #     left_foot1 = landing_position1.get('left_foot', {})
        #     left_foot2 = landing_position2.get('left_foot', {})
        #     
        #     if left_foot1 and left_foot2:
        #         left_avg_dist1 = left_foot1.get('average_distance_from_setup', 0)
        #         left_avg_dist2 = left_foot2.get('average_distance_from_setup', 0)
        #         
        #         if abs(left_avg_dist1 - left_avg_dist2) > 0.01:
        #             left_dist_diff = abs(left_avg_dist1 - left_avg_dist2)
        #             level = self._get_difference_level(left_dist_diff, 0.02, 0.05)
        #             
        #             if left_avg_dist1 < left_avg_dist2:
        #                 interpretation['differences'].append(
        #                     f"Left foot landing position shows {level} difference - Video 1 lands closer to setup position ({left_dist_diff:.3f})"
        #                 )
        #             else:
        #                 interpretation['differences'].append(
        #                     f"Left foot landing position shows {level} difference - Video 2 lands closer to setup position ({left_dist_diff:.3f})"
        #                 )
        #     
        #     # Compare right foot landing position
        #     right_foot1 = landing_position1.get('right_foot', {})
        #     right_foot2 = landing_position2.get('right_foot', {})
        #     
        #     if right_foot1 and right_foot2:
        #         right_avg_dist1 = right_foot1.get('average_distance_from_setup', 0)
        #         right_avg_dist2 = right_foot2.get('average_distance_from_setup', 0)
        #         
        #         if abs(right_avg_dist1 - right_avg_dist2) > 0.01:
        #             right_dist_diff = abs(right_avg_dist1 - right_avg_dist2)
        #             level = self._get_difference_level(right_dist_diff, 0.02, 0.05)
        #             
        #             if right_avg_dist1 < right_avg_dist2:
        #                 interpretation['differences'].append(
        #                     f"Right foot landing position shows {level} difference - Video 1 lands closer to setup position ({right_dist_diff:.3f})"
        #                 )
        #             else:
        #                 interpretation['differences'].append(
        #                     f"Right foot landing position shows {level} difference - Video 2 lands closer to setup position ({right_dist_diff:.3f})"
        #                 )
        
        # Compare landing torso angle analysis
        landing_torso1 = video1.get('landing_torso_analysis', {})
        landing_torso2 = video2.get('landing_torso_analysis', {})
        
        if 'error' not in landing_torso1 and 'error' not in landing_torso2:
            # Compare average torso angle at landing
            avg_torso_angle1 = landing_torso1.get('average_torso_angle', 0)
            avg_torso_angle2 = landing_torso2.get('average_torso_angle', 0)
            
            if abs(avg_torso_angle1 - avg_torso_angle2) > 5:
                torso_angle_diff = abs(avg_torso_angle1 - avg_torso_angle2)
                level = self._get_difference_level(torso_angle_diff, 10.0, 20.0)
                
                if avg_torso_angle1 < avg_torso_angle2:
                    interpretation['differences'].append(
                        f"Landing torso angle shows {level} difference - Video 1 has more upright torso at landing ({torso_angle_diff:.1f}°)"
                    )
                else:
                    interpretation['differences'].append(
                        f"Landing torso angle shows {level} difference - Video 2 has more upright torso at landing ({torso_angle_diff:.1f}°)"
                    )
            
            # Compare torso angle consistency (REMOVED)
            # torso_std1 = landing_torso1.get('std_torso_angle', 0)
            # torso_std2 = landing_torso2.get('std_torso_angle', 0)
            # 
            # if torso_std1 > 0 and torso_std2 > 0:
            #     torso_std_diff = abs(torso_std1 - torso_std2)
            #     level = self._get_difference_level(torso_std_diff, 2.0, 5.0)
            # 
            #     if torso_std1 < torso_std2:
            #         interpretation['differences'].append(
            #             f"Landing torso angle consistency shows {level} difference - Video 1 has more stable torso angle (std: {torso_std1:.1f}° vs {torso_std2:.1f}°)"
            #         )
            #     else:
            #         interpretation['differences'].append(
            #             f"Landing torso angle consistency shows {level} difference - Video 2 has more stable torso angle (std: {torso_std2:.1f}° vs {torso_std1:.1f}°)"
            #         )
        
        # Generate insights (REMOVED)
        # if not interpretation['differences']:
        #     interpretation['insights'].append("Both forms show similar landing patterns after follow-through")
        # else:
        #     interpretation['insights'].append("Landing phase reveals differences in foot positioning and body control")
        
        return interpretation
    
    def _get_difference_level(self, diff: float, low_threshold: float, medium_threshold: float) -> str:
        """
        Determine the level of difference based on thresholds.
        
        Args:
            diff: The difference value
            low_threshold: Threshold for low difference
            medium_threshold: Threshold for medium difference
            
        Returns:
            String indicating difference level: "low", "medium", "high", or "very high"
        """
        if diff <= low_threshold:
            return "low"
        elif diff <= medium_threshold:
            return "medium"
        elif diff <= medium_threshold * 2:
            return "high"
        else:
            return "very high"
    
    def _generate_key_insights(self, text_analysis: Dict) -> List[str]:
        """Generate key insights from all phases."""
        insights = []
        
        # Count significant differences
        total_differences = 0
        for phase, analysis in text_analysis.items():
            differences = analysis.get('differences', [])
            total_differences += len(differences)
        
        if total_differences == 0:
            insights.append("Both shooting forms show very similar characteristics across all phases")
        elif total_differences <= 3:
            insights.append(f"Shooting forms show minor differences ({total_differences} significant variations)")
        else:
            insights.append(f"Shooting forms show notable differences ({total_differences} significant variations)")
        
        # Phase-specific insights
        for phase, analysis in text_analysis.items():
            phase_insights = analysis.get('insights', [])
            if phase_insights:
                insights.extend([f"{phase.title()}: {insight}" for insight in phase_insights])
        
        return insights
    
    def _generate_recommendations(self, text_analysis: Dict) -> List[str]:
        """Generate coaching recommendations based on analysis."""
        recommendations = []
        
        # Analyze each phase for improvement opportunities
        for phase, analysis in text_analysis.items():
            differences = analysis.get('differences', [])
            
            if phase == 'setup' and differences:
                recommendations.append("Focus on consistent set-up positioning for better shot preparation")
            
            if phase == 'loading' and differences:
                recommendations.append("Work on consistent loading depth and timing for better power generation")
            
            if phase == 'rising' and differences:
                recommendations.append("Practice consistent jump height and body control during rising phase")
            
            if phase == 'release' and differences:
                recommendations.append("Maintain consistent release point and arm angles for better accuracy")
            
            if phase == 'follow_through' and differences:
                recommendations.append("Focus on maintaining stable follow-through position for better shot consistency")
        
        if not recommendations:
            recommendations.append("Both forms are well-executed. Focus on maintaining consistency")
        
        return recommendations
    
    def generate_llm_prompt(self, interpretation: Dict) -> str:
        """Generate a comprehensive prompt for LLM analysis."""
        text_analysis = interpretation.get('text_analysis', {})
        # key_insights = interpretation.get('key_insights', []) # Removed
        # recommendations = interpretation.get('recommendations', []) # Removed
        
        prompt = "BASKETBALL SHOOTING FORM ANALYSIS\n\n"
        prompt += "Based on the following detailed analysis of two basketball shooting forms, provide coaching recommendations:\n\n"
        
        # Add phase-by-phase analysis
        for phase, analysis in text_analysis.items():
            prompt += f"{phase.upper()} PHASE:\n"
            
            differences = analysis.get('differences', [])
            if differences:
                prompt += "Key Differences:\n"
                for diff in differences:
                    prompt += f"- {diff}\n"
            
            # insights = analysis.get('insights', []) # Removed
            # if insights: # Removed
            #     prompt += "Insights:\n" # Removed
            #     for insight in insights: # Removed
            #         prompt += f"- {insight}\n" # Removed
            
            prompt += "\n"
        
        # if key_insights: # Removed
        #     prompt += "OVERALL INSIGHTS:\n" # Removed
        #     for insight in key_insights: # Removed
        #         prompt += f"- {insight}\n" # Removed
        #     prompt += "\n" # Removed
        
        # if recommendations: # Removed
        #     prompt += "COACHING RECOMMENDATIONS:\n" # Removed
        #     for rec in recommendations: # Removed
        #         prompt += f"- {rec}\n" # Removed
        #     prompt += "\n" # Removed
        
        prompt += "Please provide detailed coaching advice based on this analysis, including specific drills and techniques to improve the shooting form."
        
        return prompt 