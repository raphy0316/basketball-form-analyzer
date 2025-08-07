"""
DTW Interpreter Extension

Extends existing AnalysisInterpreter with DTW analysis capabilities.
Works by adding DTW insights to existing interpretation results.
"""

try:
    from .analysis_interpreter import AnalysisInterpreter
    from .dtw_analysis.dtw_analyzer import DTWAnalyzer
    from .dtw_analysis.dtw_visualizer import DTWVisualizer
except ImportError:
    from analysis_interpreter import AnalysisInterpreter
    from dtw_analysis.dtw_analyzer import DTWAnalyzer
    from dtw_analysis.dtw_visualizer import DTWVisualizer
from typing import Dict, List, Optional
import numpy as np


class DTWInterpreterExtension:
    """
    Extension for existing AnalysisInterpreter that adds DTW analysis.
    
    Works alongside existing interpreter without modifying its behavior.
    """
    
    def __init__(self):
        self.dtw_analyzer = DTWAnalyzer()
        self.dtw_visualizer = DTWVisualizer()
        
        # Feature name mappings for user-friendly descriptions
        self.feature_names = {
            'ball_wrist_trajectory': 'Ball-Wrist Coordination',
            'shooting_arm_kinematics': 'Shooting Arm Motion',
            'lower_body_stability': 'Lower Body Foundation',
            'phase_timing_patterns': 'Timing Consistency', 
            'body_alignment': 'Overall Body Alignment'
        }
        
        # Feature descriptions
        self.feature_descriptions = {
            'ball_wrist_trajectory': 'How consistently the ball and shooting hand move together throughout the motion',
            'shooting_arm_kinematics': 'The movement pattern and joint angles of the shooting arm',
            'lower_body_stability': 'Hip and leg stability patterns during the shooting motion',
            'phase_timing_patterns': 'How similar the timing of different shooting phases are',
            'body_alignment': 'Posture and body positioning throughout the shot'
        }
    
    def extend_existing_interpretation(self, existing_interpretation: Dict,
                                     comparison_results: Dict,
                                     video1_data: Dict, video2_data: Dict,
                                     selected_hand: str) -> Dict:
        """
        Extend existing interpretation results with DTW analysis.
        
        Args:
            existing_interpretation: Results from existing AnalysisInterpreter
            comparison_results: Original comparison results
            video1_data: First video's normalized data
            video2_data: Second video's normalized data
            selected_hand: Selected shooting hand
            
        Returns:
            Enhanced interpretation with DTW insights added
        """
        print("🔄 Extending interpretation with DTW analysis...")
        
        # Perform DTW analysis
        dtw_results = self.dtw_analyzer.analyze_shooting_similarity(
            video1_data, video2_data, selected_hand
        )
        
        if 'error' in dtw_results:
            print(f"⚠️ DTW analysis failed: {dtw_results['error']}")
            # Return existing interpretation with error note
            extended_interpretation = existing_interpretation.copy()
            extended_interpretation['dtw_analysis_error'] = dtw_results['error']
            return extended_interpretation
        
        # Create extended interpretation by adding DTW content
        extended_interpretation = existing_interpretation.copy()
        
        # Add DTW-specific sections
        extended_interpretation['dtw_motion_analysis'] = self._interpret_dtw_results(dtw_results)
        extended_interpretation['motion_similarity_insights'] = self._create_motion_insights(dtw_results)
        extended_interpretation['temporal_alignment_analysis'] = self._analyze_temporal_patterns(dtw_results)
        
        # Store the original DTW analysis results for direct access
        extended_interpretation['dtw_analysis'] = dtw_results.get('dtw_analysis', {})
        
        # Enhance existing sections with DTW context
        extended_interpretation = self._enhance_existing_sections(extended_interpretation, dtw_results)
        
        # Preserve and extend existing key insights
        existing_insights = existing_interpretation.get('key_insights', [])
        dtw_insights = self._generate_dtw_insights(dtw_results)
        
        # Combine existing and DTW insights
        combined_insights = existing_insights + dtw_insights
        extended_interpretation['key_insights'] = combined_insights
        
        print("✅ DTW interpretation extension completed")
        print(f"   📊 Preserved {len(existing_insights)} existing insights")
        print(f"   🎯 Added {len(dtw_insights)} DTW insights")
        
        return extended_interpretation
    
    def _interpret_dtw_results(self, dtw_results: Dict) -> Dict:
        """Interpret DTW analysis results into human-readable format"""
        dtw_analysis = dtw_results.get('dtw_analysis', {})
        
        # Extract key metrics
        overall_similarity = dtw_analysis.get('overall_similarity', 0.0)
        feature_similarities = dtw_analysis.get('feature_similarities', {})
        phase_similarities = dtw_analysis.get('phase_similarities', {})
        grade = dtw_analysis.get('grade', 'F')
        
        # Create interpretation
        interpretation = {
            'overall_assessment': self._get_similarity_interpretation(overall_similarity),
            'similarity_level': self._get_similarity_level(overall_similarity),
            'grade': grade,
            'feature_analysis': self._interpret_feature_similarities(feature_similarities, dtw_analysis),
            'phase_analysis': self._interpret_phase_similarities(phase_similarities),
            'recommendations': self._generate_recommendations(feature_similarities, phase_similarities, overall_similarity)
        }
        
        return interpretation
    
    def _interpret_feature_similarities(self, feature_similarities: Dict, dtw_analysis: Dict) -> Dict:
        """Interpret individual feature similarities"""
        interpretations = {}
        detailed_analysis = dtw_analysis.get('detailed_analysis', {})
        
        for feature, similarity in feature_similarities.items():
            if feature in self.feature_names:
                feature_info = {
                    'name': self.feature_names[feature],
                    'description': self.feature_descriptions[feature],
                    'similarity_score': float(similarity),
                    'similarity_level': self._get_similarity_level(similarity),
                    'analysis': self._generate_feature_analysis(feature, similarity, detailed_analysis.get(feature, {}))
                }
                
                interpretations[feature] = feature_info
        
        return interpretations
    
    def _generate_feature_analysis(self, feature: str, similarity: float, detailed_data: Dict) -> str:
        """Generate detailed analysis for specific feature"""
        analysis_parts = []
        
        # Overall assessment
        level = self._get_similarity_level(similarity)
        analysis_parts.append(f"{level} similarity level at {similarity:.1f}%")
        
        # Feature-specific insights
        if feature == 'ball_wrist_trajectory':
            analysis_parts.extend(self._analyze_ball_wrist_feature(detailed_data, similarity))
        elif feature == 'shooting_arm_kinematics':
            analysis_parts.extend(self._analyze_arm_kinematics_feature(detailed_data, similarity))
        elif feature == 'lower_body_stability':
            analysis_parts.extend(self._analyze_lower_body_feature(detailed_data, similarity))
        elif feature == 'phase_timing_patterns':
            analysis_parts.extend(self._analyze_timing_feature(detailed_data, similarity))
        elif feature == 'body_alignment':
            analysis_parts.extend(self._analyze_alignment_feature(detailed_data, similarity))
        
        return '. '.join(analysis_parts)
    
    def _analyze_ball_wrist_feature(self, detailed_data: Dict, similarity: float) -> List[str]:
        """Analyze ball-wrist trajectory feature"""
        insights = []
        
        subfeatures = detailed_data.get('subfeature_similarities', {})
        
        if 'ball_trajectory' in subfeatures:
            ball_sim = subfeatures['ball_trajectory']
            if ball_sim >= 80:
                insights.append("Ball trajectory shows consistent path")
            elif ball_sim >= 60:
                insights.append("Ball trajectory shows moderate consistency with some variation")
            else:
                insights.append("Ball trajectory shows significant differences")
        
        if 'wrist_trajectory' in subfeatures:
            wrist_sim = subfeatures['wrist_trajectory']
            if wrist_sim >= 80:
                insights.append("Wrist movement pattern is highly consistent")
            elif wrist_sim >= 60:
                insights.append("Wrist movement shows good consistency")
            else:
                insights.append("Wrist movement patterns differ notably")
        
        if 'ball_wrist_distance' in subfeatures:
            distance_sim = subfeatures['ball_wrist_distance']
            if distance_sim >= 80:
                insights.append("Ball-hand coordination remains consistent throughout")
            elif distance_sim >= 60:
                insights.append("Ball-hand coordination shows moderate consistency")
            else:
                insights.append("Ball-hand coordination varies significantly")
        
        return insights
    
    def _analyze_arm_kinematics_feature(self, detailed_data: Dict, similarity: float) -> List[str]:
        """Analyze shooting arm kinematics feature"""
        insights = []
        
        subfeatures = detailed_data.get('subfeature_similarities', {})
        
        if 'elbow_angles' in subfeatures:
            elbow_sim = subfeatures['elbow_angles']
            if elbow_sim >= 80:
                insights.append("Elbow angle progression is highly consistent")
            elif elbow_sim >= 60:
                insights.append("Elbow angles show good consistency with minor variations")
            else:
                insights.append("Elbow angle patterns show notable differences")
        
        # Check temporal consistency
        dtw_data = detailed_data.get('dtw_analysis', {})
        warping_ratios = []
        for sub_analysis in dtw_data.values():
            if isinstance(sub_analysis, dict) and 'warping_ratio' in sub_analysis:
                warping_ratios.append(sub_analysis['warping_ratio'])
        
        if warping_ratios:
            avg_warping = np.mean(warping_ratios)
            if avg_warping > 1.3:
                insights.append("Arm motion timing is slower in first shooter")
            elif avg_warping < 0.7:
                insights.append("Arm motion timing is faster in first shooter")
            else:
                insights.append("Arm motion timing is well synchronized")
        
        return insights
    
    def _analyze_lower_body_feature(self, detailed_data: Dict, similarity: float) -> List[str]:
        """Analyze lower body stability feature"""
        insights = []
        
        subfeatures = detailed_data.get('subfeature_similarities', {})
        
        if 'hip_trajectory' in subfeatures:
            hip_sim = subfeatures['hip_trajectory']
            if hip_sim >= 80:
                insights.append("Hip movement shows excellent stability")
            elif hip_sim >= 60:
                insights.append("Hip movement demonstrates good stability")
            else:
                insights.append("Hip movement patterns differ significantly")
        
        if 'left_knee_angles' in subfeatures and 'right_knee_angles' in subfeatures:
            knee_sim_avg = (subfeatures['left_knee_angles'] + subfeatures['right_knee_angles']) / 2
            if knee_sim_avg >= 80:
                insights.append("Knee bend patterns are highly consistent")
            elif knee_sim_avg >= 60:
                insights.append("Knee movements show reasonable consistency")
            else:
                insights.append("Knee bend patterns show considerable variation")
        
        return insights
    
    def _analyze_timing_feature(self, detailed_data: Dict, similarity: float) -> List[str]:
        """Analyze timing patterns feature"""
        insights = []
        
        if similarity >= 80:
            insights.append("Phase timing shows excellent consistency")
        elif similarity >= 65:
            insights.append("Phase timing demonstrates good rhythm matching")
        else:
            insights.append("Phase timing shows significant rhythm differences")
        
        # Add specific timing insights if available
        subfeatures = detailed_data.get('subfeature_similarities', {})
        if 'phase_durations' in subfeatures:
            duration_sim = subfeatures['phase_durations']
            if duration_sim < 60:
                insights.append("Phase duration proportions vary notably between shooters")
        
        return insights
    
    def _analyze_alignment_feature(self, detailed_data: Dict, similarity: float) -> List[str]:
        """Analyze body alignment feature"""
        insights = []
        
        subfeatures = detailed_data.get('subfeature_similarities', {})
        
        if 'shoulder_tilt' in subfeatures:
            shoulder_sim = subfeatures['shoulder_tilt']
            if shoulder_sim >= 80:
                insights.append("Shoulder alignment is highly consistent")
            elif shoulder_sim >= 60:
                insights.append("Shoulder posture shows good consistency")
            else:
                insights.append("Shoulder alignment patterns differ")
        
        if 'torso_angle' in subfeatures:
            torso_sim = subfeatures['torso_angle']
            if torso_sim >= 80:
                insights.append("Torso posture remains consistent")
            elif torso_sim >= 60:
                insights.append("Torso alignment shows moderate consistency")
            else:
                insights.append("Torso positioning varies significantly")
        
        return insights
    
    def _interpret_phase_similarities(self, phase_similarities: Dict) -> Dict:
        """Interpret phase-specific similarity results"""
        phase_analysis = {}
        
        for phase, phase_data in phase_similarities.items():
            if isinstance(phase_data, dict) and 'similarity' in phase_data:
                similarity = phase_data['similarity']
                frame_count_1 = phase_data.get('frame_count_1', 0)
                frame_count_2 = phase_data.get('frame_count_2', 0)
                note = phase_data.get('note', '')
                
                phase_analysis[phase] = {
                    'similarity': similarity,
                    'interpretation': self._get_phase_interpretation(phase, similarity),
                    'frame_count_1': frame_count_1,
                    'frame_count_2': frame_count_2,
                    'note': note,
                    'quality': self._assess_phase_quality(similarity)
                }
        
        return phase_analysis
    
    def _get_phase_interpretation(self, phase: str, similarity: float) -> str:
        """Get human-readable interpretation for a specific phase"""
        if similarity >= 85:
            return f"Excellent {phase} phase consistency"
        elif similarity >= 75:
            return f"Good {phase} phase consistency"
        elif similarity >= 65:
            return f"Moderate {phase} phase consistency"
        elif similarity >= 50:
            return f"Fair {phase} phase consistency"
        else:
            return f"Needs improvement in {phase} phase"
    
    def _assess_phase_quality(self, similarity: float) -> str:
        """Assess the quality of a phase"""
        if similarity >= 85:
            return "excellent"
        elif similarity >= 75:
            return "good"
        elif similarity >= 65:
            return "moderate"
        elif similarity >= 50:
            return "fair"
        else:
            return "needs_improvement"
    
    def _generate_recommendations(self, feature_similarities: Dict, phase_similarities: Dict, 
                                overall_similarity: float) -> List[Dict]:
        """Generate specific recommendations based on analysis results"""
        recommendations = []
        
        # Feature-based recommendations
        for feature_name, similarity in feature_similarities.items():
            if similarity < 70:
                recommendation = self._generate_feature_recommendation(feature_name, similarity)
                if recommendation:
                    recommendations.append(recommendation)
        
        # Phase-based recommendations
        for phase, phase_data in phase_similarities.items():
            if isinstance(phase_data, dict) and 'similarity' in phase_data:
                similarity = phase_data['similarity']
                if similarity < 70:
                    recommendation = self._generate_phase_recommendation(phase, similarity)
                    if recommendation:
                        recommendations.append(recommendation)
        
        # Overall recommendations
        if overall_similarity < 75:
            recommendations.append({
                'type': 'overall',
                'issue': 'Overall shooting form needs improvement',
                'suggestion': 'Focus on consistent form across all phases',
                'importance': 'high'
            })
        
        return recommendations
    
    def _generate_feature_recommendation(self, feature_name: str, similarity: float) -> Dict:
        """Generate recommendation for a specific feature"""
        feature_descriptions = {
            'ball_wrist_trajectory': 'ball-wrist coordination',
            'shooting_arm_kinematics': 'shooting arm mechanics',
            'lower_body_stability': 'lower body stability',
            'phase_timing_patterns': 'timing consistency',
            'body_alignment': 'overall body alignment'
        }
        
        feature_desc = feature_descriptions.get(feature_name, feature_name)
        
        if similarity < 60:
            return {
                'type': 'feature',
                'feature': feature_name,
                'issue': f'Poor {feature_desc}',
                'suggestion': f'Focus on improving {feature_desc}',
                'importance': 'high'
            }
        elif similarity < 70:
            return {
                'type': 'feature',
                'feature': feature_name,
                'issue': f'Fair {feature_desc}',
                'suggestion': f'Work on {feature_desc} consistency',
                'importance': 'medium'
            }
        
        return None
    
    def _generate_phase_recommendation(self, phase: str, similarity: float) -> Dict:
        """Generate recommendation for a specific phase"""
        phase_descriptions = {
            'Setup': 'initial stance and preparation',
            'Loading': 'power generation and loading',
            'Rising': 'motion initiation and upward movement',
            'Release': 'ball release mechanics',
            'Follow-through': 'finishing motion and consistency'
        }
        
        phase_desc = phase_descriptions.get(phase, phase)
        
        if similarity < 60:
            return {
                'type': 'phase',
                'phase': phase,
                'issue': f'Poor {phase_desc}',
                'suggestion': f'Focus on improving {phase_desc}',
                'importance': 'high'
            }
        elif similarity < 70:
            return {
                'type': 'phase',
                'phase': phase,
                'issue': f'Fair {phase_desc}',
                'suggestion': f'Work on {phase_desc} consistency',
                'importance': 'medium'
            }
        
        return None
    
    def _create_motion_insights(self, dtw_results: Dict) -> Dict:
        """Create motion-specific insights from DTW analysis"""
        dtw_analysis = dtw_results.get('dtw_analysis', {})
        
        insights = {
            'coordination_patterns': self._analyze_coordination_patterns(dtw_analysis),
            'consistency_evaluation': self._evaluate_consistency(dtw_analysis),
            'timing_synchronization': self._analyze_timing_sync(dtw_analysis),
            'technique_similarities': self._identify_technique_similarities(dtw_analysis)
        }
        
        return insights
    
    def _analyze_coordination_patterns(self, dtw_analysis: Dict) -> Dict:
        """Analyze coordination patterns from DTW results"""
        detailed_analysis = dtw_analysis.get('detailed_analysis', {})
        coordination = {}
        
        # Ball-wrist coordination
        if 'ball_wrist_trajectory' in detailed_analysis:
            ball_wrist = detailed_analysis['ball_wrist_trajectory']
            coordination['ball_wrist_coordination'] = {
                'quality': self._assess_coordination_quality(ball_wrist.get('overall_similarity', 0)),
                'consistency_note': self._generate_coordination_note(ball_wrist)
            }
        
        # Multi-limb coordination
        feature_similarities = dtw_analysis.get('feature_similarities', {})
        arm_sim = feature_similarities.get('shooting_arm_kinematics', 0)
        body_sim = feature_similarities.get('lower_body_stability', 0)
        
        coordination['overall_body_coordination'] = {
            'upper_lower_sync': (arm_sim + body_sim) / 2 if arm_sim > 0 and body_sim > 0 else 0,
            'sync_quality': 'Excellent' if (arm_sim + body_sim) / 2 >= 80 else ('Good' if (arm_sim + body_sim) / 2 >= 65 else 'Needs Improvement')
        }
        
        return coordination
    
    def _evaluate_consistency(self, dtw_analysis: Dict) -> Dict:
        """Evaluate motion consistency"""
        consistency_metrics = dtw_analysis.get('motion_consistency_metrics', {})
        
        evaluation = {}
        
        # Overall consistency
        similarity_std = consistency_metrics.get('similarity_std', 0)
        if similarity_std < 10:
            evaluation['overall_consistency'] = 'High - motion patterns are very consistent across all features'
        elif similarity_std < 20:
            evaluation['overall_consistency'] = 'Moderate - some variation in motion patterns'
        else:
            evaluation['overall_consistency'] = 'Low - significant variation across different aspects of motion'
        
        # Temporal consistency
        temporal_consistency = consistency_metrics.get('temporal_consistency', 1.0)
        if 0.9 <= temporal_consistency <= 1.1:
            evaluation['temporal_consistency'] = 'Excellent timing alignment'
        elif 0.8 <= temporal_consistency <= 1.2:
            evaluation['temporal_consistency'] = 'Good timing alignment with minor variations'
        else:
            evaluation['temporal_consistency'] = 'Notable timing differences between motions'
        
        # Most/least consistent features
        most_consistent = consistency_metrics.get('most_consistent_feature', {})
        least_consistent = consistency_metrics.get('least_consistent_feature', {})
        
        if most_consistent.get('feature'):
            feature_name = self.feature_names.get(most_consistent['feature'], most_consistent['feature'])
            evaluation['strongest_aspect'] = f"{feature_name} shows the highest consistency"
        
        if least_consistent.get('feature'):
            feature_name = self.feature_names.get(least_consistent['feature'], least_consistent['feature'])
            evaluation['area_for_improvement'] = f"{feature_name} shows the most variation"
        
        return evaluation
    
    def _analyze_timing_sync(self, dtw_analysis: Dict) -> Dict:
        """Analyze timing synchronization"""
        temporal_analysis = dtw_analysis.get('temporal_alignment_analysis', {})
        
        sync_analysis = {}
        
        # Overall temporal pattern
        temporal_pattern = temporal_analysis.get('temporal_pattern', 'aligned')
        avg_warping = temporal_analysis.get('average_warping_ratio', 1.0)
        
        if temporal_pattern == 'aligned':
            sync_analysis['overall_timing'] = f"Motions are well synchronized (warping: {avg_warping:.2f}x)"
        elif temporal_pattern == 'stretched':
            sync_analysis['overall_timing'] = f"First motion is slower overall (warping: {avg_warping:.2f}x)"
        else:  # compressed
            sync_analysis['overall_timing'] = f"First motion is faster overall (warping: {avg_warping:.2f}x)"
        
        # Phase-specific timing
        feature_patterns = temporal_analysis.get('feature_temporal_patterns', {})
        timing_details = []
        
        for feature, pattern_data in feature_patterns.items():
            feature_name = self.feature_names.get(feature, feature)
            pattern = pattern_data.get('pattern', 'aligned')
            if pattern != 'aligned':
                timing_details.append(f"{feature_name} timing is {pattern}")
        
        if timing_details:
            sync_analysis['specific_timing_notes'] = timing_details
        else:
            sync_analysis['specific_timing_notes'] = ['All motion aspects show good temporal alignment']
        
        return sync_analysis
    
    def _identify_technique_similarities(self, dtw_analysis: Dict) -> Dict:
        """Identify technique similarities"""
        feature_similarities = dtw_analysis.get('feature_similarities', {})
        
        similarities = {
            'technique_strengths': [],
            'technique_differences': [],
            'overall_technique_similarity': 0.0
        }
        
        # Identify strengths (high similarity features)
        for feature, similarity in feature_similarities.items():
            feature_name = self.feature_names.get(feature, feature)
            if similarity >= 80:
                similarities['technique_strengths'].append(f"{feature_name} ({similarity:.1f}%)")
            elif similarity <= 60:
                similarities['technique_differences'].append(f"{feature_name} ({similarity:.1f}%)")
        
        # Overall technique assessment
        overall_sim = dtw_analysis.get('overall_similarity', 0)
        similarities['overall_technique_similarity'] = float(overall_sim)
        
        if overall_sim >= 85:
            similarities['technique_assessment'] = 'Very similar shooting techniques'
        elif overall_sim >= 70:
            similarities['technique_assessment'] = 'Similar shooting techniques with some variations'
        elif overall_sim >= 55:
            similarities['technique_assessment'] = 'Moderately different shooting techniques'
        else:
            similarities['technique_assessment'] = 'Distinctly different shooting techniques'
        
        return similarities
    
    def _analyze_temporal_patterns(self, dtw_results: Dict) -> Dict:
        """Analyze temporal alignment patterns"""
        dtw_analysis = dtw_results.get('dtw_analysis', {})
        temporal_analysis = dtw_analysis.get('temporal_alignment_analysis', {})
        
        return {
            'warping_analysis': {
                'average_warping_ratio': temporal_analysis.get('average_warping_ratio', 1.0),
                'warping_consistency': temporal_analysis.get('warping_consistency', 100.0),
                'temporal_pattern': temporal_analysis.get('temporal_pattern', 'aligned')
            },
            'phase_timing_comparison': self._compare_phase_timing(dtw_analysis),
            'motion_rhythm_analysis': self._analyze_motion_rhythm(temporal_analysis)
        }
    
    def _compare_phase_timing(self, dtw_analysis: Dict) -> Dict:
        """Compare phase timing between motions"""
        metadata = dtw_analysis.get('metadata', {})
        phase_dist_1 = metadata.get('phase_distribution_1', {})
        phase_dist_2 = metadata.get('phase_distribution_2', {})
        
        comparison = {}
        
        phases = ['Setup', 'Loading', 'Rising', 'Release', 'Follow-through']
        for phase in phases:
            frames_1 = phase_dist_1.get(phase, 0)
            frames_2 = phase_dist_2.get(phase, 0)
            
            if frames_1 > 0 and frames_2 > 0:
                ratio = frames_1 / frames_2 if frames_2 > 0 else 1.0
                if ratio > 1.3:
                    comparison[phase] = f"First motion spends more time in {phase} phase"
                elif ratio < 0.7:
                    comparison[phase] = f"Second motion spends more time in {phase} phase"
                else:
                    comparison[phase] = f"{phase} phase timing is similar"
            elif frames_1 > 0 or frames_2 > 0:
                comparison[phase] = f"{phase} phase present in only one motion"
        
        return comparison
    
    def _analyze_motion_rhythm(self, temporal_analysis: Dict) -> Dict:
        """Analyze overall motion rhythm"""
        avg_warping = temporal_analysis.get('average_warping_ratio', 1.0)
        warping_consistency = temporal_analysis.get('warping_consistency', 100.0)
        
        rhythm_analysis = {}
        
        # Overall rhythm assessment
        if warping_consistency >= 80:
            rhythm_analysis['rhythm_consistency'] = 'Consistent rhythm throughout motion'
        elif warping_consistency >= 60:
            rhythm_analysis['rhythm_consistency'] = 'Generally consistent rhythm with some variation'
        else:
            rhythm_analysis['rhythm_consistency'] = 'Inconsistent rhythm patterns'
        
        # Tempo comparison
        if 0.95 <= avg_warping <= 1.05:
            rhythm_analysis['tempo_comparison'] = 'Nearly identical tempo'
        elif avg_warping > 1.2:
            rhythm_analysis['tempo_comparison'] = 'First motion notably slower'
        elif avg_warping < 0.8:
            rhythm_analysis['tempo_comparison'] = 'First motion notably faster'
        else:
            rhythm_analysis['tempo_comparison'] = 'Slightly different tempo'
        
        return rhythm_analysis
    
    def _enhance_existing_sections(self, interpretation: Dict, dtw_results: Dict) -> Dict:
        """Enhance existing interpretation sections with DTW context"""
        dtw_analysis = dtw_results.get('dtw_analysis', {})
        
        # Add DTW context to phase analysis
        if 'text_analysis' in interpretation:
            for phase in interpretation['text_analysis']:
                phase_section = interpretation['text_analysis'][phase]
                dtw_context = self._get_phase_dtw_context(phase, dtw_analysis)
                if dtw_context:
                    phase_section['dtw_motion_context'] = dtw_context
        
        # Enhance phase transition analysis with DTW timing insights
        if 'phase_transition_analysis' in interpretation:
            timing_insights = self._extract_timing_insights(dtw_analysis)
            interpretation['phase_transition_analysis']['motion_timing_patterns'] = timing_insights
        
        return interpretation
    
    def _get_phase_dtw_context(self, phase: str, dtw_analysis: Dict) -> Optional[str]:
        """Get DTW context for specific phase"""
        # This is a simplified version - could be enhanced with phase-specific DTW analysis
        overall_similarity = dtw_analysis.get('overall_similarity', 0)
        
        if overall_similarity >= 80:
            return f"Motion patterns in {phase} phase show high consistency"
        elif overall_similarity >= 65:
            return f"Motion patterns in {phase} phase show good consistency"
        elif overall_similarity >= 50:
            return f"Motion patterns in {phase} phase show moderate consistency"
        else:
            return f"Motion patterns in {phase} phase show notable differences"
    
    def _extract_timing_insights(self, dtw_analysis: Dict) -> Dict:
        """Extract timing insights from DTW analysis"""
        temporal_analysis = dtw_analysis.get('temporal_alignment_analysis', {})
        
        return {
            'overall_timing_consistency': temporal_analysis.get('warping_consistency', 100.0),
            'tempo_variation': temporal_analysis.get('average_warping_ratio', 1.0),
            'timing_pattern': temporal_analysis.get('temporal_pattern', 'aligned')
        }
    
    def _generate_dtw_insights(self, dtw_results: Dict) -> List[str]:
        """Generate key insights from DTW analysis for integration with existing insights"""
        insights = []
        dtw_analysis = dtw_results.get('dtw_analysis', {})
        
        overall_similarity = dtw_analysis.get('overall_similarity', 0)
        feature_similarities = dtw_analysis.get('feature_similarities', {})
        grade = dtw_analysis.get('grade', 'N/A')
        
        # Overall similarity insight
        insights.append(f"Motion similarity analysis: {overall_similarity:.1f}% overall similarity (Grade: {grade})")
        
        # Best and worst features
        if feature_similarities:
            best_feature = max(feature_similarities.items(), key=lambda x: x[1])
            worst_feature = min(feature_similarities.items(), key=lambda x: x[1])
            
            best_name = self.feature_names.get(best_feature[0], best_feature[0])
            worst_name = self.feature_names.get(worst_feature[0], worst_feature[0])
            
            insights.append(f"Strongest similarity: {best_name} ({best_feature[1]:.1f}%)")
            insights.append(f"Greatest difference: {worst_name} ({worst_feature[1]:.1f}%)")
        
        # Timing insight
        temporal_analysis = dtw_analysis.get('temporal_alignment_analysis', {})
        temporal_pattern = temporal_analysis.get('temporal_pattern', 'aligned')
        if temporal_pattern != 'aligned':
            insights.append(f"Timing pattern: First motion is {temporal_pattern} relative to second")
        else:
            insights.append("Timing pattern: Well synchronized motion timing")
        
        # Consistency insight
        consistency_metrics = dtw_analysis.get('motion_consistency_metrics', {})
        similarity_std = consistency_metrics.get('similarity_std', 0)
        if similarity_std < 10:
            insights.append("Motion consistency: High consistency across all movement aspects")
        elif similarity_std < 20:
            insights.append("Motion consistency: Moderate variation between different movement aspects")
        else:
            insights.append("Motion consistency: Notable variation across different movement aspects")
        
        return insights
    
    # Helper methods
    def _get_similarity_interpretation(self, similarity: float) -> str:
        """Convert similarity score to interpretation"""
        if similarity >= 90:
            return "Nearly identical motion patterns"
        elif similarity >= 80:
            return "Very similar motion patterns with minor variations"
        elif similarity >= 70:
            return "Similar motion patterns with some notable differences"
        elif similarity >= 60:
            return "Moderately similar patterns with several differences"
        else:
            return "Significantly different motion patterns"
    
    def _get_similarity_level(self, similarity: float) -> str:
        """Get similarity level description"""
        if similarity >= 85:
            return "Excellent"
        elif similarity >= 75:
            return "Very Good"
        elif similarity >= 65:
            return "Good"
        elif similarity >= 55:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _assess_coordination_quality(self, similarity: float) -> str:
        """Assess coordination quality"""
        if similarity >= 85:
            return "Excellent coordination"
        elif similarity >= 75:
            return "Good coordination"
        elif similarity >= 65:
            return "Moderate coordination"
        else:
            return "Poor coordination"
    
    def _generate_coordination_note(self, ball_wrist_analysis: Dict) -> str:
        """Generate coordination note from ball-wrist analysis"""
        subfeatures = ball_wrist_analysis.get('subfeature_similarities', {})
        
        if not subfeatures:
            return "Coordination analysis not available"
        
        avg_similarity = np.mean(list(subfeatures.values()))
        
        if avg_similarity >= 80:
            return "Ball and wrist move in excellent coordination throughout the motion"
        elif avg_similarity >= 65:
            return "Ball and wrist show good coordination with minor inconsistencies"
        else:
            return "Ball and wrist coordination shows room for improvement"
    
    def _evaluate_motion_consistency(self, dtw_analysis: Dict) -> Dict:
        """Evaluate overall motion consistency"""
        consistency_metrics = dtw_analysis.get('motion_consistency_metrics', {})
        
        return {
            'consistency_score': consistency_metrics.get('similarity_consistency', 0),
            'temporal_consistency': consistency_metrics.get('temporal_consistency', 1.0),
            'variability_assessment': 'Low variability' if consistency_metrics.get('similarity_std', 20) < 10 else 'High variability'
        }
    
    def _analyze_motion_patterns(self, detailed_analyses: Dict) -> Dict:
        """Analyze motion patterns from detailed DTW analyses"""
        patterns = {
            'dominant_patterns': [],
            'inconsistent_patterns': [],
            'overall_pattern_quality': 'Good'
        }
        
        # Analyze each feature's patterns
        for feature, analysis in detailed_analyses.items():
            if 'error' in analysis:
                continue
                
            similarity = analysis.get('overall_similarity', 0)
            feature_name = self.feature_names.get(feature, feature)
            
            if similarity >= 80:
                patterns['dominant_patterns'].append(f"{feature_name} shows consistent patterns")
            elif similarity <= 60:
                patterns['inconsistent_patterns'].append(f"{feature_name} shows variable patterns")
        
        # Overall assessment
        avg_similarities = []
        for analysis in detailed_analyses.values():
            if 'error' not in analysis:
                avg_similarities.append(analysis.get('overall_similarity', 0))
        
        if avg_similarities:
            overall_avg = np.mean(avg_similarities)
            if overall_avg >= 80:
                patterns['overall_pattern_quality'] = 'Excellent'
            elif overall_avg >= 65:
                patterns['overall_pattern_quality'] = 'Good'
            else:
                patterns['overall_pattern_quality'] = 'Needs Improvement'
        
        return patterns