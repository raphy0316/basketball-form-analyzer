"""
DTW Analyzer

Main DTW analysis coordinator that integrates with existing comparison system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .dtw_feature_extractor import DTWFeatureExtractor
    from .dtw_similarity_calculator import DTWSimilarityCalculator
    from .dtw_config import DTW_FEATURE_WEIGHTS, PHASE_IMPORTANCE_WEIGHTS, SIMILARITY_GRADES, CONFIDENCE_THRESHOLDS
    from .setup_posture_scorer import SetupPostureScorer
    from .release_posture_scorer import ReleasePostureScorer
    from .loading_dtw_feature_extractor import LoadingDTWFeatureExtractor
except ImportError:
    from dtw_feature_extractor import DTWFeatureExtractor
    from dtw_similarity_calculator import DTWSimilarityCalculator
    from dtw_config import DTW_FEATURE_WEIGHTS, PHASE_IMPORTANCE_WEIGHTS, SIMILARITY_GRADES, CONFIDENCE_THRESHOLDS
    from setup_posture_scorer import SetupPostureScorer
    from release_posture_scorer import ReleasePostureScorer
    from loading_dtw_feature_extractor import LoadingDTWFeatureExtractor
from typing import Dict, Tuple
import numpy as np


class DTWAnalyzer:
    """
    Main DTW analysis coordinator.
    
    Integrates DTW analysis with existing shooting comparison pipeline.
    """
    
    def __init__(self):
        self.feature_extractor = DTWFeatureExtractor()
        self.similarity_calculator = DTWSimilarityCalculator()
        self.setup_scorer = SetupPostureScorer()
        self.release_scorer = ReleasePostureScorer()
        self.loading_dtw_extractor = LoadingDTWFeatureExtractor()
        self.feature_weights = DTW_FEATURE_WEIGHTS.copy()
        self.phase_weights = PHASE_IMPORTANCE_WEIGHTS.copy()
        self.similarity_grades = SIMILARITY_GRADES.copy()
        self.confidence_thresholds = CONFIDENCE_THRESHOLDS.copy()
    
    def analyze_shooting_similarity(self, video1_data: Dict, video2_data: Dict, 
                                  selected_hand: str, setup_analysis: Dict = None, 
                                  release_analysis: Dict = None, loading_analysis: Dict = None,
                                  followthrough_analysis: Dict = None, rising_analysis: Dict = None) -> Dict:
        """
        Perform complete DTW analysis between two shooting motions.
        
        Args:
            video1_data: First video's normalized data
            video2_data: Second video's normalized data
            selected_hand: Selected shooting hand ('left' or 'right')
            
        Returns:
            Comprehensive DTW analysis results
        """
        print(f"Starting DTW analysis for {selected_hand} hand shooting...")
        
        # Extract DTW features from both videos
        print("Extracting DTW features...")
        features1 = self.feature_extractor.extract_dtw_features(video1_data, selected_hand)
        features2 = self.feature_extractor.extract_dtw_features(video2_data, selected_hand)
        
        if 'error' in features1:
            return {'error': f'Video 1 feature extraction failed: {features1["error"]}'}
        
        if 'error' in features2:
            return {'error': f'Video 2 feature extraction failed: {features2["error"]}'}
        
        # Calculate global motion similarity (대체된 feature similarity 계산)
        print("Calculating global motion similarity...")
        
        # 글로벌 모션 유사도를 위한 데이터 준비
        user_motion_data = self._prepare_global_motion_data(video1_data, features1)
        ref_motion_data = self._prepare_global_motion_data(video2_data, features2)
        
        # 글로벌 점수 계산 
        global_result = self.similarity_calculator.compute_global_score(user_motion_data, ref_motion_data)
        
        # 기존 구조와의 호환성을 위해 결과 매핑
        global_similarity = global_result.get('global_score', 0.0)
        
        # 기존 feature_similarities 구조 유지 (하위호환성)
        feature_similarities = {
            'global_motion_similarity': global_similarity
        }
        detailed_analyses = {
            'global_motion_analysis': global_result
        }
        
        # Calculate phase-specific similarities
        print("Calculating phase-specific similarities...")
        phase_similarities = self._calculate_phase_specific_similarities(
            features1, features2, video1_data, video2_data, setup_analysis, release_analysis, loading_analysis, followthrough_analysis, rising_analysis
        )
        
        # Calculate overall similarity score (feature-based + phase-based)
        overall_similarity = self._calculate_overall_similarity(feature_similarities, phase_similarities)
        
        # Generate grade and confidence
        grade = self._calculate_grade(overall_similarity)
        confidence = self._assess_analysis_confidence(detailed_analyses, features1, features2)
        
        # Generate comprehensive analysis
        analysis_results = {
            'dtw_analysis': {
                'overall_similarity': float(overall_similarity),
                'grade': grade,
                'feature_similarities': {k: float(v) for k, v in feature_similarities.items()},
                'phase_similarities': phase_similarities,
                'detailed_analysis': detailed_analyses,
                'motion_consistency_metrics': self._calculate_consistency_metrics(detailed_analyses),
                'temporal_alignment_analysis': self._analyze_temporal_alignment(detailed_analyses),
                'metadata': {
                    'video1_frames': features1['metadata']['total_frames'],
                    'video2_frames': features2['metadata']['total_frames'],
                    'selected_hand': selected_hand,
                    'analysis_confidence': confidence,
                    'phase_distribution_1': features1['metadata']['phase_distribution'],
                    'phase_distribution_2': features2['metadata']['phase_distribution'],
                    'extraction_success': True
                }
            }
        }
        
        print(f"DTW analysis completed:")
        print(f"   Overall similarity: {overall_similarity:.1f}% (Grade: {grade})")
        print(f"   Analysis confidence: {confidence}")
        
        return analysis_results
    
    def _calculate_phase_specific_similarities(self, features1: Dict, features2: Dict,
                                             video1_data: Dict, video2_data: Dict, setup_analysis: Dict = None, 
                                             release_analysis: Dict = None, loading_analysis: Dict = None,
                                             followthrough_analysis: Dict = None, rising_analysis: Dict = None) -> Dict:
        """
        Calculate phase-specific DTW similarities.
        
        Args:
            features1: First video's DTW features
            features2: Second video's DTW features
            video1_data: First video's raw data
            video2_data: Second video's raw data
            
        Returns:
            Dictionary of phase-specific similarities
        """
        # Organize frames by phase
        phase_frames1 = self._organize_frames_by_phase(video1_data.get('frames', []))
        phase_frames2 = self._organize_frames_by_phase(video2_data.get('frames', []))
        
        # Debug: Print phase distribution
        print("   🔍 Phase distribution:")
        for phase in ['Setup', 'Loading', 'Rising', 'Release', 'Follow-through']:
            count1 = len(phase_frames1.get(phase, []))
            count2 = len(phase_frames2.get(phase, []))
            print(f"      • {phase}: Video1={count1}, Video2={count2}")
        
        # Calculate phase-specific similarities
        phase_similarities = {}
        phases = ['Setup', 'Loading', 'Rising', 'Release', 'Follow-through']
        
        for phase in phases:
            phase1_frames = phase_frames1.get(phase, [])
            phase2_frames = phase_frames2.get(phase, [])
            
            # Special handling for Setup phase - use posture comparison instead of DTW
            if phase == 'Setup':
                print(f"   🔸 Setup phase: Using posture comparison instead of DTW")
                if setup_analysis:
                    setup_similarity_result = self.setup_scorer.calculate_setup_similarity(
                        setup_analysis.get('video1', {}),
                        setup_analysis.get('video2', {})
                    )
                    
                    phase_similarities[phase] = {
                        'similarity': setup_similarity_result['similarity_score'],
                        'frame_count_1': len(phase1_frames),
                        'frame_count_2': len(phase2_frames),
                        'note': f'Setup posture similarity analysis',
                        'feature_scores': setup_similarity_result.get('feature_scores', {}),
                        'method': 'posture_comparison'
                    }
                    print(f"   ✅ Setup posture similarity: {setup_similarity_result['similarity_score']:.1f}%")
                else:
                    phase_similarities[phase] = {
                        'similarity': 0.0,
                        'frame_count_1': len(phase1_frames),
                        'frame_count_2': len(phase2_frames),
                        'note': 'Setup analysis data not available',
                        'method': 'posture_comparison'
                    }
                continue
            
            # Special handling for Release phase - use posture comparison instead of DTW
            if phase == 'Release':
                print(f"   🔸 Release phase: Using posture comparison instead of DTW")
                if release_analysis:
                    release_similarity_result = self.release_scorer.calculate_release_similarity(
                        release_analysis.get('video1', {}),
                        release_analysis.get('video2', {})
                    )
                    
                    phase_similarities[phase] = {
                        'similarity': release_similarity_result['similarity_score'],
                        'frame_count_1': len(phase1_frames),
                        'frame_count_2': len(phase2_frames),
                        'note': f'Release posture similarity analysis',
                        'feature_scores': release_similarity_result.get('feature_scores', {}),
                        'method': 'posture_comparison'
                    }
                    print(f"   ✅ Release posture similarity: {release_similarity_result['similarity_score']:.1f}%")
                else:
                    phase_similarities[phase] = {
                        'similarity': 0.0,
                        'frame_count_1': len(phase1_frames),
                        'frame_count_2': len(phase2_frames),
                        'note': 'Release analysis data not available',
                        'method': 'posture_comparison'
                    }
                continue
            
            # Special handling for Loading phase - use integrated DTW + static analysis
            if phase == 'Loading':
                print(f"   🔸 Loading phase: Using integrated DTW + static analysis")
                
                # Extract Loading-specific DTW features
                loading_features1 = self.loading_dtw_extractor.extract_loading_dtw_features(video1_data)
                loading_features2 = self.loading_dtw_extractor.extract_loading_dtw_features(video2_data)
                
                if 'error' in loading_features1 or 'error' in loading_features2:
                    print(f"   ⚠️ Loading DTW feature extraction failed")
                    dtw_loading_similarity = 0.0
                else:
                    # Calculate Loading DTW similarity
                    dtw_loading_similarity = self._calculate_loading_dtw_similarity(loading_features1, loading_features2)
                    print(f"   📊 Loading DTW similarity: {dtw_loading_similarity:.1f}%")
                
                # Integrate DTW with static analysis
                if loading_analysis:
                    loading_analysis1 = loading_analysis.get('video1', {})
                    loading_analysis2 = loading_analysis.get('video2', {})
                    
                    integrated_result = self.similarity_calculator.calculate_loading_integrated_similarity(
                        dtw_loading_similarity, loading_analysis1, loading_analysis2
                    )
                    
                    phase_similarities[phase] = {
                        'similarity': integrated_result['overall_similarity'],
                        'frame_count_1': len(phase1_frames),
                        'frame_count_2': len(phase2_frames),
                        'note': 'Integrated DTW + static analysis for Loading phase',
                        'method': 'integrated_dtw_static',
                        'dtw_similarity': integrated_result['dtw_similarity'],
                        'static_similarity': integrated_result.get('static_overall_similarity', 0.0),
                        'static_components': integrated_result.get('static_similarities', {}),
                        'weights': integrated_result.get('weights', {})
                    }
                    print(f"   ✅ Loading integrated similarity: {integrated_result['overall_similarity']:.1f}%")
                else:
                    # Fallback to DTW only
                    phase_similarities[phase] = {
                        'similarity': dtw_loading_similarity,
                        'frame_count_1': len(phase1_frames),
                        'frame_count_2': len(phase2_frames),
                        'note': 'DTW analysis only (static analysis not available)',
                        'method': 'loading_dtw_only'
                    }
                    print(f"   ✅ Loading DTW-only similarity: {dtw_loading_similarity:.1f}%")
                continue
            
            if not phase1_frames or not phase2_frames:
                # Special handling for Loading phase when no frames are detected - fallback
                if phase == 'Loading':
                    print(f"   🔍 Special handling for Loading phase with no frames")
                    # Give a default similarity for Loading phase when not detected
                    loading_similarity = 75.0  # Default similarity for missing Loading phase
                    phase_similarities[phase] = {
                        'similarity': loading_similarity,
                        'frame_count_1': len(phase1_frames),
                        'frame_count_2': len(phase2_frames),
                        'note': f'Loading phase not detected - assuming direct Setup→Rising transition (default similarity: {loading_similarity}%)'
                    }
                else:
                    phase_similarities[phase] = {
                        'similarity': 0.0,
                        'frame_count_1': len(phase1_frames),
                        'frame_count_2': len(phase2_frames),
                        'note': f'Phase not present in one or both motions'
                    }
                continue
            
            # Calculate phase-specific similarity
            # Special handling for Rising phase with integrated DTW + static analysis
            if phase == 'Rising' and rising_analysis:
                print(f"   🔸 {phase}: Using integrated DTW + static analysis")
                
                # First get DTW similarity for Rising phase
                phase_sim = self.similarity_calculator.calculate_phase_specific_similarity(
                    features1, features2, {phase: phase1_frames}, {phase: phase2_frames}
                )
                
                if phase in phase_sim:
                    dtw_rising_similarity = phase_sim[phase]['similarity']
                    print(f"   📊 Rising DTW similarity: {dtw_rising_similarity:.1f}%")
                    
                    # Integrate DTW with static analysis
                    rising_analysis1 = rising_analysis.get('video1', {})
                    rising_analysis2 = rising_analysis.get('video2', {})
                    
                    integrated_result = self.similarity_calculator.calculate_rising_integrated_similarity(
                        dtw_rising_similarity, rising_analysis1, rising_analysis2
                    )
                    
                    phase_similarities[phase] = {
                        'similarity': integrated_result['overall_similarity'],
                        'frame_count_1': len(phase1_frames),
                        'frame_count_2': len(phase2_frames),
                        'note': 'Integrated DTW + static analysis for Rising phase',
                        'method': 'integrated_dtw_static',
                        'dtw_similarity': integrated_result['dtw_similarity'],
                        'static_similarity': integrated_result.get('static_overall_similarity', 0.0),
                        'static_components': integrated_result.get('static_similarities', {}),
                        'weights': integrated_result.get('weights', {})
                    }
                    print(f"   ✅ Rising integrated similarity: {integrated_result['overall_similarity']:.1f}%")
                else:
                    # Fallback if DTW failed
                    phase_similarities[phase] = {
                        'similarity': 0.0,
                        'frame_count_1': len(phase1_frames),
                        'frame_count_2': len(phase2_frames),
                        'note': f'Rising DTW analysis failed'
                    }
            # Special handling for Follow-through phase with static comparison
            elif phase == 'Follow-through' and followthrough_analysis:
                followthrough1 = followthrough_analysis.get('video1', {})
                followthrough2 = followthrough_analysis.get('video2', {})
                
                phase_sim = self.similarity_calculator.calculate_phase_specific_similarity(
                    features1, features2, {phase: phase1_frames}, {phase: phase2_frames},
                    followthrough1, followthrough2
                )
                
                if phase in phase_sim:
                    phase_similarities[phase] = phase_sim[phase]
                else:
                    phase_similarities[phase] = {
                        'similarity': 0.0,
                        'frame_count_1': len(phase1_frames),
                        'frame_count_2': len(phase2_frames),
                        'note': f'Follow-through analysis failed'
                    }
                print(f"   🔸 {phase}: {phase_similarities[phase]['similarity']:.1f}%")
            else:
                # Standard DTW analysis for other phases
                phase_sim = self.similarity_calculator.calculate_phase_specific_similarity(
                    features1, features2, {phase: phase1_frames}, {phase: phase2_frames}
                )
                
                if phase in phase_sim:
                    phase_similarities[phase] = phase_sim[phase]
                else:
                    phase_similarities[phase] = {
                        'similarity': 0.0,
                        'frame_count_1': len(phase1_frames),
                        'frame_count_2': len(phase2_frames),
                        'note': f'Phase analysis failed'
                    }
                print(f"   🔸 {phase}: {phase_similarities[phase]['similarity']:.1f}%")
        
        return phase_similarities
    
    def _calculate_loading_dtw_similarity(self, loading_features1: Dict, loading_features2: Dict) -> float:
        """
        Calculate Loading phase DTW similarity using specialized features.
        
        Args:
            loading_features1: Loading DTW features from video 1
            loading_features2: Loading DTW features from video 2
            
        Returns:
            Loading phase similarity score (0-100)
        """
        print(f"      🔍 Calculating Loading DTW similarity...")
        
        feature_similarities = {}
        
        # A. Loading leg kinematics (40%)
        leg_kinematics1 = loading_features1.get('loading_leg_kinematics', {})
        leg_kinematics2 = loading_features2.get('loading_leg_kinematics', {})
        
        if leg_kinematics1 is not None and leg_kinematics2 is not None and len(leg_kinematics1) > 0 and len(leg_kinematics2) > 0:
            leg_sim = self.similarity_calculator.calculate_feature_similarity(
                leg_kinematics1, leg_kinematics2, 'loading_leg_kinematics'
            )
            feature_similarities['loading_leg_kinematics'] = leg_sim['overall_similarity']
            print(f"         • Leg kinematics: {leg_sim['overall_similarity']:.1f}%")
        
        # B. Loading upper body dynamics (35%)
        upper_body1 = loading_features1.get('loading_upper_body_dynamics', {})
        upper_body2 = loading_features2.get('loading_upper_body_dynamics', {})
        
        if upper_body1 is not None and upper_body2 is not None and len(upper_body1) > 0 and len(upper_body2) > 0:
            upper_body_sim = self.similarity_calculator.calculate_feature_similarity(
                upper_body1, upper_body2, 'loading_upper_body_dynamics'
            )
            feature_similarities['loading_upper_body_dynamics'] = upper_body_sim['overall_similarity']
            print(f"         • Upper body dynamics: {upper_body_sim['overall_similarity']:.1f}%")
        
        # C. Loading timing patterns (25%)
        timing1 = loading_features1.get('loading_timing_patterns', {})
        timing2 = loading_features2.get('loading_timing_patterns', {})
        
        if timing1 is not None and timing2 is not None and len(timing1) > 0 and len(timing2) > 0:
            timing_sim = self.similarity_calculator.calculate_feature_similarity(
                timing1, timing2, 'loading_timing_patterns'
            )
            feature_similarities['loading_timing_patterns'] = timing_sim['overall_similarity']
            print(f"         • Timing patterns: {timing_sim['overall_similarity']:.1f}%")
        
        # Calculate weighted overall similarity
        if not feature_similarities:
            return 0.0
        
        # Use loading-specific weights
        loading_weights = {
            'loading_leg_kinematics': 0.40,
            'loading_upper_body_dynamics': 0.35,
            'loading_timing_patterns': 0.25
        }
        
        weighted_similarity = 0.0
        total_weight = 0.0
        
        for feature_name, similarity in feature_similarities.items():
            weight = loading_weights.get(feature_name, 0.0)
            weighted_similarity += similarity * weight
            total_weight += weight
        
        overall_similarity = weighted_similarity / total_weight if total_weight > 0 else 0.0
        
        print(f"      🎯 Loading DTW overall similarity: {overall_similarity:.1f}%")
        
        return overall_similarity
    
    def _prepare_global_motion_data(self, video_data: Dict, features: Dict) -> Dict:
        """글로벌 모션 유사도 계산을 위한 데이터 준비"""
        try:
            frames = video_data.get('frames', [])
            
            # wrist_y와 com_y 추출
            wrist_y = []
            com_y = []
            
            for frame in frames:
                # wrist 좌표 추출 (shooting hand)
                keypoints = frame.get('keypoints', [])
                if keypoints and len(keypoints) > 10:  # 손목은 일반적으로 인덱스 9, 10
                    wrist_y.append(keypoints[10][1])  # 오른손 손목 Y 좌표
                else:
                    wrist_y.append(0.0)
                
                # center of mass Y 계산 (간단한 approximation)
                if keypoints and len(keypoints) > 0:
                    valid_points = [kp[1] for kp in keypoints if len(kp) > 1]
                    if valid_points:
                        com_y.append(sum(valid_points) / len(valid_points))
                    else:
                        com_y.append(0.0)
                else:
                    com_y.append(0.0)
            
            # Phase 정보 준비
            phases = {}
            phase_frames = self._organize_frames_by_phase(frames)
            
            for phase_name, phase_frame_list in phase_frames.items():
                if phase_frame_list:
                    start_idx = min([f.get('frame_index', 0) for f in phase_frame_list])
                    end_idx = max([f.get('frame_index', 0) for f in phase_frame_list])
                    duration = (end_idx - start_idx + 1) / 30.0  # 30fps 가정
                    
                    phases[phase_name.lower()] = {
                        'start': start_idx,
                        'end': end_idx,
                        'duration': duration
                    }
            
            return {
                'wrist_y': wrist_y,
                'com_y': com_y,
                'fps': 30,  # 기본값
                'phases': phases
            }
            
        except Exception as e:
            print(f"   Warning: Error preparing global motion data: {e}")
            return {
                'wrist_y': [],
                'com_y': [],
                'fps': 30,
                'phases': {}
            }
    
    def _organize_frames_by_phase(self, frames: list) -> Dict[str, list]:
        """Organize frames by shooting phase"""
        phase_frames = {
            'Setup': [],
            'Loading': [], 
            'Rising': [],
            'Release': [],
            'Follow-through': [],
            'General': []
        }
        
        # Phase name mapping to handle different naming conventions
        phase_mapping = {
            'setup': 'Setup',
            'Set-up': 'Setup',  # Add this mapping for "Set-up" → "Setup"
            'loading': 'Loading',
            'Loading-Rising': 'Loading-Rising',  # Keep as separate phase
            'rising': 'Rising',
            'release': 'Release',
            'follow_through': 'Follow-through',
            'follow-through': 'Follow-through',
            'followthrough': 'Follow-through'
        }
        
        for frame in frames:
            phase = frame.get('phase', 'General')
            
            # Normalize phase name
            if phase in phase_mapping:
                normalized_phase = phase_mapping[phase]
            elif phase in phase_frames:
                normalized_phase = phase
            else:
                normalized_phase = 'General'
            
            # Add frame to the appropriate phase(s)
            if normalized_phase == 'Loading-Rising':
                # Loading-Rising frames should be included in both Loading and Rising
                phase_frames['Loading'].append(frame)
                phase_frames['Rising'].append(frame)
            else:
                phase_frames[normalized_phase].append(frame)
        
        return phase_frames
    
    def _calculate_overall_similarity(self, feature_similarities: Dict, phase_similarities: Dict) -> float:
        """Calculate overall similarity from feature and phase similarities"""
        
        # 글로벌 모션 유사도 가져오기 (기존 feature similarity를 대체)
        global_motion_similarity = feature_similarities.get('global_motion_similarity', 0.0)
        
        # Calculate weighted phase similarity
        phase_similarity = 0.0
        total_phase_weight = 0.0
        
        for phase_name, phase_data in phase_similarities.items():
            if isinstance(phase_data, dict) and 'similarity' in phase_data:
                similarity = phase_data['similarity']
            else:
                similarity = phase_data if isinstance(phase_data, (int, float)) else 0.0
            
            weight = self.phase_weights.get(phase_name, 0.1)
            phase_similarity += similarity * weight
            total_phase_weight += weight
        
        if total_phase_weight > 0:
            phase_similarity /= total_phase_weight
        
        # Enhanced differentiation: Apply extreme non-linear scaling to maximize gap between similar and different motions
        def enhance_differentiation(similarity):
            if similarity >= 95:
                return similarity + 20  # Boost very similar motions extremely
            elif similarity >= 90:
                return similarity + 15  # Boost very similar motions significantly
            elif similarity >= 80:
                return similarity + 8   # Boost similar motions
            elif similarity >= 70:
                return similarity + 3   # Slight boost for moderately similar motions
            elif similarity >= 60:
                return similarity - 15  # Reduce medium similarity more
            elif similarity >= 50:
                return similarity - 30  # Significantly reduce low similarity
            elif similarity >= 40:
                return similarity - 40  # Heavily reduce very low similarity
            else:
                return max(10, similarity - 50)  # Drastically reduce extremely low similarity
        
        # Apply differentiation enhancement
        global_similarity_enhanced = enhance_differentiation(global_motion_similarity)
        phase_similarity_enhanced = enhance_differentiation(phase_similarity)
        
        # 글로벌 모션 점수와 페이즈 점수 결합 (글로벌 65%, 페이즈 35%)
        overall_similarity = (global_similarity_enhanced * 0.65) + (phase_similarity_enhanced * 0.35)
        
        # Ensure reasonable bounds
        overall_similarity = max(10.0, min(100.0, overall_similarity))
        
        print(f"   📊 Global Motion Similarity: {global_motion_similarity:.1f}")
        
        print(f"   📊 Phase similarities:")
        for phase_name, phase_data in phase_similarities.items():
            if isinstance(phase_data, dict) and 'similarity' in phase_data:
                similarity = phase_data['similarity']
            else:
                similarity = phase_data if isinstance(phase_data, (int, float)) else 0.0
            
            weight = self.phase_weights.get(phase_name, 0.1)
            print(f"      • {phase_name}: {similarity:.1f} (weight: {weight:.2f})")
        
        print(f"   🎯 Overall calculation:")
        print(f"      • Global motion: {global_motion_similarity:.1f} → Enhanced: {global_similarity_enhanced:.1f}")
        print(f"      • Phase similarity: {phase_similarity:.1f} → Enhanced: {phase_similarity_enhanced:.1f}")
        print(f"      • Final overall: {overall_similarity:.1f} (65% global + 35% phase)")
        
        return overall_similarity
    
    def _calculate_grade(self, similarity: float) -> str:
        """Convert similarity score to letter grade"""
        for grade, threshold in sorted(self.similarity_grades.items(), 
                                     key=lambda x: x[1], reverse=True):
            if similarity >= threshold:
                return grade
        return 'F'
    
    def _assess_analysis_confidence(self, detailed_analyses: Dict, features1: Dict, features2: Dict) -> str:
        """Assess confidence level of DTW analysis"""
        # Count successful analyses
        successful_analyses = 0
        total_analyses = len(detailed_analyses)
        
        for analysis in detailed_analyses.values():
            if 'error' not in analysis and analysis.get('overall_similarity', 0) > 0:
                successful_analyses += 1
        
        success_rate = successful_analyses / total_analyses if total_analyses > 0 else 0
        
        # Check data quality
        data_quality_score = self._assess_data_quality(features1, features2)
        
        # Combined confidence
        combined_score = (success_rate + data_quality_score) / 2
        
        if combined_score >= self.confidence_thresholds['high']:
            return 'High'
        elif combined_score >= self.confidence_thresholds['medium']:
            return 'Medium'
        else:
            return 'Low'
    
    def _assess_data_quality(self, features1: Dict, features2: Dict) -> float:
        """Assess quality of input feature data"""
        quality_scores = []
        
        # Check frame counts
        frames1 = features1['metadata']['total_frames']
        frames2 = features2['metadata']['total_frames']
        
        if frames1 > 30 and frames2 > 30:  # Minimum reasonable length
            frame_score = min(1.0, min(frames1, frames2) / max(frames1, frames2))
            quality_scores.append(frame_score)
        else:
            quality_scores.append(0.3)  # Low score for very short sequences
        
        # Check phase distribution
        phases1 = features1['metadata']['phase_distribution']
        phases2 = features2['metadata']['phase_distribution']
        
        shooting_phases = ['Setup', 'Loading', 'Rising', 'Release', 'Follow-through']
        phase_coverage1 = sum(1 for phase in shooting_phases if phases1.get(phase, 0) > 0)
        phase_coverage2 = sum(1 for phase in shooting_phases if phases2.get(phase, 0) > 0)
        
        phase_score = min(phase_coverage1, phase_coverage2) / len(shooting_phases)
        quality_scores.append(phase_score)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _calculate_consistency_metrics(self, detailed_analyses: Dict) -> Dict:
        """Calculate motion consistency metrics from DTW analyses"""
        consistency_metrics = {}
        
        # Overall consistency score
        similarities = []
        warping_ratios = []
        
        for feature_name, analysis in detailed_analyses.items():
            if 'error' in analysis:
                continue
                
            similarities.append(analysis.get('overall_similarity', 0))
            
            # Extract warping ratios from subfeatures
            dtw_analysis = analysis.get('dtw_analysis', {})
            for subfeature, sub_analysis in dtw_analysis.items():
                if isinstance(sub_analysis, dict) and 'warping_ratio' in sub_analysis:
                    warping_ratios.append(sub_analysis['warping_ratio'])
        
        if similarities:
            consistency_metrics['similarity_std'] = float(np.std(similarities))
            consistency_metrics['similarity_mean'] = float(np.mean(similarities))
            consistency_metrics['similarity_consistency'] = float(max(0, 100 - np.std(similarities)))
        
        if warping_ratios:
            consistency_metrics['temporal_consistency'] = float(np.mean(warping_ratios))
            consistency_metrics['temporal_variability'] = float(np.std(warping_ratios))
        
        # Feature-specific consistency
        consistency_metrics['most_consistent_feature'] = self._find_most_consistent_feature(detailed_analyses)
        consistency_metrics['least_consistent_feature'] = self._find_least_consistent_feature(detailed_analyses)
        
        return consistency_metrics
    
    def _analyze_temporal_alignment(self, detailed_analyses: Dict) -> Dict:
        """Analyze temporal alignment patterns from DTW results"""
        alignment_analysis = {}
        
        # Collect warping information
        warping_info = {}
        
        for feature_name, analysis in detailed_analyses.items():
            if 'error' in analysis:
                continue
            
            feature_warping = []
            dtw_analysis = analysis.get('dtw_analysis', {})
            
            for subfeature, sub_analysis in dtw_analysis.items():
                if isinstance(sub_analysis, dict) and 'warping_ratio' in sub_analysis:
                    feature_warping.append({
                        'subfeature': subfeature,
                        'warping_ratio': sub_analysis['warping_ratio'],
                        'distance': sub_analysis.get('distance', sub_analysis.get('combined_distance', 0))
                    })
            
            if feature_warping:
                warping_info[feature_name] = feature_warping
        
        # Overall temporal analysis
        all_warping_ratios = []
        for feature_data in warping_info.values():
            for item in feature_data:
                all_warping_ratios.append(item['warping_ratio'])
        
        if all_warping_ratios:
            alignment_analysis['average_warping_ratio'] = float(np.mean(all_warping_ratios))
            alignment_analysis['warping_consistency'] = float(100 - min(100, np.std(all_warping_ratios) * 50))
            
            # Classify temporal patterns
            avg_warping = np.mean(all_warping_ratios)
            if avg_warping > 1.2:
                alignment_analysis['temporal_pattern'] = 'stretched'
            elif avg_warping < 0.8:
                alignment_analysis['temporal_pattern'] = 'compressed'  
            else:
                alignment_analysis['temporal_pattern'] = 'aligned'
        
        # Feature-specific temporal analysis
        alignment_analysis['feature_temporal_patterns'] = {}
        for feature_name, feature_data in warping_info.items():
            if feature_data:
                avg_warping = np.mean([item['warping_ratio'] for item in feature_data])
                alignment_analysis['feature_temporal_patterns'][feature_name] = {
                    'average_warping': float(avg_warping),
                    'pattern': 'stretched' if avg_warping > 1.2 else ('compressed' if avg_warping < 0.8 else 'aligned')
                }
        
        return alignment_analysis
    
    def _find_most_consistent_feature(self, detailed_analyses: Dict) -> Dict:
        """Find the most consistent feature across DTW analysis"""
        feature_scores = {}
        
        for feature_name, analysis in detailed_analyses.items():
            if 'error' in analysis:
                continue
            
            similarity = analysis.get('overall_similarity', 0)
            
            # Look for consistency indicators in subfeatures
            dtw_analysis = analysis.get('dtw_analysis', {})
            warping_ratios = []
            
            for sub_analysis in dtw_analysis.values():
                if isinstance(sub_analysis, dict) and 'warping_ratio' in sub_analysis:
                    warping_ratios.append(sub_analysis['warping_ratio'])
            
            # Score based on similarity and temporal consistency
            consistency_score = similarity
            if warping_ratios:
                # Lower warping variability = higher consistency
                warping_consistency = max(0, 100 - np.std(warping_ratios) * 50)
                consistency_score = (similarity + warping_consistency) / 2
            
            feature_scores[feature_name] = consistency_score
        
        if feature_scores:
            best_feature = max(feature_scores.items(), key=lambda x: x[1])
            return {
                'feature': best_feature[0],
                'score': float(best_feature[1])
            }
        
        return {'feature': 'none', 'score': 0.0}
    
    def _find_least_consistent_feature(self, detailed_analyses: Dict) -> Dict:
        """Find the least consistent feature across DTW analysis"""
        feature_scores = {}
        
        for feature_name, analysis in detailed_analyses.items():
            if 'error' in analysis:
                feature_scores[feature_name] = 0.0
                continue
            
            similarity = analysis.get('overall_similarity', 0)
            
            # Look for consistency indicators in subfeatures
            dtw_analysis = analysis.get('dtw_analysis', {})
            warping_ratios = []
            
            for sub_analysis in dtw_analysis.values():
                if isinstance(sub_analysis, dict) and 'warping_ratio' in sub_analysis:
                    warping_ratios.append(sub_analysis['warping_ratio'])
            
            # Score based on similarity and temporal consistency
            consistency_score = similarity
            if warping_ratios:
                # Higher warping variability = lower consistency
                warping_consistency = max(0, 100 - np.std(warping_ratios) * 50)
                consistency_score = (similarity + warping_consistency) / 2
            
            feature_scores[feature_name] = consistency_score
        
        if feature_scores:
            worst_feature = min(feature_scores.items(), key=lambda x: x[1])
            return {
                'feature': worst_feature[0],
                'score': float(worst_feature[1])
            }
        
        return {'feature': 'none', 'score': 0.0}