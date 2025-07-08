# -*- coding: utf-8 -*-
"""
포즈 데이터 정규화 레이어
추출된 포즈 데이터를 분석에 적합한 형태로 정규화
"""

import numpy as np
from typing import Dict, List, Optional

class PoseNormalizationLayer:
    def __init__(self):
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        # 이전 프레임의 엉덩이 정보 저장
        self.previous_hip_data = None
        self.low_confidence_frames = []  # 신뢰도 낮은 프레임 기록

    def select_anchor_hip(self, pose: Dict, confidence_threshold: float = 0.3) -> Optional[Dict]:
        """ANCHOR POINT 방식으로 엉덩이 키포인트 선택"""
        left_hip_conf = pose.get('left_hip', {}).get('confidence', 0)
        right_hip_conf = pose.get('right_hip', {}).get('confidence', 0)
        
        # 둘 다 신뢰도가 기준 이상인 경우
        if left_hip_conf >= confidence_threshold and right_hip_conf >= confidence_threshold:
            # 신뢰도가 높은 쪽 선택
            if left_hip_conf >= right_hip_conf:
                selected_hip = pose['left_hip']
                hip_source = 'left_hip'
            else:
                selected_hip = pose['right_hip']
                hip_source = 'right_hip'
            
            # 이전 프레임 정보 업데이트
            self.previous_hip_data = {
                'x': selected_hip['x'],
                'y': selected_hip['y'],
                'confidence': selected_hip['confidence'],
                'source': hip_source
            }
            
            return {
                'left_hip': selected_hip,
                'right_hip': selected_hip,  # 대칭을 위해 같은 값 사용
                'anchor_source': hip_source,
                'used_previous': False
            }
        
        # 둘 다 신뢰도가 낮은 경우
        elif left_hip_conf < confidence_threshold and right_hip_conf < confidence_threshold:
            # 이전 프레임 정보가 있으면 사용
            if self.previous_hip_data:
                print(f"⚠️ 엉덩이 신뢰도 낮음 (left: {left_hip_conf:.3f}, right: {right_hip_conf:.3f}) - 이전 프레임 사용")
                
                # 낮은 신뢰도 프레임 기록
                self.low_confidence_frames.append({
                    'frame_number': pose.get('frame_number', 0),
                    'left_hip_conf': left_hip_conf,
                    'right_hip_conf': right_hip_conf,
                    'used_previous': True
                })
                
                return {
                    'left_hip': self.previous_hip_data,
                    'right_hip': self.previous_hip_data,
                    'anchor_source': self.previous_hip_data['source'],
                    'used_previous': True
                }
            else:
                print(f"❌ 엉덩이 신뢰도 낮음 (left: {left_hip_conf:.3f}, right: {right_hip_conf:.3f}) - 이전 프레임 없음")
                return None
        
        # 한쪽만 신뢰도가 높은 경우
        else:
            if left_hip_conf >= confidence_threshold:
                selected_hip = pose['left_hip']
                hip_source = 'left_hip'
            else:
                selected_hip = pose['right_hip']
                hip_source = 'right_hip'
            
            # 이전 프레임 정보 업데이트
            self.previous_hip_data = {
                'x': selected_hip['x'],
                'y': selected_hip['y'],
                'confidence': selected_hip['confidence'],
                'source': hip_source
            }
            
            return {
                'left_hip': selected_hip,
                'right_hip': selected_hip,
                'anchor_source': hip_source,
                'used_previous': False
            }

    def normalize_pose_by_body_ratio(self, pose: Dict) -> Dict:
        """신체 비율을 기준으로 포즈 정규화 (ANCHOR POINT 방식)"""
        # 필수 키포인트 확인
        required_keypoints = ['left_shoulder', 'right_shoulder']
        missing_keypoints = [kp for kp in required_keypoints if kp not in pose]
        
        if missing_keypoints:
            print(f"⚠️ 경고: 누락된 키포인트: {missing_keypoints}")
            return self.normalize_pose_by_bounds(pose)
        
        # ANCHOR POINT 방식으로 엉덩이 선택
        hip_data = self.select_anchor_hip(pose)
        if hip_data is None:
            print("❌ 엉덩이 키포인트를 선택할 수 없음 - bounds 방식 사용")
            return self.normalize_pose_by_bounds(pose)
        
        # 어깨 중심점 계산
        left_shoulder = pose['left_shoulder']
        right_shoulder = pose['right_shoulder']
        shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        
        # 선택된 엉덩이 중심점 계산
        left_hip = hip_data['left_hip']
        right_hip = hip_data['right_hip']
        hip_center_x = (left_hip['x'] + right_hip['x']) / 2
        hip_center_y = (left_hip['y'] + right_hip['y']) / 2
        
        # 신체 길이 (어깨-엉덩이 거리) 계산
        body_length = np.sqrt(
            (shoulder_center_x - hip_center_x)**2 + 
            (shoulder_center_y - hip_center_y)**2
        )
        
        if body_length < 0.001:
            body_length = 1.0
        
        # 신체 중심점 계산
        body_center_x = (shoulder_center_x + hip_center_x) / 2
        body_center_y = (shoulder_center_y + hip_center_y) / 2
        
        # 모든 키포인트를 신체 비율로 정규화
        norm_pose = {}
        for name, kp in pose.items():
            relative_x = (kp['x'] - body_center_x) / body_length
            relative_y = (kp['y'] - body_center_y) / body_length
            
            norm_pose[name] = {
                'x': relative_x,
                'y': relative_y,
                'confidence': kp['confidence']
            }
        
        # ANCHOR POINT로 선택된 엉덩이 키포인트 추가
        left_hip_relative_x = (left_hip['x'] - body_center_x) / body_length
        left_hip_relative_y = (left_hip['y'] - body_center_y) / body_length
        right_hip_relative_x = (right_hip['x'] - body_center_x) / body_length
        right_hip_relative_y = (right_hip['y'] - body_center_y) / body_length
        
        norm_pose['left_hip'] = {
            'x': left_hip_relative_x,
            'y': left_hip_relative_y,
            'confidence': left_hip['confidence']
        }
        norm_pose['right_hip'] = {
            'x': right_hip_relative_x,
            'y': right_hip_relative_y,
            'confidence': right_hip['confidence']
        }
        
        return norm_pose

    def normalize_pose_by_bounds(self, pose: Dict) -> Dict:
        """좌표 범위를 기준으로 포즈 정규화 (0-1 범위)"""
        xs = [kp['x'] for kp in pose.values()]
        ys = [kp['y'] for kp in pose.values()]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        norm_pose = {}
        for name, kp in pose.items():
            norm_pose[name] = {
                'x': (kp['x'] - min_x) / (max_x - min_x) if max_x > min_x else 0.5,
                'y': (kp['y'] - min_y) / (max_y - min_y) if max_y > min_y else 0.5,
                'confidence': kp['confidence']
            }
        return norm_pose

    def filter_low_confidence_poses(self, pose_data: List[Dict], confidence_threshold: float = 0.3) -> List[Dict]:
        """낮은 신뢰도의 키포인트 필터링"""
        filtered_data = []
        
        for frame_data in pose_data:
            filtered_pose = {}
            for name, kp in frame_data['pose'].items():
                if kp['confidence'] >= confidence_threshold:
                    filtered_pose[name] = kp
            
            # 필수 키포인트는 신뢰도가 낮아도 유지
            required_keypoints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
            for kp_name in required_keypoints:
                if kp_name in frame_data['pose'] and kp_name not in filtered_pose:
                    filtered_pose[kp_name] = frame_data['pose'][kp_name]
                    print(f"⚠️ {kp_name} 키포인트 유지 (신뢰도: {frame_data['pose'][kp_name]['confidence']:.3f})")
            
            if len(filtered_pose) >= 8:  # 최소 8개 키포인트가 있어야 유효
                frame_data['pose'] = filtered_pose
                filtered_data.append(frame_data)
        
        print(f"신뢰도 필터링: {len(pose_data)} -> {len(filtered_data)} 프레임")
        return filtered_data

    def normalize_all_poses(self, pose_data: List[Dict], method: str = "body_ratio") -> List[Dict]:
        """모든 포즈 데이터 정규화"""
        print(f"포즈 데이터 정규화 시작 ({method} 방식)")
        
        normalized_data = []
        for frame_data in pose_data:
            if method == "body_ratio":
                normalized_pose = self.normalize_pose_by_body_ratio(frame_data['pose'])
            elif method == "bounds":
                normalized_pose = self.normalize_pose_by_bounds(frame_data['pose'])
            else:
                raise ValueError(f"알 수 없는 정규화 방식: {method}")
            
            normalized_frame = {
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp'],
                'pose': normalized_pose
            }
            normalized_data.append(normalized_frame)
        
        print(f"정규화 완료: {len(normalized_data)} 프레임")
        return normalized_data

    def validate_pose_data(self, pose_data: List[Dict]) -> bool:
        """포즈 데이터 유효성 검증"""
        if not pose_data:
            return False
        
        required_keypoints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        
        for frame_data in pose_data:
            pose = frame_data['pose']
            
            # 필수 키포인트 확인
            for kp in required_keypoints:
                if kp not in pose:
                    print(f"❌ 누락된 키포인트: {kp}")
                    return False
            
            # 신뢰도 확인
            valid_keypoints = sum(1 for kp in pose.values() if kp['confidence'] > 0.2)
            if valid_keypoints < 8:  # 최소 키포인트 수를 8개로 낮춤
                print(f"❌ 유효한 키포인트 수 부족: {valid_keypoints}/8")
                return False
        
        return True

    def get_low_confidence_report(self) -> Dict:
        """낮은 신뢰도 프레임 분석 리포트"""
        if not self.low_confidence_frames:
            return {"message": "낮은 신뢰도 프레임 없음"}
        
        # 연속된 낮은 신뢰도 구간 찾기
        consecutive_ranges = []
        start_frame = self.low_confidence_frames[0]['frame_number']
        end_frame = start_frame
        
        for i in range(1, len(self.low_confidence_frames)):
            current_frame = self.low_confidence_frames[i]['frame_number']
            if current_frame == end_frame + 1:
                end_frame = current_frame
            else:
                if end_frame - start_frame >= 3:  # 3프레임 이상 연속
                    consecutive_ranges.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'duration': end_frame - start_frame + 1
                    })
                start_frame = current_frame
                end_frame = current_frame
        
        # 마지막 구간 처리
        if end_frame - start_frame >= 3:
            consecutive_ranges.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration': end_frame - start_frame + 1
            })
        
        return {
            "total_low_confidence_frames": len(self.low_confidence_frames),
            "consecutive_ranges": consecutive_ranges,
            "warning": "이 구간들은 분석 시 제외하는 것을 권장합니다."
        } 