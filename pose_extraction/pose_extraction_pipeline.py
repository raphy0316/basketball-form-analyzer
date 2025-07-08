# -*- coding: utf-8 -*-
"""
포즈 추출 통합 파이프라인
원본 절대좌표만 추출하여 JSON으로 저장
"""

import os
import sys
from typing import Dict, List, Optional

# 레이어 모듈들 import
from .pose_model_layer import PoseModelLayer
from .pose_storage_layer import PoseStorageLayer

class PoseExtractionPipeline:
    def __init__(self, output_dir: str = "data"):
        """포즈 추출 파이프라인 초기화"""
        self.model_layer = PoseModelLayer()
        self.storage_layer = PoseStorageLayer(output_dir)
        
        print("포즈 추출 파이프라인 초기화 완료")
        print("=" * 50)

    def extract_poses(self, video_path: str, confidence_threshold: float = 0.3) -> str:
        """
        원본 절대좌표 포즈 추출 파이프라인 실행
        
        Args:
            video_path: 비디오 파일 경로
            confidence_threshold: 신뢰도 임계값
        
        Returns:
            저장된 파일 경로
        """
        print(f"🎬 비디오 파일: {video_path}")
        print(f"🎯 신뢰도 임계값: {confidence_threshold}")
        print("-" * 50)
        
        try:
            # 1단계: 모델 레이어 - 원본 포즈 데이터 추출
            print("🔍 1단계: 원본 포즈 데이터 추출 중...")
            raw_pose_data = self.model_layer.extract_poses_from_video(video_path)
            print(f"✅ 추출 완료: {len(raw_pose_data)} 프레임")
            
            # 2단계: 신뢰도 필터링 (낮은 신뢰도 키포인트 제거)
            print("\n🔄 2단계: 신뢰도 필터링 중...")
            filtered_data = self._filter_low_confidence_poses(raw_pose_data, confidence_threshold)
            print(f"✅ 필터링 완료: {len(filtered_data)} 프레임")
            
            # 3단계: 원본 절대좌표 JSON 저장
            print("\n💾 3단계: 원본 데이터 저장 중...")
            base_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_pose_original"
            saved_file = self.storage_layer.save_original_as_json(filtered_data, f"{base_filename}.json")
            
            print("✅ 저장 완료")
            print("=" * 50)
            
            # 결과 요약
            self._print_summary(filtered_data, saved_file)
            
            return saved_file
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            raise

    def _filter_low_confidence_poses(self, pose_data: List[Dict], confidence_threshold: float) -> List[Dict]:
        """낮은 신뢰도 키포인트 필터링"""
        filtered_data = []
        
        for frame_data in pose_data:
            filtered_frame = {
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp'],
                'pose': {}
            }
            
            for kp_name, kp_data in frame_data['pose'].items():
                if kp_data['confidence'] >= confidence_threshold:
                    # 원본 절대좌표만 저장
                    filtered_frame['pose'][kp_name] = {
                        'x': kp_data['x'],  # 원본 픽셀 좌표
                        'y': kp_data['y'],  # 원본 픽셀 좌표
                        'confidence': kp_data['confidence']
                    }
            
            filtered_data.append(filtered_frame)
        
        return filtered_data

    def _print_summary(self, pose_data: List[Dict], saved_file: str):
        """결과 요약 출력"""
        print("\n📋 추출 결과 요약:")
        print(f"   • 총 프레임 수: {len(pose_data)}")
        print(f"   • 키포인트 수: {len(pose_data[0]['pose']) if pose_data else 0}")
        print(f"   • 저장된 파일: {os.path.basename(saved_file)}")
        print(f"   • 좌표 시스템: 원본 절대좌표 (픽셀 단위)")

    def get_pipeline_info(self) -> Dict:
        """파이프라인 정보 반환"""
        storage_info = self.storage_layer.get_storage_info()
        
        return {
            "model_info": {
                "model_name": self.model_layer.model_name,
                "keypoint_count": len(self.model_layer.keypoint_names)
            },
            "storage_info": storage_info
        }

def main():
    """메인 실행 함수"""
    print("🏀 농구 포즈 추출 파이프라인 (원본 절대좌표)")
    print("=" * 50)
    
    # 파이프라인 초기화
    pipeline = PoseExtractionPipeline()
    
    # 비디오 파일 경로 설정
    video_path = "../References/stephen_curry_multy_person_part.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
        return
    
    try:
        # 포즈 추출 실행
        saved_file = pipeline.extract_poses(
            video_path=video_path,
            confidence_threshold=0.3
        )
        
        print("\n🎉 포즈 추출 파이프라인 완료!")
        
        # 파이프라인 정보 출력
        info = pipeline.get_pipeline_info()
        print(f"\n📊 파이프라인 정보:")
        print(f"   • 모델: {info['model_info']['model_name']}")
        print(f"   • 키포인트 수: {info['model_info']['keypoint_count']}")
        print(f"   • 저장소: {info['storage_info']['output_dir']}")
        
    except Exception as e:
        print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 