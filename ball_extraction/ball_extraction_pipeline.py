# -*- coding: utf-8 -*-
"""
농구공 추출 통합 파이프라인
원본 절대좌표만 추출하여 JSON으로 저장
"""

import os
import sys
import cv2
from typing import Dict, List, Optional

# 레이어 모듈들 import
from .ball_detection_layer import BallDetectionLayer
from .ball_storage_layer import BallStorageLayer

class BallExtractionPipeline:
    def __init__(self, model_path: str = "ball_extraction/yolov8n736-customContinue.pt", output_dir: str = "data"):
        """농구공 추출 파이프라인 초기화"""
        self.detection_layer = BallDetectionLayer(model_path)
        self.storage_layer = BallStorageLayer(output_dir)
        
        print("농구공 추출 파이프라인 초기화 완료")
        print("=" * 50)

    def extract_ball_trajectory(self, video_path: str, conf_threshold: float = 0.15,
                               classes: List[int] = [0, 1, 2], iou_threshold: float = 0.1,
                               min_confidence: float = 0.3, min_ball_size: float = 10.0) -> str:
        """
        원본 절대좌표 농구공 추출 파이프라인 실행
        
        Args:
            video_path: 비디오 파일 경로
            conf_threshold: YOLO 신뢰도 임계값
            classes: 감지할 클래스
            iou_threshold: IoU 임계값
            min_confidence: 최소 신뢰도 (필터링용)
            min_ball_size: 최소 공 크기 (픽셀)
        
        Returns:
            저장된 파일 경로
        """
        print(f"🏀 비디오 파일: {video_path}")
        print(f"🎯 신뢰도 임계값: {conf_threshold}")
        print(f"📊 필터링 임계값: {min_confidence}")
        print("-" * 50)
        
        try:
            # 1단계: 감지 레이어 - 원본 공 궤적 추출
            print("🔍 1단계: 원본 농구공 궤적 추출 중...")
            raw_ball_trajectory = self.detection_layer.extract_ball_trajectory_from_video(
                video_path, conf_threshold, classes, iou_threshold
            )
            print(f"✅ 추출 완료: {len(raw_ball_trajectory)} 프레임")
            
            # 2단계: 신뢰도 및 크기 필터링
            print("\n🔄 2단계: 신뢰도 필터링 중...")
            filtered_trajectory = self.detection_layer.filter_ball_detections(
                raw_ball_trajectory, min_confidence, min_ball_size
            )
            print(f"✅ 필터링 완료: {len(filtered_trajectory)} 프레임")
            
            # 3단계: 원본 절대좌표 JSON 저장
            print("\n💾 3단계: 원본 데이터 저장 중...")
            base_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_ball_original"
            saved_file = self.storage_layer.save_original_as_json(filtered_trajectory, f"{base_filename}.json")
            
            print("✅ 저장 완료")
            print("=" * 50)
            
            # 결과 요약
            self._print_summary(filtered_trajectory, saved_file)
            
            return saved_file
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            raise

    def _print_summary(self, ball_trajectory: List[Dict], saved_file: str):
        """결과 요약 출력"""
        # 통계 정보
        stats = self.detection_layer.get_ball_statistics(ball_trajectory)
        
        print("\n📋 공 궤적 추출 결과 요약:")
        print(f"   • 총 프레임 수: {stats['total_frames']}")
        print(f"   • 공 감지 프레임: {stats['frames_with_ball']}")
        print(f"   • 감지율: {stats['detection_rate']:.2%}")
        print(f"   • 총 감지된 공: {stats['total_balls_detected']}개")
        print(f"   • 평균 신뢰도: {stats['avg_confidence']:.3f}")
        print(f"   • 저장된 파일: {os.path.basename(saved_file)}")
        print(f"   • 좌표 시스템: 원본 절대좌표 (픽셀 단위)")

    def get_pipeline_info(self) -> Dict:
        """파이프라인 정보 반환"""
        storage_info = self.storage_layer.get_storage_info()
        
        return {
            "model_info": {
                "model_path": self.detection_layer.model_path
            },
            "storage_info": storage_info
        }

def main():
    """메인 실행 함수"""
    print("🏀 농구공 궤적 추출 파이프라인 (원본 절대좌표)")
    print("=" * 50)
    
    # 파이프라인 초기화
    pipeline = BallExtractionPipeline()
    
    # 비디오 파일 경로 설정
    video_path = "../References/stephen_curry_multy_person_part.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
        return
    
    try:
        # 공 궤적 추출 실행
        saved_file = pipeline.extract_ball_trajectory(
            video_path=video_path,
            conf_threshold=0.15,
            classes=[0, 1, 2],
            iou_threshold=0.1,
            min_confidence=0.3,
            min_ball_size=10.0
        )
        
        print("\n🎉 농구공 궤적 추출 파이프라인 완료!")
        
        # 파이프라인 정보 출력
        info = pipeline.get_pipeline_info()
        print(f"\n📊 파이프라인 정보:")
        print(f"   • 모델: {info['model_info']['model_path']}")
        print(f"   • 저장소: {info['storage_info']['output_dir']}")
        
    except Exception as e:
        print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 