# Basketball Shooting Form Comparison

슛폼 비교를 위한 파이프라인입니다. 두 개의 동영상을 선택하여 DTW(Dynamic Time Warping) 기법을 사용해 슛폼을 비교 분석합니다.

## 기능

### 주요 기능
- **두 동영상 선택**: GUI를 통한 비교할 동영상 선택
- **자동 데이터 처리**: Integrated Pipeline을 사용한 포즈/공 데이터 추출
- **DTW 기반 비교**: 좌표 기반 및 특징 기반 비교 분석
- **페이즈별 비교**: Loading & Rising, Release, Follow-through 단계별 비교
- **결과 저장**: JSON 형식으로 비교 결과 저장

### 비교 분석 종류
1. **전체 페이즈 비교**
   - 좌표 기반 (Coordinate-based)
   - 특징 기반 (Feature-based)

2. **페이즈별 비교**
   - Loading & Rising 페이즈
   - Release 페이즈
   - Follow-through 페이즈

## 사용 방법

### 기본 실행
```bash
cd shooting_comparison
python shooting_comparison_pipeline.py
```

### 프로그래밍 방식 사용
```python
from shooting_comparison import ShootingComparisonPipeline

# 파이프라인 초기화
pipeline = ShootingComparisonPipeline()

# 전체 비교 실행
success = pipeline.run_comparison()

# 개별 단계 실행
video1_path, video2_path = pipeline.select_videos()
video1_data = pipeline.process_video_data(video1_path)
video2_data = pipeline.process_video_data(video2_path)
results = pipeline.perform_comparison()
```

## 파일 구조

```
shooting_comparison/
├── __init__.py                     # 모듈 초기화
├── shooting_comparison_pipeline.py # 메인 파이프라인
├── results/                        # 비교 결과 저장 디렉토리
└── README.md                       # 설명서
```

## 의존성

- basketball_shooting_integrated_pipeline
- data_collection.dtw_processor
- tkinter (GUI)
- opencv-python
- numpy
- json

## 출력 결과

### 콘솔 출력
- 처리 과정 실시간 표시
- DTW 거리 결과
- 페이즈별 통계
- 비교 요약

### 파일 출력
- `comparison_{video1}_vs_{video2}_{timestamp}.json`
- DTW 분석 결과
- 메타데이터
- 페이즈 통계

## 결과 해석

### DTW 거리
- **낮은 값**: 유사한 슛폼
- **높은 값**: 다른 슛폼

### 페이즈 분포
- 각 페이즈별 프레임 수
- 두 영상 간 페이즈 비교

## 예시

```bash
🏀 Basketball Shooting Form Comparison Pipeline
============================================================

🎬 STEP 1: Select Videos for Comparison
==================================================
📹 Select the first video (Reference):
✅ First video selected: stephen_curry_part.mp4
📹 Select the second video (Comparison):
✅ Second video selected: sample_shot.mp4

🔄 STEP 2: Processing Videos
==================================================
🔍 Processing: stephen_curry_part.mp4
✅ Found existing results: stephen_curry_part_normalized_output.json
📊 Loaded 150 frames

🔍 Processing: sample_shot.mp4
🚀 Processing video with integrated pipeline...
✅ Successfully processed 120 frames

🔄 STEP 3: Performing DTW Comparison
==================================================
📊 Performing coordinate-based overall comparison...
📊 Performing feature-based overall comparison...
📊 Performing loading & rising phases comparison...
📊 Performing release phase comparison...
📊 Performing follow-through phase comparison...
✅ DTW comparison completed successfully!

💾 STEP 4: Saving Comparison Results
==================================================
✅ Comparison results saved: comparison_stephen_curry_part_vs_sample_shot_20250801_232600.json
📁 Location: shooting_comparison/results/comparison_stephen_curry_part_vs_sample_shot_20250801_232600.json

📋 COMPARISON SUMMARY
==================================================
📹 Video 1 (Reference): stephen_curry_part.mp4
📹 Video 2 (Comparison): sample_shot.mp4
🖐 Selected Hand: right
📊 Video 1 Frames: 150
📊 Video 2 Frames: 120

🔍 DTW Distance Results:
  • Coordinate Overall: 245.67
  • Feature Overall: 198.34
  • Loading & Rising: 156.78
  • Release: 89.12
  • Follow-through: 134.56

📈 Phase Distribution:
  • Follow-through: Video1=25, Video2=20
  • General: Video1=80, Video2=65
  • Loading: Video1=15, Video2=12
  • Release: Video1=8, Video2=6
  • Rising: Video1=12, Video2=10
  • Set-up: Video1=10, Video2=7

🎉 Shooting form comparison completed successfully!
```

## Notes

### Bugs fixes

* dip point reversed y
* selected hand is always right in analyzers(follow-through, rising)
* swapping logic in normalization
* height reversed y

### TODO 
* dip point angles are not calculated
* frame numbers in jump height is not actual frame number of the whole video, it's just the frame index of rising frames
ex
```bash
Video 2 Rising Analysis:
    Total Rising Time: 0.7451052631578947s
    Rising Frames: 13
    Loading-Rising Frames: 32
    Combined Rising Frames: 45
    Jump Analysis:
      Max Jump Height: 1.8575
      Max Height Frame: 44 # this is just frame index of the rising frames
      Max Height Time: 0.729s
      Setup Time: 0.000s
      Relative Timing: -0.729s
```
* made up toes and fingers
* frame numbers in follow-through is not actual frame number of the whole video, it's just the frame index of follow-through frames
```bash
Video 2 Follow-through Analysis:
    Total Follow-through Time: 0.7119894736842105s
    Follow-through Frames: 43
    Max Elbow Angle Analysis:
      Max Elbow Angle: 180.00°
      Max Elbow Frame Index: 7
      Arm Angles Std: 10.34°
      Body Angles Std: 0.41°
      Leg Angles Std: 0.39°
      Overall Angles Std: 7.20°
    Stability Analysis:
```
* selected hand is always right in release analyzer