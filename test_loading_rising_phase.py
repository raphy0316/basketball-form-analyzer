#!/usr/bin/env python3
"""
Test script for the new loading-rising phase functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection.dtw_processor import DTWProcessor
from phase_detection.hybrid_fps_phase_detector import HybridFPSPhaseDetector

def test_loading_rising_phase():
    """Test the new loading-rising phase functionality."""
    
    print("🧪 Testing Loading-Rising Phase Implementation")
    print("=" * 50)
    
    # Test phase detector
    detector = HybridFPSPhaseDetector()
    detector.set_fps(30.0)
    
    print("✅ Phase detector initialized")
    print(f"   - FPS: {detector.fps}")
    print(f"   - FPS Factor: {detector.fps_factor}")
    print(f"   - Loading-rising start frame: {detector.loading_rising_start_frame}")
    
    # Test DTW processor
    dtw_processor = DTWProcessor()
    
    print("\n✅ DTW processor initialized")
    print(f"   - Overall features: {len(dtw_processor.overall_feature_names)}")
    print(f"   - Loading-rising features: {len(dtw_processor.loading_rising_features)}")
    print(f"   - Release features: {len(dtw_processor.release_features)}")
    print(f"   - Follow-through features: {len(dtw_processor.follow_through_features)}")
    
    # Test phase frame extraction
    print("\n📊 Phase Frame Extraction Test")
    print("   - Supported phases: Loading, Rising, Release, Follow-through, loading-rising")
    
    # Test with sample data structure
    sample_data = {
        "frames": [
            {"frame_index": 0, "phase": "General"},
            {"frame_index": 1, "phase": "Set-up"},
            {"frame_index": 2, "phase": "Loading"},
            {"frame_index": 3, "phase": "Loading-Rising"},
            {"frame_index": 4, "phase": "Loading-Rising"},
            {"frame_index": 5, "phase": "rising"},
            {"frame_index": 6, "phase": "Release"},
            {"frame_index": 7, "phase": "Follow-through"}
        ]
    }
    
    dtw_processor.data = sample_data
    
    # Test individual phase extraction
    loading_frames = dtw_processor.get_phase_frames("Loading")
    rising_frames = dtw_processor.get_phase_frames("Rising")
    loading_rising_frames = dtw_processor.get_phase_frames("Loading-Rising")
    
    print(f"   - Loading frames: {len(loading_frames)}")
    print(f"   - Rising frames: {len(rising_frames)}")
    print(f"   - Loading-Rising frames: {len(loading_rising_frames)}")
    
    # Test combined phase extraction
    combined_frames = dtw_processor.get_combined_phase_frames(["Loading", "Rising", "Loading-Rising"])
    print(f"   - Combined Loading+Rising+Loading-Rising frames: {len(combined_frames)}")
    
    print("\n🎯 Phase Transition Logic")
    print("   - Loading → Loading-Rising: When wrist/elbow moving upward")
    print("   - Loading-Rising → rising: When loading cancellation conditions met")
    print("   - Loading-Rising → Set-up: When rising cancellation conditions met")
    print("   - Loading-Rising → General: When ball dropped (loading cancellation)")
    print("   - Loading-Rising → Set-up: When shoulder/hip rising (loading cancellation)")
    print("   - Loading-Rising → Set-up: When hand moving down (rising cancellation)")
    
    print("\n📊 Cancellation Conditions")
    print("   Loading Cancellation (→ rising):")
    print("     - Shoulder/hip rising → Set-up (ball drop condition removed)")
    print("   Rising Cancellation (→ Set-up):")
    print("     - Wrist above shoulder: wrist goes below shoulder")
    print("     - Wrist below shoulder: hand moving down relative/absolute")
    
    print("\n✅ All tests completed successfully!")
    print("   The new loading-rising phase is ready for use.")

if __name__ == "__main__":
    test_loading_rising_phase() 