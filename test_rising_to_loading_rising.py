#!/usr/bin/env python3
"""
Test script for Rising → Loading-Rising phase transition

This script tests the new functionality where Rising phase can transition
to Loading-Rising phase when loading conditions (hip/shoulder moving down) are detected.
"""

import os
import sys
import json
from typing import List, Dict

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase_detection.hybrid_fps_phase_detector import HybridFPSPhaseDetector


def create_test_pose_data() -> List[Dict]:
    """Create test pose data for Rising → Loading-Rising transition"""
    test_data = []
    
    # Base pose keypoints
    base_pose = {
        'left_shoulder': {'x': 200, 'y': 150},
        'right_shoulder': {'x': 250, 'y': 150},
        'left_elbow': {'x': 180, 'y': 200},
        'right_elbow': {'x': 270, 'y': 200},
        'left_wrist': {'x': 160, 'y': 250},
        'right_wrist': {'x': 290, 'y': 250},
        'left_hip': {'x': 210, 'y': 300},
        'right_hip': {'x': 240, 'y': 300},
        'left_knee': {'x': 215, 'y': 400},
        'right_knee': {'x': 235, 'y': 400},
        'left_ankle': {'x': 220, 'y': 500},
        'right_ankle': {'x': 230, 'y': 500},
        'left_eye': {'x': 215, 'y': 100},
        'right_eye': {'x': 235, 'y': 100}
    }
    
    # Frame 0: Initial Rising state
    frame0 = {'pose': base_pose.copy()}
    test_data.append(frame0)
    
    # Frame 1-3: Rising phase continues (hip/shoulder stable)
    for i in range(1, 4):
        frame = {'pose': base_pose.copy()}
        # Slight wrist movement up (maintaining Rising)
        frame['pose']['left_wrist']['y'] -= i * 2
        frame['pose']['right_wrist']['y'] -= i * 2
        test_data.append(frame)
    
    # Frame 4-6: Hip, shoulder moving down AND knees bending (should trigger Loading-Rising)
    for i in range(4, 7):
        frame = {'pose': base_pose.copy()}
        # Wrist still up from previous frames
        frame['pose']['left_wrist']['y'] -= 6
        frame['pose']['right_wrist']['y'] -= 6
        
        # Hip and shoulder moving down (loading condition)
        movement = (i - 3) * 8  # Significant downward movement
        frame['pose']['left_hip']['y'] += movement
        frame['pose']['right_hip']['y'] += movement
        frame['pose']['left_shoulder']['y'] += movement
        frame['pose']['right_shoulder']['y'] += movement
        
        # Knees bending (angles decreasing)
        knee_bend = (i - 3) * 10  # Knees moving closer together (angle decreasing)
        frame['pose']['left_knee']['y'] += movement - knee_bend  # Knee moves up relative to hip
        frame['pose']['right_knee']['y'] += movement - knee_bend
        test_data.append(frame)
    
    # Frame 7-9: Continue in Loading-Rising phase
    for i in range(7, 10):
        frame = {'pose': base_pose.copy()}
        # Maintain positions
        frame['pose']['left_wrist']['y'] -= 6
        frame['pose']['right_wrist']['y'] -= 6
        frame['pose']['left_hip']['y'] += 24  # Stable at down position
        frame['pose']['right_hip']['y'] += 24
        frame['pose']['left_shoulder']['y'] += 24
        frame['pose']['right_shoulder']['y'] += 24
        # Maintain knee bent position
        frame['pose']['left_knee']['y'] += 24 - 30
        frame['pose']['right_knee']['y'] += 24 - 30
        test_data.append(frame)
    
    return test_data


def create_test_ball_data() -> List[Dict]:
    """Create test ball data"""
    test_data = []
    
    # Ball held in hand for most frames
    for i in range(10):
        ball_frame = {
            'ball_detections': [{
                'center_x': 290,  # Near right wrist
                'center_y': 250 - i * 0.5,  # Slightly moving up
                'width': 30,
                'height': 30,
                'confidence': 0.9
            }]
        }
        test_data.append(ball_frame)
    
    return test_data


def test_rising_to_loading_rising_transition():
    """Test Rising → Loading-Rising transition"""
    print("🧪 Testing Rising → Loading-Rising Transition")
    print("=" * 50)
    
    # Create test data
    pose_data = create_test_pose_data()
    ball_data = create_test_ball_data()
    
    # Initialize detector
    detector = HybridFPSPhaseDetector()
    detector.set_fps(30.0)
    
    # Test phase transitions
    current_phase = "Rising"  # Start in Rising phase
    print(f"Initial phase: {current_phase}")
    
    for frame_idx in range(1, len(pose_data)):
        next_phase = detector.check_phase_transition(
            current_phase=current_phase,
            frame_idx=frame_idx,
            pose_data=pose_data,
            ball_data=ball_data,
            selected_hand="right"
        )
        
        if next_phase != current_phase:
            print(f"Frame {frame_idx}: {current_phase} → {next_phase}")
            current_phase = next_phase
        else:
            print(f"Frame {frame_idx}: {current_phase} (no change)")
    
    print(f"\nFinal phase: {current_phase}")
    
    # Check if transition occurred
    expected_transitions = ["Rising", "Loading-Rising"]
    if current_phase == "Loading-Rising":
        print("✅ SUCCESS: Rising → Loading-Rising transition detected!")
        return True
    else:
        print("❌ FAILED: Expected transition to Loading-Rising not detected")
        return False


def test_loading_rising_cancellation():
    """Test Loading-Rising cancellation conditions"""
    print("\n🧪 Testing Loading-Rising Cancellation")
    print("=" * 50)
    
    # Create test data with cancellation scenario
    pose_data = create_test_pose_data()
    ball_data = create_test_ball_data()
    
    # Add frames that should trigger cancellation
    base_pose = pose_data[0]['pose'].copy()
    
    # Frame 10-12: Shoulder/hip rising (should trigger loading cancellation → Rising)
    for i in range(3):
        frame = {'pose': base_pose.copy()}
        # Wrist up
        frame['pose']['left_wrist']['y'] -= 6
        frame['pose']['right_wrist']['y'] -= 6
        
        # Hip and shoulder rising (cancellation condition)
        rise_movement = i * 15  # Significant upward movement
        frame['pose']['left_hip']['y'] -= rise_movement
        frame['pose']['right_hip']['y'] -= rise_movement
        frame['pose']['left_shoulder']['y'] -= rise_movement
        frame['pose']['right_shoulder']['y'] -= rise_movement
        
        pose_data.append(frame)
        ball_data.append(ball_data[-1])  # Same ball data
    
    # Initialize detector
    detector = HybridFPSPhaseDetector()
    detector.set_fps(30.0)
    
    # Start in Loading-Rising phase
    current_phase = "Loading-Rising"
    print(f"Initial phase: {current_phase}")
    
    # Test cancellation
    for frame_idx in range(10, len(pose_data)):
        next_phase = detector.check_phase_transition(
            current_phase=current_phase,
            frame_idx=frame_idx,
            pose_data=pose_data,
            ball_data=ball_data,
            selected_hand="right"
        )
        
        if next_phase != current_phase:
            print(f"Frame {frame_idx}: {current_phase} → {next_phase}")
            current_phase = next_phase
            break
        else:
            print(f"Frame {frame_idx}: {current_phase} (no change)")
    
    print(f"\nFinal phase: {current_phase}")
    
    # Check if cancellation occurred
    if current_phase == "Rising":
        print("✅ SUCCESS: Loading-Rising → Rising cancellation detected!")
        return True
    else:
        print("❌ FAILED: Expected cancellation to Rising not detected")
        return False


def main():
    """Main test function"""
    print("🏀 Rising → Loading-Rising Transition Test")
    print("=" * 60)
    
    try:
        # Test 1: Rising → Loading-Rising transition
        test1_success = test_rising_to_loading_rising_transition()
        
        # Test 2: Loading-Rising cancellation
        test2_success = test_loading_rising_cancellation()
        
        # Overall result
        print("\n" + "=" * 60)
        if test1_success and test2_success:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Rising → Loading-Rising transition works correctly")
            print("✅ Loading-Rising cancellation works correctly")
        else:
            print("❌ SOME TESTS FAILED!")
            if not test1_success:
                print("❌ Rising → Loading-Rising transition failed")
            if not test2_success:
                print("❌ Loading-Rising cancellation failed")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()