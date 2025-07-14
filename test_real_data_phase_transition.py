#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to check General → Set-up phase transition with real data
"""

import json
import os
from basketball_shooting_analyzer import BasketballShootingAnalyzer

def test_real_data_phase_transition():
    """Test the General → Set-up phase transition with real data"""
    
    # Load real data
    ball_file = "data/extracted_data/stephen_curry_part_ball_normalized.json"
    pose_file = "data/extracted_data/stephen_curry_part_pose_normalized.json"
    
    print("🧪 Testing General → Set-up phase transition with REAL DATA")
    print("=" * 60)
    
    # Load ball data
    with open(ball_file, 'r', encoding='utf-8') as f:
        ball_data = json.load(f)
    
    # Load pose data
    with open(pose_file, 'r', encoding='utf-8') as f:
        pose_data = json.load(f)
    
    print(f"📊 Data loaded:")
    print(f"  Ball data frames: {len(ball_data['normalized_ball_data'])}")
    print(f"  Pose data frames: {len(pose_data['normalized_pose_data'])}")
    print()
    
    # Test first 20 frames
    for i in range(min(20, len(ball_data['normalized_ball_data']))):
        ball_frame = ball_data['normalized_ball_data'][i]
        pose_frame = pose_data['normalized_pose_data'][i]
        
        # Extract ball info
        ball_center_x = ball_frame.get('center_x', 0)
        ball_center_y = ball_frame.get('center_y', 0)
        ball_width = ball_frame.get('width', 0)
        ball_height = ball_frame.get('height', 0)
        ball_radius = (ball_width + ball_height) / 4  # Approximate radius
        
        # Extract wrist info
        pose_info = pose_frame.get('normalized_pose', {})
        left_wrist = pose_info.get('left_wrist', {'x': 0, 'y': 0})
        right_wrist = pose_info.get('right_wrist', {'x': 0, 'y': 0})
        
        # Calculate average wrist position
        wrist_x = (left_wrist.get('x', 0) + right_wrist.get('x', 0)) / 2
        wrist_y = (left_wrist.get('y', 0) + right_wrist.get('y', 0)) / 2
        
        # Calculate distance
        distance = abs(ball_center_y - wrist_y)
        
        # Calculate thresholds
        close_threshold = ball_radius * 1.3
        medium_threshold = ball_radius * 2.0
        far_threshold = ball_radius * 3.0
        
        print(f"Frame {i}:")
        print(f"  Ball: center=({ball_center_x:.2f}, {ball_center_y:.2f}), radius={ball_radius:.3f}")
        print(f"  Wrist: ({wrist_x:.2f}, {wrist_y:.2f})")
        print(f"  Distance: {distance:.3f}")
        print(f"  Thresholds: close={close_threshold:.3f}, medium={medium_threshold:.3f}, far={far_threshold:.3f}")
        
        # Check transition
        if distance < close_threshold:
            print(f"  ✅ Would transition to Set-up (Close contact)")
        elif distance < medium_threshold:
            print(f"  ✅ Would transition to Set-up (Medium contact)")
        elif distance < far_threshold:
            print(f"  ✅ Would transition to Set-up (Far contact)")
        else:
            print(f"  ❌ No transition (Distance too far)")
        
        print()

def test_with_analyzer():
    """Test using the actual analyzer class"""
    
    print("\n🔧 Testing with BasketballShootingAnalyzer class")
    print("=" * 60)
    
    analyzer = BasketballShootingAnalyzer()
    
    # Load data into analyzer format
    ball_file = "data/extracted_data/stephen_curry_part_ball_normalized.json"
    pose_file = "data/extracted_data/stephen_curry_part_pose_normalized.json"
    
    with open(ball_file, 'r', encoding='utf-8') as f:
        ball_data = json.load(f)
    
    with open(pose_file, 'r', encoding='utf-8') as f:
        pose_data = json.load(f)
    
    # Convert to analyzer format
    normalized_data = []
    for i in range(len(pose_data['normalized_pose_data'])):
        pose_frame = pose_data['normalized_pose_data'][i]
        ball_frame = ball_data['normalized_ball_data'][i] if i < len(ball_data['normalized_ball_data']) else {}
        
        frame_data = {
            "frame_index": i,
            "phase": "General",  # Start with General
            "normalized_pose": pose_frame.get('normalized_pose', {}),
            "normalized_ball": {
                "detected": True,
                "center_x": ball_frame.get('center_x', 0),
                "center_y": ball_frame.get('center_y', 0),
                "radius": (ball_frame.get('width', 0) + ball_frame.get('height', 0)) / 4
            }
        }
        normalized_data.append(frame_data)
    
    analyzer.normalized_data = normalized_data
    analyzer.phases = ["General"] * len(normalized_data)
    
    # Test first 10 frames
    for i in range(min(10, len(normalized_data))):
        current_phase = analyzer.phases[i]
        frame_data = normalized_data[i]
        
        ball_data = frame_data["normalized_ball"]
        pose_data = frame_data["normalized_pose"]
        
        ball_detected = ball_data.get("detected", False)
        ball_y = ball_data.get("center_y", 0)
        ball_radius = ball_data.get("radius", 0.1)
        
        left_wrist = pose_data.get("left_wrist", {"y": 0})
        right_wrist = pose_data.get("right_wrist", {"y": 0})
        wrist_y = (left_wrist.get("y", 0) + right_wrist.get("y", 0)) / 2
        
        distance = abs(ball_y - wrist_y)
        
        close_threshold = ball_radius * 1.3
        medium_threshold = ball_radius * 2.0
        far_threshold = ball_radius * 3.0
        
        print(f"Frame {i}:")
        print(f"  Current phase: {current_phase}")
        print(f"  Ball detected: {ball_detected}")
        print(f"  Ball Y: {ball_y:.3f}, Wrist Y: {wrist_y:.3f}")
        print(f"  Distance: {distance:.3f}")
        print(f"  Ball radius: {ball_radius:.3f}")
        print(f"  Thresholds: close={close_threshold:.3f}, medium={medium_threshold:.3f}, far={far_threshold:.3f}")
        
        if current_phase == "General" and ball_detected:
            if distance < close_threshold:
                print(f"  ✅ Would transition to Set-up (Close contact)")
            elif distance < medium_threshold:
                print(f"  ✅ Would transition to Set-up (Medium contact)")
            elif distance < far_threshold:
                print(f"  ✅ Would transition to Set-up (Far contact)")
            else:
                print(f"  ❌ No transition (Distance too far)")
        else:
            print(f"  ❌ No transition (Phase not General or ball not detected)")
        
        print()

if __name__ == "__main__":
    test_real_data_phase_transition()
    test_with_analyzer() 