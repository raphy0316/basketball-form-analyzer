#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to check if General → Set-up phase transition is working
"""

import json
import os
from basketball_shooting_analyzer import BasketballShootingAnalyzer

def test_phase_transition():
    """Test the General → Set-up phase transition"""
    
    # Initialize analyzer
    analyzer = BasketballShootingAnalyzer()
    
    # Create test data
    test_data = []
    for i in range(100):
        frame_data = {
            "frame_index": i,
            "phase": "General",  # Start with General
            "normalized_pose": {
                "left_wrist": {"x": 0.5, "y": 0.6},
                "right_wrist": {"x": 0.5, "y": 0.6}
            },
            "normalized_ball": {
                "detected": True,
                "center_x": 0.5,
                "center_y": 0.65,  # Close to wrist
                "radius": 0.1
            }
        }
        test_data.append(frame_data)
    
    # Set test data
    analyzer.normalized_data = test_data
    analyzer.phases = ["General"] * 100
    
    print("🧪 Testing General → Set-up phase transition")
    print("=" * 50)
    
    # Test the transition logic
    for i in range(10):  # Test first 10 frames
        current_phase = analyzer.phases[i]
        
        # Get ball and wrist data
        ball_data = test_data[i]["normalized_ball"]
        pose_data = test_data[i]["normalized_pose"]
        
        ball_detected = ball_data.get("detected", False)
        ball_y = ball_data.get("center_y", 0)
        ball_radius = ball_data.get("radius", 0.1)
        wrist_y = (pose_data.get("left_wrist", {"y": 0})["y"] + 
                  pose_data.get("right_wrist", {"y": 0})["y"]) / 2
        
        distance = abs(ball_y - wrist_y)
        
        # Calculate thresholds
        close_threshold = ball_radius * 1.2
        medium_threshold = ball_radius * 2.0
        far_threshold = ball_radius * 3.0
        
        print(f"Frame {i}:")
        print(f"  Current phase: {current_phase}")
        print(f"  Ball detected: {ball_detected}")
        print(f"  Ball Y: {ball_y:.3f}")
        print(f"  Wrist Y: {wrist_y:.3f}")
        print(f"  Distance: {distance:.3f}")
        print(f"  Ball radius: {ball_radius:.3f}")
        print(f"  Thresholds: close={close_threshold:.3f}, medium={medium_threshold:.3f}, far={far_threshold:.3f}")
        
        # Check transition
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
    test_phase_transition() 