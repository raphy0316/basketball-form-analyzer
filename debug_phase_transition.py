#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to check what's happening in the actual phase transition process
"""

import json
import os
from basketball_shooting_analyzer import BasketballShootingAnalyzer

def debug_phase_transition():
    """Debug the actual phase transition process"""
    
    print("🔍 DEBUGGING PHASE TRANSITION PROCESS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = BasketballShootingAnalyzer()
    
    # Load real data
    ball_file = "data/extracted_data/stephen_curry_part_ball_normalized.json"
    pose_file = "data/extracted_data/stephen_curry_part_pose_normalized.json"
    
    with open(ball_file, 'r', encoding='utf-8') as f:
        ball_data = json.load(f)
    
    with open(pose_file, 'r', encoding='utf-8') as f:
        pose_data = json.load(f)
    
    # Convert to analyzer format (exactly as the analyzer expects)
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
    
    print(f"📊 Data prepared:")
    print(f"  Total frames: {len(normalized_data)}")
    print(f"  Initial phases: {analyzer.phases[:5]}...")
    print()
    
    # Test the _check_phase_transition function directly
    print("🧪 Testing _check_phase_transition function directly:")
    print("-" * 40)
    
    for i in range(min(10, len(normalized_data))):
        frame_data = normalized_data[i]
        pose = frame_data["normalized_pose"]
        ball_info = frame_data["normalized_ball"]
        
        # Extract data exactly as the function does
        ball_x = ball_info.get('center_x', 0)
        ball_y = ball_info.get('center_y', 0)
        ball_detected = ball_info.get('detected', False)
        ball_radius = ball_info.get('radius', 0.1)
        
        left_wrist = pose.get('left_wrist', {'x': 0, 'y': 0})
        right_wrist = pose.get('right_wrist', {'x': 0, 'y': 0})
        wrist_y = (left_wrist.get('y', 0) + right_wrist.get('y', 0)) / 2
        
        distance = abs(ball_y - wrist_y)
        
        # Calculate thresholds
        close_threshold = ball_radius * 2.0
        medium_threshold = ball_radius * 4.0
        far_threshold = ball_radius * 6.0
        
        print(f"Frame {i}:")
        print(f"  Ball: detected={ball_detected}, y={ball_y:.3f}, radius={ball_radius:.3f}")
        print(f"  Wrist: y={wrist_y:.3f}")
        print(f"  Distance: {distance:.3f}")
        print(f"  Thresholds: close={close_threshold:.3f}, medium={medium_threshold:.3f}, far={far_threshold:.3f}")
        
        # Test transition logic
        if ball_detected:
            if distance < close_threshold:
                print(f"  ✅ WOULD TRANSITION: Close contact")
            elif distance < medium_threshold:
                print(f"  ✅ WOULD TRANSITION: Medium contact")
            elif distance < far_threshold:
                print(f"  ✅ WOULD TRANSITION: Far contact")
            else:
                print(f"  ❌ NO TRANSITION: Distance too far")
        else:
            print(f"  ❌ NO TRANSITION: Ball not detected")
        
        print()
    
    # Now test the actual segment_shooting_phases function
    print("🧪 Testing segment_shooting_phases function:")
    print("-" * 40)
    
    # Clear phases and run the actual function
    analyzer.phases = []
    analyzer.segment_shooting_phases()
    
    print(f"\n📊 Results from segment_shooting_phases:")
    print(f"  Total phases: {len(analyzer.phases)}")
    
    # Check first 20 frames
    for i in range(min(20, len(analyzer.phases))):
        print(f"  Frame {i}: {analyzer.phases[i]}")
    
    # Count phases
    phase_counts = {}
    for phase in analyzer.phases:
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    print(f"\n📈 Phase distribution:")
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count} frames")

if __name__ == "__main__":
    debug_phase_transition() 