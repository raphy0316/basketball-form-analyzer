#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check phase distribution after analysis
"""

import json
import os
from basketball_shooting_analyzer import BasketballShootingAnalyzer

def check_phase_distribution():
    """Check what phases are actually generated"""
    
    print("🔍 CHECKING PHASE DISTRIBUTION")
    print("=" * 50)
    
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
                "center_x": ball_frame.get('center_x', 0),
                "center_y": ball_frame.get('center_y', 0),
                "width": ball_frame.get('width', 0),
                "height": ball_frame.get('height', 0)
            },
            "ball_detected": True  # Add this field
        }
        normalized_data.append(frame_data)
    
    analyzer.normalized_data = normalized_data
    analyzer.phases = []
    
    print(f"📊 Data prepared:")
    print(f"  Total frames: {len(normalized_data)}")
    print()
    
    # Run the actual analysis
    print("🧪 Running segment_shooting_phases...")
    analyzer.segment_shooting_phases()
    
    print(f"\n📊 Results:")
    print(f"  Total phases: {len(analyzer.phases)}")
    
    # Check first 30 frames
    print(f"\n📋 First 30 frames:")
    for i in range(min(30, len(analyzer.phases))):
        print(f"  Frame {i}: {analyzer.phases[i]}")
    
    # Count phases
    phase_counts = {}
    for phase in analyzer.phases:
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    print(f"\n📈 Phase distribution:")
    for phase, count in phase_counts.items():
        percentage = (count / len(analyzer.phases)) * 100
        print(f"  {phase}: {count} frames ({percentage:.1f}%)")
    
    # Check if General phase exists
    if "General" in phase_counts:
        print(f"\n✅ General phase found: {phase_counts['General']} frames")
    else:
        print(f"\n❌ General phase NOT found!")
    
    # Check transition points
    print(f"\n🔄 Phase transitions:")
    prev_phase = None
    transition_count = 0
    for i, phase in enumerate(analyzer.phases):
        if prev_phase and phase != prev_phase:
            print(f"  Frame {i}: {prev_phase} → {phase}")
            transition_count += 1
        prev_phase = phase
    
    print(f"  Total transitions: {transition_count}")

if __name__ == "__main__":
    check_phase_distribution() 