#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for detailed shooting form analysis
Demonstrates the new biomechanical analysis features for loading and rising phases
"""

import os
import sys
from basketball_shooting_analyzer import BasketballShootingAnalyzer, DetailedShootingFormAnalyzer

def test_detailed_analysis():
    """Test the detailed shooting form analysis functionality"""
    print("🏀 Testing Detailed Shooting Form Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = BasketballShootingAnalyzer()
    
    # Check for available data
    if not os.path.exists(analyzer.extracted_data_dir):
        print("❌ No extracted data directory found.")
        print("   Please run the main analysis pipeline first to extract pose and ball data.")
        return False
    
    # Look for existing normalized data
    normalized_files = []
    for file in os.listdir(analyzer.extracted_data_dir):
        if file.endswith('_pose_normalized.json'):
            normalized_files.append(file)
    
    if not normalized_files:
        print("❌ No normalized data files found.")
        print("   Please run the main analysis pipeline first.")
        return False
    
    print(f"📁 Found {len(normalized_files)} normalized data files:")
    for i, file in enumerate(normalized_files, 1):
        print(f"   {i}. {file}")
    
    # Select a file to analyze
    if len(normalized_files) == 1:
        selected_file = normalized_files[0]
    else:
        try:
            choice = int(input(f"\nSelect file to analyze (1-{len(normalized_files)}): ")) - 1
            if 0 <= choice < len(normalized_files):
                selected_file = normalized_files[choice]
            else:
                print("❌ Invalid selection.")
                return False
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input.")
            return False
    
    print(f"\n🔍 Analyzing: {selected_file}")
    
    # Load the normalized data
    file_path = os.path.join(analyzer.extracted_data_dir, selected_file)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            import json
            data = json.load(f)
            analyzer.normalized_data = data.get('normalized_pose_data', [])
    except Exception as e:
        print(f"❌ Failed to load normalized data: {e}")
        return False
    
    if not analyzer.normalized_data:
        print("❌ No normalized data found in file.")
        return False
    
    print(f"✅ Loaded {len(analyzer.normalized_data)} frames of normalized data")
    
    # Perform phase segmentation
    print("\n📐 Performing phase segmentation...")
    analyzer.segment_shooting_phases()
    
    if not analyzer.phases:
        print("❌ No phases detected.")
        return False
    
    print(f"✅ Detected {len(set(analyzer.phases))} phases: {list(set(analyzer.phases))}")
    
    # Perform detailed analysis
    print("\n🔬 Performing detailed form analysis...")
    analyzer.perform_detailed_form_analysis()
    
    # Display results summary
    print("\n📊 Detailed Analysis Results Summary:")
    print("=" * 40)
    
    if "loading_phase" in analyzer.detailed_analysis_results:
        loading = analyzer.detailed_analysis_results["loading_phase"]
        print("\n🎯 Loading Phase Analysis (Dip):")
        print("  " + "="*35)
        print(f"  • Dip Frame: {loading.get('dip_frame', 'N/A')}")
        
        # Stance asymmetry
        stance = loading.get('stance_asymmetry', {})
        print(f"  • Stance Alignment: {stance.get('stance_alignment', 'N/A')}")
        print(f"  • Horizontal Distance: {stance.get('horizontal_distance', 0):.3f}")
        print(f"  • Stance Level: {stance.get('stance_level', 'N/A')}")
        
        # Ball-hip analysis
        print(f"  • Ball-Hip Vertical Distance: {loading.get('ball_hip_vertical_distance', 0):.3f}")
        
        # Joint angles
        joint_angles = loading.get('joint_angles', {})
        hip_knee_ankle = joint_angles.get('hip_knee_ankle', {})
        shoulder_hip_knee = joint_angles.get('shoulder_hip_knee', {})
        print(f"  • Hip-Knee-Ankle Angle: {hip_knee_ankle.get('average_angle', 0):.1f}°")
        print(f"  • Shoulder-Hip-Knee Angle: {shoulder_hip_knee.get('average_angle', 0):.1f}°")
        print(f"  • Hip-Shoulder Line Angle: {joint_angles.get('hip_shoulder_line_angle', 0):.1f}°")
        
        # Hip compression
        hip_compression = loading.get('hip_height_compression', {})
        print(f"  • Hip Compression Depth: {hip_compression.get('compression_depth', 0):.3f}")
        print(f"  • Compression Percentage: {hip_compression.get('compression_percentage', 0):.1f}%")
        
        # Knee symmetry
        knee_symmetry = loading.get('knee_angle_symmetry', {})
        print(f"  • Knee Symmetry Score: {knee_symmetry.get('symmetry_score', 0):.3f}")
        print(f"  • Symmetry Level: {knee_symmetry.get('symmetry_level', 'N/A')}")
        
        # Feet placement
        feet = loading.get('feet_placement', {})
        print(f"  • Stance Width: {feet.get('stance_width', 0):.3f}")
        print(f"  • Stance Type: {feet.get('stance_type', 'N/A')}")
        
        # Timing
        print(f"  • Loading Duration: {loading.get('loading_phase_duration', 0)} frames")
        print(f"  • Time in Dip: {loading.get('time_in_dip', 0)} frames")
        print(f"  • Dip Depth: {loading.get('dip_depth', 0):.3f}")
    
    if "rising_phase" in analyzer.detailed_analysis_results:
        rising = analyzer.detailed_analysis_results["rising_phase"]
        print("\n🚀 Rising Phase Analysis (Windup & Setpoint):")
        print("  " + "="*40)
        print(f"  • Setpoint Frame: {rising.get('setpoint_frame', 'N/A')}")
        print(f"  • Dip to Setpoint Duration: {rising.get('dip_to_setpoint_duration', 0)} frames")
        
        # Windup analysis
        windup = rising.get('windup_analysis', {})
        print(f"  • Windup Duration: {windup.get('windup_duration', 0)} frames")
        
        # Ball trajectory analysis
        trajectory = windup.get('ball_trajectory', {})
        print(f"  • Ball Trajectory Distance: {trajectory.get('total_distance', 0):.3f}")
        print(f"  • Ball Trajectory Smoothness: {trajectory.get('trajectory_smoothness', 0):.3f}")
        print(f"  • Vertical Movement: {trajectory.get('vertical_movement', 0):.3f}")
        print(f"  • Horizontal Movement: {trajectory.get('horizontal_movement', 0):.3f}")
        
        # Velocity profile
        velocity = windup.get('velocity_profile', {})
        print(f"  • Average Velocity: {velocity.get('average_velocity', 0):.3f}")
        print(f"  • Max Velocity: {velocity.get('max_velocity', 0):.3f}")
        print(f"  • Velocity Consistency: {velocity.get('velocity_consistency', 0):.3f}")
        
        # Joint dynamics
        joint_dynamics = windup.get('joint_dynamics', {})
        elbow_tuck = joint_dynamics.get('elbow_tuck_analysis', {})
        print(f"  • Elbow Movement: {elbow_tuck.get('elbow_movement', 'N/A')}")
        print(f"  • Elbow Flare Magnitude: {elbow_tuck.get('flare_magnitude', 0):.3f}")
        
        # Center of mass
        com = windup.get('center_of_mass', {})
        print(f"  • COM Movement Direction: {com.get('movement_direction', 'N/A')}")
        print(f"  • COM Vertical Direction: {com.get('vertical_direction', 'N/A')}")
        
        # Ankle extension
        ankle = windup.get('ankle_extension', {})
        print(f"  • Ankle Extension Pattern: {ankle.get('extension_pattern', 'N/A')}")
        print(f"  • Max Ankle Extension: {ankle.get('max_extension', 0):.3f}")
        
        # Synchronization
        sync = windup.get('synchronization', {})
        print(f"  • Ball-Leg Synchronization: {sync.get('synchronization_quality', 'N/A')}")
        print(f"  • Synchronization Correlation: {sync.get('correlation', 0):.3f}")
        
        # Setpoint analysis
        setpoint = rising.get('setpoint_analysis', {})
        
        # Key angles at setpoint
        key_angles = setpoint.get('key_angles', {})
        shoulder_elbow_wrist = key_angles.get('shoulder_elbow_wrist', {})
        elbow_shoulder_hip = key_angles.get('elbow_shoulder_hip', {})
        hip_knee_ankle = key_angles.get('hip_knee_ankle', {})
        print(f"  • Setpoint Shoulder-Elbow-Wrist: {shoulder_elbow_wrist.get('average_angle', 0):.1f}°")
        print(f"  • Setpoint Elbow-Shoulder-Hip: {elbow_shoulder_hip.get('average_angle', 0):.1f}°")
        print(f"  • Setpoint Hip-Knee-Ankle: {hip_knee_ankle.get('average_angle', 0):.1f}°")
        print(f"  • Setpoint Hip-Shoulder Line: {key_angles.get('hip_shoulder_line', 0):.1f}°")
        
        # Ball-eye distances
        ball_eye = setpoint.get('ball_eye_distances', {})
        print(f"  • Ball-Eye Horizontal Distance: {ball_eye.get('horizontal', 0):.3f}")
        print(f"  • Ball-Eye Vertical Distance: {ball_eye.get('vertical', 0):.3f}")
        print(f"  • Ball-Eye Total Distance: {ball_eye.get('total_distance', 0):.3f}")
        
        # Shooting pocket positioning
        pocket = setpoint.get('shooting_pocket_positioning', {})
        print(f"  • Ball in Line with Eye: {pocket.get('in_line_with_eye', False)}")
        print(f"  • Ball Centered: {pocket.get('centered', False)}")
        print(f"  • Positioning Quality: {pocket.get('positioning_quality', 'N/A')}")
        
        # Stability check
        stability = setpoint.get('stability_check', {})
        print(f"  • Shoulders Squared: {stability.get('shoulders_squared', False)}")
        print(f"  • Hips Squared: {stability.get('hips_squared', False)}")
        print(f"  • Knees Locked: {stability.get('knees_locked', False)}")
        print(f"  • Stability Score: {stability.get('stability_score', 0):.3f}")
        
        # Stance analysis
        stance = setpoint.get('stance_comparison', {})
        print(f"  • Ankle Stance Width: {stance.get('ankle_width', 0):.3f}")
        print(f"  • Shoulder Width: {stance.get('shoulder_width', 0):.3f}")
        print(f"  • Width Ratio: {stance.get('width_ratio', 0):.2f}")
        print(f"  • Stance Type: {stance.get('stance_type', 'N/A')}")
        
        # Momentum direction
        momentum = setpoint.get('momentum_direction', {})
        print(f"  • Post-Setpoint Movement: {momentum.get('movement', 'N/A')}")
        print(f"  • Movement Magnitude: {momentum.get('movement_magnitude', 0):.3f}")
        
        # Head position
        head = setpoint.get('head_position', {})
        print(f"  • Gaze Estimate: {head.get('gaze_estimate', 'N/A')}")
    
    # Save results
    print("\n💾 Saving detailed analysis results...")
    base_name = os.path.splitext(selected_file)[0].replace('_pose_normalized', '')
    analyzer.save_detailed_analysis_results(f"test_{base_name}.mp4", overwrite_mode=True)
    
    print("\n✅ Detailed analysis test completed successfully!")
    return True

def main():
    """Main function"""
    try:
        success = test_detailed_analysis()
        if success:
            print("\n🎉 Test completed successfully!")
            print("Check the 'data/results/' directory for detailed analysis files.")
        else:
            print("\n❌ Test failed.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 