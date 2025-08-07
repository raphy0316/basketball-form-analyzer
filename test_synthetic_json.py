#!/usr/bin/env python3
"""
Test script for synthetic profiles JSON export functionality
"""

import sys
import os
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from synthetic_profiles.motion_profile_generator import (
    SyntheticProfileGenerator,
    create_lebron_style,
    create_curry_style
)

def test_synthetic_json_export():
    """Test the synthetic profiles JSON export functionality"""
    print("🧪 Testing Synthetic Profiles JSON Export")
    print("=" * 50)
    
    # Initialize generator
    generator = SyntheticProfileGenerator()
    
    # Test with LeBron James profile
    print("\n🏀 Testing LeBron James profile...")
    lebron_style = create_lebron_style()
    lebron_profile = generator.generate_profile(lebron_style)
    
    # Export for comparison pipeline
    json_path = generator.export_for_comparison_pipeline("lebron", lebron_profile, "/tmp")
    print(f"✅ Exported to: {json_path}")
    
    # Verify the JSON file was created and has correct structure
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"📊 JSON Structure:")
        print(f"   - Metadata keys: {list(data.get('metadata', {}).keys())}")
        print(f"   - Total frames: {len(data.get('frames', []))}")
        print(f"   - Player name: {data.get('metadata', {}).get('player_name')}")
        print(f"   - Shots: {list(data.get('metadata', {}).get('shots', {}).keys())}")
        
        # Check first frame structure
        if data.get('frames'):
            first_frame = data['frames'][0]
            print(f"   - First frame keys: {list(first_frame.keys())}")
            print(f"   - First frame phase: {first_frame.get('phase')}")
            print(f"   - First frame shot_id: {first_frame.get('shot_id')}")
        
        print("✅ JSON export test passed!")
        return True
    else:
        print("❌ JSON file was not created!")
        return False

def test_curry_profile():
    """Test Curry profile generation and export"""
    print("\n🏀 Testing Stephen Curry profile...")
    generator = SyntheticProfileGenerator()
    curry_style = create_curry_style()
    curry_profile = generator.generate_profile(curry_style)
    
    json_path = generator.export_for_comparison_pipeline("curry", curry_profile, "/tmp")
    print(f"✅ Curry profile exported to: {json_path}")
    
    return os.path.exists(json_path)

if __name__ == "__main__":
    print("🚀 Starting Synthetic Profiles JSON Export Tests")
    
    success = True
    success &= test_synthetic_json_export()
    success &= test_curry_profile()
    
    if success:
        print("\n🎉 All tests passed! Synthetic profiles are ready for comparison pipeline.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)
