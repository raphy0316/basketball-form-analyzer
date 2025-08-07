"""
Shot Detector

Real-time shot detection and lifecycle management for basketball shooting analysis.
Manages shot start/end/cancel transitions and provides shot-specific torso measurements.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class ShotDetector:
    """
    Real-time shot detector that manages shot lifecycle and torso measurements.
    
    Provides shot state information and fixed torso measurements for consistent
    phase detection during shot execution.
    """
    
    def __init__(self):
        """Initialize shot detector with default state."""
        # Shot state management
        self.is_shot_active = False
        self.current_shot_id = 0
        self.current_shot_start = None
        self.current_shot_end = None
        
        # Torso management
        self.current_shot_fixed_torso = None
        self.torso_tracking_active = True  # Whether to update rolling torso
        self.rolling_torso_values = []  # Last 4 torso measurements
        self.rolling_torso_frames = []  # Corresponding frame indices
        
        # Shot collection (same format as analyzer's output)
        self.shots = []  # List of completed shots with metadata
        self.frame_shots = []  # Shot assignment for each frame
        
        # Previous phase tracking for transition detection
        self.previous_phase = "General"
        
        # Phase tracking for current shot
        self.current_shot_phases = []  # Track all phases in current shot
    
    def get_shot_torso(self, pose: Dict = None) -> float:
        """
        Get torso length for current frame.
        
        Args:
            pose: Current pose data (for rolling torso calculation)
            
        Returns:
            Fixed torso for active shot, or rolling torso otherwise
        """
        if self.is_shot_active and self.current_shot_fixed_torso is not None:
            # Return fixed torso for active shot
            return self.current_shot_fixed_torso
        elif len(self.rolling_torso_values) >= 4:
            # Return rolling average
            return np.mean(self.rolling_torso_values[-4:])
        elif len(self.rolling_torso_values) > 0:
            # Return available average
            return np.mean(self.rolling_torso_values)
        else:
            # Fallback calculation from current pose
            if pose:
                return self._calculate_torso_from_pose(pose)
            return 0.0
    
    def update_rolling_torso(self, frame_idx: int, pose: Dict):
        """
        Update rolling torso measurement.
        
        Args:
            frame_idx: Current frame index
            pose: Current pose data
        """
        if not self.torso_tracking_active:
            return  # Skip if tracking is paused
        
        torso_length = self._calculate_torso_from_pose(pose)
        if torso_length > 0:
            self.rolling_torso_values.append(torso_length)
            self.rolling_torso_frames.append(frame_idx)
            
            # Keep only last 4 measurements
            if len(self.rolling_torso_values) > 4:
                self.rolling_torso_values.pop(0)
                self.rolling_torso_frames.pop(0)
    
    def detect_shot_transitions(self, frame_idx: int, current_phase: str) -> bool:
        """
        Detect shot transitions and manage shot state.
        
        Args:
            frame_idx: Current frame index
            current_phase: Current phase detected
            
        Returns:
            True if shot state changed, False otherwise
        """
        prev_phase = self.previous_phase
        state_changed = False
        
        # Track current shot phases
        if self.is_shot_active:
            if not self.current_shot_phases or self.current_shot_phases[-1] != current_phase:
                self.current_shot_phases.append(current_phase)
        
        # 1. Shot start: General → Set-up (새로운 shot 시작)
        if (prev_phase == "General" and current_phase == "Set-up"):
            if self.is_shot_active:
                print(f"   ⚠️ FORCE CANCEL: New shot starting while shot {self.current_shot_id} is active at frame {frame_idx}")
                self._cancel_current_shot(frame_idx, reason="Forced by new shot start")
            
            self._start_new_shot(frame_idx)
            print(f"   🔄 TRANSITION: {prev_phase} → {current_phase} at frame {frame_idx} (Shot active: {self.is_shot_active})")
            print(f"   🏀 SHOT START: General → Set-up transition at frame {frame_idx}")
            state_changed = True
        
        # 2. Shot cancel: Various phases → General (shot 감지 취소)
        elif (self.is_shot_active and current_phase == "General" and 
              prev_phase in ["Set-up", "Loading", "Rising", "Loading-Rising", "Release"]):
            print(f"   🔄 TRANSITION: {prev_phase} → {current_phase} at frame {frame_idx} (Shot active: {self.is_shot_active})")
            print(f"   ❌ SHOT CANCEL: {prev_phase} → General transition at frame {frame_idx}")
            self._cancel_current_shot(frame_idx, reason=f"{prev_phase} → General (abnormal return)")
            state_changed = True
        
        # 3. Shot cancel: Backward transitions (Rising/Loading-Rising → Set-up)
        # Cancel current shot and immediately start new shot
        elif (self.is_shot_active and prev_phase in ["Rising", "Loading-Rising"] and current_phase == "Set-up"):
            print(f"   🔄 TRANSITION: {prev_phase} → {current_phase} at frame {frame_idx} (Shot active: {self.is_shot_active})")
            print(f"   ❌ SHOT CANCEL: {prev_phase} → Set-up backward transition at frame {frame_idx}")
            self._cancel_current_shot(frame_idx, reason=f"{prev_phase} → Set-up (backward motion)")
            
            # Immediately start new shot since we're already in Set-up
            print(f"   🔄 IMMEDIATE RESTART: Starting new shot at Set-up phase")
            self._start_new_shot(frame_idx)
            state_changed = True
            
        # 4. Shot cancel: Loading backward transitions (Loading → Set-up)
        # Cancel current shot and immediately start new shot  
        elif (self.is_shot_active and prev_phase == "Loading" and current_phase == "Set-up"):
            print(f"   🔄 TRANSITION: {prev_phase} → {current_phase} at frame {frame_idx} (Shot active: {self.is_shot_active})")
            print(f"   ❌ SHOT CANCEL: Loading → Set-up backward transition at frame {frame_idx}")
            self._cancel_current_shot(frame_idx, reason="Loading → Set-up (backward motion)")
            
            # Immediately start new shot since we're already in Set-up
            print(f"   🔄 IMMEDIATE RESTART: Starting new shot at Set-up phase")
            self._start_new_shot(frame_idx)
            state_changed = True
        
        # 5. Shot end: Follow-through → General (shot 완료)
        elif (prev_phase == "Follow-through" and current_phase == "General" and self.is_shot_active):
            print(f"   🔄 TRANSITION: {prev_phase} → {current_phase} at frame {frame_idx} (Shot active: {self.is_shot_active})")
            print(f"   🎯 SHOT COMPLETE: Follow-through → General transition at frame {frame_idx}")
            self._complete_current_shot(frame_idx)
            state_changed = True
        
        # 6. Meaningful transition within shot: Set-up → Loading/Rising (fix torso)
        elif (self.is_shot_active and prev_phase == "Set-up" and 
              current_phase in ["Loading", "Rising", "Loading-Rising"] and self.torso_tracking_active):
            print(f"   🔄 TRANSITION: {prev_phase} → {current_phase} at frame {frame_idx} (Shot active: {self.is_shot_active})")
            print(f"   🔒 TORSO FIX: Set-up → {current_phase} meaningful transition at frame {frame_idx}")
            self._fix_shot_torso(frame_idx)
            state_changed = True
        
        # 7. Regular transitions (no state change)
        elif prev_phase != current_phase:
            if self.is_shot_active:
                print(f"   🔄 TRANSITION: {prev_phase} → {current_phase} at frame {frame_idx} (Shot active: {self.is_shot_active})")
            else:
                # Only log non-shot transitions occasionally to reduce noise
                if frame_idx % 60 == 0:  # Every 2 seconds
                    print(f"   🔄 TRANSITION: {prev_phase} → {current_phase} at frame {frame_idx} (Shot active: {self.is_shot_active})")
        
        # Update previous phase
        self.previous_phase = current_phase
        
        # Periodic status updates
        if frame_idx % 30 == 0:  # Every second
            if self.is_shot_active:
                phases_summary = " → ".join(set(self.current_shot_phases)) if self.current_shot_phases else "None"
                print(f"   📊 Frame {frame_idx}: Shot {self.current_shot_id} active (phases: {phases_summary}), Torso: {'Fixed' if not self.torso_tracking_active else 'Tracking'}")
            else:
                print(f"   📊 Frame {frame_idx}: No active shot, waiting for General → Set-up")
        
        return state_changed
    
    def _start_new_shot(self, frame_idx: int):
        """Start a new shot and initialize tracking."""
        self.current_shot_id += 1
        self.current_shot_start = frame_idx
        self.current_shot_end = None
        self.is_shot_active = True
        self.torso_tracking_active = True  # Keep tracking until meaningful transition
        self.current_shot_phases = ["Set-up"]  # Initialize with Set-up phase
        
        print(f"\n🟢 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🏀 SHOT {self.current_shot_id} STARTED at frame {frame_idx}")
        print(f"   📍 Start frame: {frame_idx}")
        print(f"   🎯 State: Shot Active = True")
        print(f"   📏 Torso: Tracking Active (waiting for meaningful transition)")
        print(f"   🔄 Initial phase: Set-up")
        print(f"🟢 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    def _cancel_current_shot(self, frame_idx: int, reason: str = "Unknown"):
        """Cancel current shot and reset state."""
        if self.is_shot_active:
            # Capture shot summary before reset
            shot_duration = frame_idx - self.current_shot_start
            phases_sequence = " → ".join(self.current_shot_phases) if self.current_shot_phases else "None"
            torso_status = f"Fixed ({self.current_shot_fixed_torso:.4f})" if self.current_shot_fixed_torso else "Not Fixed"
            
            print(f"\n🔴 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"❌ SHOT {self.current_shot_id} CANCELLED at frame {frame_idx}")
            print(f"   📍 Frame range: {self.current_shot_start} → {frame_idx} ({shot_duration} frames)")
            print(f"   🔄 Phase sequence: {phases_sequence}")
            print(f"   📏 Torso status: {torso_status}")
            print(f"   💭 Cancellation reason: {reason}")
            print(f"   🎯 State change: Shot Active = False")
            print(f"🔴 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            # Reset shot state
            self.is_shot_active = False
            self.current_shot_start = None
            self.current_shot_end = None
            self.current_shot_fixed_torso = None
            self.torso_tracking_active = True  # Resume rolling torso tracking
            self.rolling_torso_values = []  # Clear rolling window for fresh start
            self.rolling_torso_frames = []
            self.current_shot_phases = []  # Clear phase tracking
            
            print(f"   🔄 Waiting for next General → Set-up to start new shot...")
    
    def _fix_shot_torso(self, frame_idx: int):
        """Fix torso for current shot when meaningful transition is found."""
        if not self.is_shot_active:
            return
        
        # Use available torso data from rolling window
        if len(self.rolling_torso_values) > 0:
            self.current_shot_fixed_torso = np.mean(self.rolling_torso_values)
            self.torso_tracking_active = False  # Stop updating rolling torso
            
            print(f"\n🔒 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"📏 TORSO FIXED for Shot {self.current_shot_id} at frame {frame_idx}")
            print(f"   🎯 Fixed torso value: {self.current_shot_fixed_torso:.4f}")
            print(f"   📊 Based on {len(self.rolling_torso_values)} rolling measurements")
            print(f"   ⏸️ Rolling torso tracking: PAUSED")
            print(f"   🔄 Shot will use this fixed torso for all remaining frames")
            print(f"🔒 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        else:
            print(f"   ⚠️ WARNING: No rolling torso data available, keeping tracking active")
    
    def _complete_current_shot(self, frame_idx: int):
        """Complete current shot and add to shots list."""
        if self.is_shot_active and self.current_shot_start is not None:
            self.current_shot_end = frame_idx
            
            # Use sequential numbering for completed shots (1, 2, 3...)
            sequential_shot_id = len(self.shots) + 1
            
            # Capture shot summary
            shot_duration = frame_idx - self.current_shot_start + 1
            phases_sequence = " → ".join(self.current_shot_phases) if self.current_shot_phases else "None"
            torso_status = f"{self.current_shot_fixed_torso:.4f}" if self.current_shot_fixed_torso else "Not Fixed"
            
            # Create shot info (same format as analyzer output)
            shot_info = {
                "shot_id": sequential_shot_id,  # Sequential numbering for pipeline compatibility
                "original_shot_id": self.current_shot_id,  # Preserve original ID for debugging
                "start_frame": self.current_shot_start,
                "end_frame": frame_idx,
                "total_frames": shot_duration,
                "fixed_torso": self.current_shot_fixed_torso
            }
            self.shots.append(shot_info)
            
            print(f"\n🟢 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"🎯 SHOT {self.current_shot_id} COMPLETED at frame {frame_idx}")
            print(f"   📍 Frame range: {self.current_shot_start} → {frame_idx} ({shot_duration} frames)")
            print(f"   🔄 Full phase sequence: {phases_sequence}")
            print(f"   📏 Fixed torso: {torso_status}")
            print(f"   💾 Saved as Shot {sequential_shot_id} (sequential numbering)")
            print(f"   🎯 State change: Shot Active = False")
            print(f"🟢 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            # Reset for next shot
            self.is_shot_active = False
            self.current_shot_start = None
            self.current_shot_end = None
            self.current_shot_fixed_torso = None
            self.torso_tracking_active = True  # Resume rolling torso tracking
            self.rolling_torso_values = []  # Clear rolling window for fresh start
            self.rolling_torso_frames = []
            self.current_shot_phases = []  # Clear phase tracking
    
    def finalize_frame_shots(self, total_frames: int):
        """
        Assign shot information to each frame (call after processing all frames).
        
        Args:
            total_frames: Total number of frames processed
        """
        # Handle active shot at video end
        if self.is_shot_active and self.current_shot_start is not None:
            last_frame = total_frames - 1
            
            # Only complete shots that reached Release or Follow-through phase
            if self.previous_phase in ["Release", "Follow-through"]:
                print(f"\n🎬 VIDEO END: Completing active shot in {self.previous_phase} phase")
                self._complete_current_shot(last_frame)
            else:
                print(f"\n🎬 VIDEO END: Discarding incomplete shot in {self.previous_phase} phase (not saved)")
                print(f"   📍 Incomplete shot: frames {self.current_shot_start} → {last_frame}")
                print(f"   🔄 Phase sequence: {' → '.join(self.current_shot_phases) if self.current_shot_phases else 'None'}")
                print(f"   💭 Reason: Shot must reach Release or Follow-through to be saved")
                
                # Reset state without saving to shots list
                self.is_shot_active = False
                self.current_shot_start = None
                self.current_shot_end = None
                self.current_shot_fixed_torso = None
                self.torso_tracking_active = True
                self.rolling_torso_values = []
                self.rolling_torso_frames = []
                self.current_shot_phases = []
        
        # Initialize frame_shots list - same length as total frames
        self.frame_shots = [None] * total_frames
        
        # Assign shot_id to frames within each shot range
        for shot in self.shots:
            shot_id = shot['shot_id']
            start_frame = shot['start_frame']
            end_frame = shot['end_frame']
            
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx < total_frames:
                    self.frame_shots[frame_idx] = shot_id
        
        print(f"\n🎯 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📋 SHOT PROCESSING COMPLETED")
        print(f"   🎯 Total shots detected: {len(self.shots)}")
        print(f"   📊 Total frames processed: {total_frames}")
        print(f"🎯 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Validation report
        self._print_shot_validation_report()
    
    def get_shots_metadata(self) -> List[Dict]:
        """
        Get shots metadata in the same format as analyzer output.
        
        Returns:
            List of shot metadata dictionaries
        """
        shots_metadata = []
        for shot in self.shots:
            shot_meta = {
                "shot_id": shot['shot_id'],
                "start_frame": shot['start_frame'],
                "end_frame": shot['end_frame'],
                "total_frames": shot['total_frames'],
                "fixed_torso": shot['fixed_torso']
            }
            shots_metadata.append(shot_meta)
        
        return shots_metadata
    
    def _print_shot_validation_report(self):
        """Print detailed validation report for shot detection."""
        print(f"\n📊 Shot Detection Validation Report:")
        print("=" * 50)
        
        if not self.shots:
            print("❌ No shots detected")
            return
        
        for shot in self.shots:
            shot_id = shot['shot_id']
            original_id = shot.get('original_shot_id', shot_id)
            start = shot['start_frame']
            end = shot['end_frame']
            total = shot['total_frames']
            torso = shot['fixed_torso']
            
            print(f"🎯 Shot {shot_id}:")
            print(f"   📍 Frames: {start} → {end} ({total} frames)")
            print(f"   🔢 Original ID: {original_id}")
            print(f"   📏 Fixed torso: {torso:.4f}" if torso else "   📏 Fixed torso: None")
            
            # Check for common issues
            if total < 30:
                print(f"   ⚠️  Very short shot ({total} frames)")
            if not torso:
                print(f"   ⚠️  No torso fixed (no meaningful transition)")
        
        # Overall statistics
        assigned_frames = sum(1 for shot in self.frame_shots if shot is not None)
        coverage = (assigned_frames / len(self.frame_shots)) * 100 if self.frame_shots else 0
        print(f"\n📈 Coverage: {assigned_frames}/{len(self.frame_shots)} frames ({coverage:.1f}%)")
        
        # Validate shot continuity
        self._validate_shot_continuity()
    
    def _validate_shot_continuity(self):
        """Check for overlapping or missing shot assignments."""
        if not self.shots:
            return
        
        issues = []
        prev_end = -1
        
        for shot in sorted(self.shots, key=lambda x: x['start_frame']):
            start = shot['start_frame']
            end = shot['end_frame']
            shot_id = shot['shot_id']
            
            # Check for overlap
            if start <= prev_end:
                issues.append(f"Shot {shot_id} overlaps with previous shot")
            
            # Check for reasonable duration
            if end - start < 10:
                issues.append(f"Shot {shot_id} is very short ({end - start} frames)")
            
            prev_end = end
        
        if issues:
            print(f"\n⚠️  Shot continuity issues:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print(f"\n✅ Shot continuity validation passed")
    
    def _calculate_torso_from_pose(self, pose: Dict) -> float:
        """
        Calculate torso length from pose data.
        
        Args:
            pose: Pose data
            
        Returns:
            Torso length in normalized units
        """
        # Check if required keypoints exist
        required_keypoints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        for keypoint in required_keypoints:
            if keypoint not in pose:
                return 0.0
            kp_data = pose[keypoint]
            if not isinstance(kp_data, dict) or 'x' not in kp_data or 'y' not in kp_data:
                return 0.0
        
        # Get shoulder and hip positions
        left_shoulder = pose['left_shoulder']
        right_shoulder = pose['right_shoulder']
        left_hip = pose['left_hip']
        right_hip = pose['right_hip']
        
        # Confidence threshold for valid measurements
        confidence_threshold = 0.3
        
        valid_torso_lengths = []
        
        # Check left side torso (left shoulder to left hip)
        left_shoulder_conf = left_shoulder.get('confidence', 1.0)
        left_hip_conf = left_hip.get('confidence', 1.0)
        
        if left_shoulder_conf >= confidence_threshold and left_hip_conf >= confidence_threshold:
            left_torso_length = np.sqrt(
                (left_shoulder['x'] - left_hip['x'])**2 + 
                (left_shoulder['y'] - left_hip['y'])**2
            )
            valid_torso_lengths.append(left_torso_length)
        
        # Check right side torso (right shoulder to right hip)
        right_shoulder_conf = right_shoulder.get('confidence', 1.0)
        right_hip_conf = right_hip.get('confidence', 1.0)
        
        if right_shoulder_conf >= confidence_threshold and right_hip_conf >= confidence_threshold:
            right_torso_length = np.sqrt(
                (right_shoulder['x'] - right_hip['x'])**2 + 
                (right_shoulder['y'] - right_hip['y'])**2
            )
            valid_torso_lengths.append(right_torso_length)
        
        # Return average of valid measurements
        if valid_torso_lengths:
            return np.mean(valid_torso_lengths)
        else:
            return 0.0