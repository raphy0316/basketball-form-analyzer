#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video resolution checker for Bakke, EdgeCase, and Standard folders
"""

import cv2
import os
import glob
from typing import Dict, List, Tuple
from collections import defaultdict

def get_video_properties(video_path: str) -> Tuple[int, int, float, int]:
    """Get video resolution, FPS, and total frames"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0, 0.0, 0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return width, height, fps, total_frames

def check_video_resolutions():
    """Check resolutions of videos in Bakke, EdgeCase, and Standard folders"""
    
    print("🎬 Video Resolution Checker")
    print("=" * 60)
    
    # Define folders to check
    folders = ["Bakke", "EdgeCase", "Standard"]
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    
    # Results storage
    all_videos = []
    resolution_stats = defaultdict(int)
    folder_stats = defaultdict(list)
    
    for folder in folders:
        folder_path = os.path.join("data", "video", folder)
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            continue
        
        print(f"\n📁 Checking {folder} folder...")
        print("-" * 40)
        
        folder_videos = []
        
        # Find all video files in the folder
        for ext in video_extensions:
            pattern = os.path.join(folder_path, ext)
            video_files = glob.glob(pattern)
            
            for video_path in video_files:
                video_name = os.path.basename(video_path)
                width, height, fps, total_frames = get_video_properties(video_path)
                
                if width == 0 or height == 0:
                    print(f"❌ Could not read: {video_name}")
                    continue
                
                resolution = f"{width}x{height}"
                duration_sec = total_frames / fps if fps > 0 else 0
                
                video_info = {
                    "name": video_name,
                    "path": video_path,
                    "folder": folder,
                    "width": width,
                    "height": height,
                    "resolution": resolution,
                    "fps": fps,
                    "total_frames": total_frames,
                    "duration_sec": duration_sec
                }
                
                all_videos.append(video_info)
                folder_stats[folder].append(video_info)
                resolution_stats[resolution] += 1
                
                print(f"✅ {video_name}")
                print(f"   Resolution: {resolution}")
                print(f"   FPS: {fps:.1f}")
                print(f"   Duration: {duration_sec:.1f}s ({total_frames} frames)")
        
        print(f"📊 {folder}: {len(folder_stats[folder])} videos found")
    
    # Print summary
    print(f"\n📈 RESOLUTION SUMMARY")
    print("=" * 60)
    
    if not all_videos:
        print("❌ No videos found in any folder")
        return
    
    print(f"Total videos: {len(all_videos)}")
    print(f"Folders checked: {', '.join(folders)}")
    
    # Resolution statistics
    print(f"\n🎯 Resolution Distribution:")
    print("-" * 30)
    for resolution, count in sorted(resolution_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_videos)) * 100
        print(f"{resolution}: {count} videos ({percentage:.1f}%)")
    
    # Folder statistics
    print(f"\n📁 Videos per folder:")
    print("-" * 30)
    for folder in folders:
        count = len(folder_stats[folder])
        if count > 0:
            resolutions = set(v["resolution"] for v in folder_stats[folder])
            print(f"{folder}: {count} videos")
            print(f"  Resolutions: {', '.join(sorted(resolutions))}")
        else:
            print(f"{folder}: 0 videos")
    
    # Detailed list
    print(f"\n📋 DETAILED VIDEO LIST")
    print("=" * 60)
    
    for folder in folders:
        if folder_stats[folder]:
            print(f"\n📁 {folder} folder:")
            print("-" * 30)
            
            for video in sorted(folder_stats[folder], key=lambda x: x["name"]):
                print(f"• {video['name']}")
                print(f"  {video['resolution']} @ {video['fps']:.1f}fps")
                print(f"  Duration: {video['duration_sec']:.1f}s")
    
    # Save results to file
    save_results_to_file(all_videos, resolution_stats)

def save_results_to_file(all_videos: List[Dict], resolution_stats: Dict[str, int]):
    """Save results to a text file"""
    
    output_file = "video_resolution_report.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Video Resolution Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total videos: {len(all_videos)}\n")
        f.write(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Resolution summary
        f.write("Resolution Distribution:\n")
        f.write("-" * 30 + "\n")
        for resolution, count in sorted(resolution_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_videos)) * 100
            f.write(f"{resolution}: {count} videos ({percentage:.1f}%)\n")
        
        f.write("\nDetailed List:\n")
        f.write("=" * 30 + "\n")
        
        for video in sorted(all_videos, key=lambda x: (x["folder"], x["name"])):
            f.write(f"{video['folder']}/{video['name']}\n")
            f.write(f"  Resolution: {video['resolution']}\n")
            f.write(f"  FPS: {video['fps']:.1f}\n")
            f.write(f"  Duration: {video['duration_sec']:.1f}s\n")
            f.write(f"  Frames: {video['total_frames']}\n\n")
    
    print(f"\n✅ Results saved to: {output_file}")

if __name__ == "__main__":
    check_video_resolutions() 