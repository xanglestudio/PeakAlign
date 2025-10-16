# PeakAlign v1.4 - xanglestudio.com/PeakAlign
# Copyright (c) 2025 Xangle Studio - xanglestudio.com
# Licensed under the MIT License. See LICENSE file in the project root for full license text.

#!/usr/bin/env python3
"""
Synced Video Audio Peak Detector
-------------------------------
This script analyzes video files to find audio peaks and calculates the exact
aligned frame ranges for extraction, saving the results to a peak.json file
in the video folder.

Requirements:
------------
Python 3.6 or higher
Required packages (install via pip):
    pip install numpy moviepy scipy tqdm

Usage:
------
1. Basic audio peak analysis:
   python audio_peak_detector_sync.py /path/to/videos

2. Specify video file extensions:
   python audio_peak_detector_sync.py /path/to/videos --extensions .mp4,.avi,.mov
   
3. Analyze only the first X% of each video:
   python audio_peak_detector_sync.py /path/to/videos --analyze-percent 20.0
"""

import os
import numpy as np
from moviepy.editor import VideoFileClip
import argparse
from statistics import mean
from pathlib import Path
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import tempfile
import json


def analyze_video_audio(video_path, analyze_percent=100.0, analyze_end=False):
    """
    Analyze a video file to find the frame with the highest audio peak.
    
    Args:
        video_path (str): Path to the video file
        analyze_percent (float): Only analyze X percent of the video
        analyze_end (bool): If True, analyze the ending part instead of beginning
    
    Returns:
        dict: Analysis results or error information
    """
    try:
        print(f"\nAnalyzing: {os.path.basename(video_path)}")
        clip = VideoFileClip(video_path)
        
        if clip.audio is None:
            print(f"No audio track in {os.path.basename(video_path)}")
            clip.close()
            return None
            
        # Calculate analysis duration and start time
        full_duration = clip.duration
        analyze_duration = full_duration * (analyze_percent / 100.0)
        
        if analyze_percent < 100.0:
            if analyze_end:
                start_time = full_duration - analyze_duration
                print(f"Analyzing last {analyze_percent:.1f}% of video ({analyze_duration:.2f}s from {start_time:.2f}s to {full_duration:.2f}s)")
            else:
                start_time = 0
                print(f"Analyzing first {analyze_percent:.1f}% of video ({analyze_duration:.2f}s from 0s to {analyze_duration:.2f}s)")
        else:
            start_time = 0
            
        try:
            # Extract audio for the specified duration and position
            audio_subclip = clip.audio.subclip(start_time, start_time + analyze_duration)
            audio_array = audio_subclip.to_soundarray()
        except Exception:
            try:
                temp_audio_file = os.path.join(tempfile.gettempdir(), 
                                              f"temp_audio_{os.path.basename(video_path)}.wav")
                
                # Write the analyzed portion to temporary file
                audio_subclip = clip.audio.subclip(start_time, start_time + analyze_duration)
                audio_subclip.write_audiofile(temp_audio_file, logger=None)
                
                from scipy.io import wavfile
                sample_rate, audio_array = wavfile.read(temp_audio_file)
                if os.path.exists(temp_audio_file):
                    os.remove(temp_audio_file)
            except Exception as e2:
                print(f"Failed to extract audio from {os.path.basename(video_path)}")
                clip.close()
                return None
        
        if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        audio_abs = np.abs(audio_array)
        peak_index = np.argmax(audio_abs)
        
        # Calculate peak time relative to the analyzed segment
        peak_time_in_segment = peak_index / clip.audio.fps
        # Calculate actual peak time in the full video
        peak_time_seconds = start_time + peak_time_in_segment
        peak_frame = int(peak_time_seconds * clip.fps)
        
        frame_duration_ms = 1000 / clip.fps
        frame_start_time_ms = peak_frame * frame_duration_ms
        peak_time_ms = peak_time_seconds * 1000
        ms_within_frame = peak_time_ms - frame_start_time_ms
        
        # Calculate total frames in video
        total_frames = int(clip.fps * clip.duration)
        
        result = {
            'filename': os.path.basename(video_path),
            'full_path': str(Path(video_path).resolve()),
            'peak_frame': peak_frame,
            'peak_time_seconds': peak_time_seconds,
            'ms_within_frame': ms_within_frame,
            'intensity': float(audio_abs[peak_index]),
            'frame_rate': clip.fps,
            'duration': clip.duration,
            'total_frames': total_frames,
            'analyzed_percent': analyze_percent,
            'analyzed_duration': analyze_duration,
            'analyzed_start_time': start_time,
            'analyzed_end': analyze_end,
            'width': clip.w,
            'height': clip.h
        }
        
        clip.close()
        return result
        
    except Exception as e:
        print(f"Error processing {os.path.basename(video_path)}: {str(e)}")
        return None


def process_videos_parallel(files, folder_path, processes, analyze_percent=100.0, analyze_end=False):
    """Process multiple videos in parallel for audio analysis."""
    with Pool(processes=processes) as pool:
        results = list(tqdm(
            pool.imap(
                partial(analyze_video_audio, analyze_percent=analyze_percent, analyze_end=analyze_end),
                [os.path.join(folder_path, f) for f in files]
            ),
            total=len(files),
            desc="Analyzing videos"
        ))
    return [r for r in results if r is not None]


def calculate_sync_frames(results, frames_to_extract):
    """
    Calculate synchronized frame ranges for each video based on peak alignment.
    
    Args:
        results (list): List of video analysis results
        frames_to_extract (int): Total number of frames to extract per video
        
    Returns:
        tuple: (updated_results, common_frames)
    """
    if not results:
        return [], 0
    
    # Find minimum peak frame across all videos to use as reference
    min_peak_frame = min(r['peak_frame'] for r in results)
    
    # Initialize tracking variables
    max_pre_frames = min_peak_frame
    max_post_frames = min(r['total_frames'] - r['peak_frame'] - 1 for r in results)
    
    # Calculate how many frames to extract before and after peaks
    # Try to balance around the peak if possible
    target_pre_frames = frames_to_extract // 2
    target_post_frames = frames_to_extract - target_pre_frames
    
    # Adjust if we have fewer frames available than requested
    pre_frames = min(max_pre_frames, target_pre_frames)
    post_frames = min(max_post_frames, target_post_frames)
    
    # If we didn't get all the frames we wanted, try to compensate
    total_frames = pre_frames + 1 + post_frames  # +1 for the peak frame
    if total_frames < frames_to_extract:
        # Try to add more pre-frames if possible
        additional_pre = min(max_pre_frames - pre_frames, frames_to_extract - total_frames)
        pre_frames += additional_pre
        total_frames += additional_pre
        
        # If still not enough, try to add more post-frames
        if total_frames < frames_to_extract:
            additional_post = min(max_post_frames - post_frames, frames_to_extract - total_frames)
            post_frames += additional_post
            total_frames += additional_post
    
    # Calculate frame offsets and ranges for each video
    updated_results = []
    for video in results:
        # For each video, the peak frame should be aligned across all videos
        # Calculate relative offset from the minimum peak frame
        peak_offset = video['peak_frame'] - min_peak_frame
        
        # Calculate exact extraction start and end frames for this video
        start_frame = video['peak_frame'] - pre_frames
        end_frame = video['peak_frame'] + post_frames  # end_frame is inclusive in our system
        
        # Ensure we don't go out of bounds
        if start_frame < 0:
            start_frame = 0
        if end_frame >= video['total_frames']:
            end_frame = video['total_frames'] - 1
        
        # Calculate actual frames extracted
        actual_frames = end_frame - start_frame + 1
        
        # Add extraction info to video data
        updated_video = dict(video)
        updated_video['extraction'] = {
            'start_frame': start_frame,
            'end_frame': end_frame,  # Inclusive end frame
            'frame_count': actual_frames,
            'peak_offset': peak_offset,
            'aligned_peak_frame': pre_frames  # Peak frame position in extracted sequence
        }
        
        updated_results.append(updated_video)
    
    # Calculate common frame count (minimum across all videos)
    common_frames = min(v['extraction']['frame_count'] for v in updated_results)
    
    # Update each video with the common frame count
    for video in updated_results:
        # Adjust end frame to match common frame count
        if video['extraction']['frame_count'] > common_frames:
            # Calculate excess frames
            excess = video['extraction']['frame_count'] - common_frames
            
            # Try to keep frames balanced around peak by removing from both ends
            pre_excess = excess // 2
            post_excess = excess - pre_excess
            
            # Adjust start and end frames
            video['extraction']['start_frame'] += pre_excess
            video['extraction']['end_frame'] -= post_excess
            video['extraction']['frame_count'] = common_frames
    
    return updated_results, common_frames


def main():
    parser = argparse.ArgumentParser(description='Synced audio peak detector with aligned frame ranges')
    parser.add_argument('folder', help='Folder containing video files')
    parser.add_argument('--extensions', default='.mp4,.mov', 
                      help='Comma-separated video extensions to process')
    parser.add_argument('--processes', type=int, default=max(1, cpu_count() - 1),
                      help='Number of parallel processes for audio analysis')
    parser.add_argument('--frames-to-extract', type=int, default=300,
                      help='Target number of frames to extract per video')
    parser.add_argument('--analyze-percent', type=float, default=100.0,
                      help='Only analyze X percent of each video (default: 100.0)')
    parser.add_argument('--analyze-end', action='store_true',
                      help='Analyze the ending part instead of the beginning (use with --analyze-percent)')
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    folder_path = args.folder
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in args.extensions.split(',')]
    
    files = sorted([f for f in os.listdir(folder_path) 
                   if any(f.lower().endswith(ext) for ext in extensions)])
    
    if not files:
        print("No video files found in the specified directory.")
        return
        
    print(f"\nFound {len(files)} video files to process")
    print(f"Using {args.processes} processes for audio analysis")
    
    # Audio Analysis
    analysis_start = datetime.now()
    
    # Validate analyze_percent parameter
    if args.analyze_percent <= 0 or args.analyze_percent > 100:
        print(f"ERROR: analyze-percent must be between 0.1 and 100.0 (got {args.analyze_percent})")
        return
    
    if args.analyze_percent < 100:
        if args.analyze_end:
            print(f"Analyzing the last {args.analyze_percent:.1f}% of each video")
        else:
            print(f"Analyzing the first {args.analyze_percent:.1f}% of each video")
        
    results = process_videos_parallel(files, folder_path, args.processes, args.analyze_percent, args.analyze_end)
    analysis_time = datetime.now() - analysis_start
    
    if not results:
        print("\nNo valid results found.")
        return
    
    print("\nAUDIO ANALYSIS RESULTS")
    print("=" * 65)
    print("File                  Frame    Time(s)    Ms in Frame    Intensity")
    print("-" * 65)
    
    for result in sorted(results, key=lambda x: x['filename']):
        print(f"{result['filename']:<20} {result['peak_frame']:<8} "
              f"{result['peak_time_seconds']:<10.3f} {result['ms_within_frame']:<13.2f} "
              f"{result['intensity']:.6f}")
    
    ms_values = [r['ms_within_frame'] for r in results]
    min_ms = min(ms_values)
    max_ms = max(ms_values)
    avg_ms = mean(ms_values)
    
    print("\nSTATISTICS:")
    print(f"Ms in Frame - Min: {min_ms:.2f}, Max: {max_ms:.2f}, Average: {avg_ms:.2f}")
    
    highest_peak = max(results, key=lambda x: x['intensity'])
    print("\nHIGHEST PEAK:", end=' ')
    print(f"{highest_peak['filename']}, Frame {highest_peak['peak_frame']}, "
          f"{highest_peak['peak_time_seconds']:.3f}s, {highest_peak['ms_within_frame']:.2f}ms, "
          f"Int:{highest_peak['intensity']:.6f}")
    
    # Calculate synchronized frame ranges
    print("\nCALCULATING SYNCHRONIZED FRAME RANGES")
    print("=" * 65)
    synced_results, common_frames = calculate_sync_frames(results, args.frames_to_extract)
    
    print(f"Target frames to extract: {args.frames_to_extract}")
    print(f"Common frame count: {common_frames}")
    print(f"Min peak frame: {min(r['peak_frame'] for r in results)}")
    
    # Print extraction ranges
    print("\nSYNCHRONIZED EXTRACTION RANGES:")
    print("=" * 80)
    print("File                  Start Frame  End Frame  Count  Peak Frame  Peak Offset")
    print("-" * 80)
    
    for video in synced_results:
        extraction = video['extraction']
        print(f"{video['filename']:<20} {extraction['start_frame']:<12} "
              f"{extraction['end_frame']:<10} {extraction['frame_count']:<6} "
              f"{video['peak_frame']:<11} {extraction['peak_offset']}")
    
    # Create metadata dictionary
    metadata = {
        'analysis_timestamp': datetime.now().isoformat(),
        'video_count': len(results),
        'min_peak_frame': min(r['peak_frame'] for r in results),
        'frames_requested': args.frames_to_extract,
        'frames_per_video': common_frames,
        'analyzed_percent': args.analyze_percent,
        'analyzed_end': args.analyze_end,
        'stats': {
            'min_ms_within_frame': min_ms,
            'max_ms_within_frame': max_ms,
            'avg_ms_within_frame': avg_ms
        },
        'highest_peak': {
            'filename': highest_peak['filename'],
            'peak_frame': highest_peak['peak_frame'],
            'intensity': highest_peak['intensity']
        },
        'videos': synced_results
    }
    
    # Save to peak.json in the video folder
    peak_json_path = os.path.join(folder_path, 'peak.json')
    with open(peak_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nAnalysis results saved to: {peak_json_path}")
    if args.analyze_percent < 100:
        analysis_part = "last" if args.analyze_end else "first"
        print(f"Analysis coverage: {analysis_part.title()} {args.analyze_percent:.1f}% of each video")
    else:
        print(f"Analysis coverage: Full video analysis")
    print(f"Synchronized frame ranges have been calculated for the extractor")
    
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print("\nPERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"Audio analysis time: {analysis_time}")
    print(f"Total processing time: {total_time}")
    print(f"\nUse frame_extractor.py to extract synced frames.")


if __name__ == "__main__":
    main()