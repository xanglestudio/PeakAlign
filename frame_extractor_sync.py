# PeakAlign v1.4 - xanglestudio.com/PeakAlign
# Copyright (c) 2025 Xangle Studio - xanglestudio.com
# Licensed under the MIT License. See LICENSE file in the project root for full license text.

#!/usr/bin/env python3
"""
Synchronized Frame Extractor
--------------------------
This script extracts frames from videos based on the synchronized frame ranges
calculated by the audio_peak_detector_sync.py script and stored in peak.json.

Requirements:
------------
Python 3.6 or higher
Required packages (install via pip):
    pip install opencv-python tqdm

Usage:
------
1. Basic extraction (by iteration - default):
   python frame_extractor_sync.py /path/to/videos

2. Extract by camera instead of by iteration:
   python frame_extractor_sync.py /path/to/videos --by-camera

3. Specify output directory:
   python frame_extractor_sync.py /path/to/videos --output extracted_frames

4. Process specific cameras:
   python frame_extractor_sync.py /path/to/videos --start-index 40 --end-index 46
"""

import os
import cv2
import json
import argparse
import logging
import sys
from pathlib import Path
import shutil
from datetime import datetime
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('frame_extractor.log', mode='w')
    ]
)

def extract_frames_by_iteration(video_path, camera_index, output_dir, start_frame, end_frame, frame_start_offset=0, frame_end_offset=None):
    """
    Extract frames from a video file and organize them by iteration.
    (One folder per frame, with all cameras inside)
    
    Args:
        video_path (str): Path to the video file
        camera_index (int): Numeric index of the camera (used for filename)
        output_dir (str): Base output directory
        start_frame (int): First frame to extract from video (inclusive)
        end_frame (int): Last frame to extract from video (inclusive)
        frame_start_offset (int): Offset to start extracting from in the frame sequence
        frame_end_offset (int): Offset to end extracting at in the frame sequence
        
    Returns:
        int: Number of frames extracted
    """
    # Calculate actual video frame range
    total_frames_in_sequence = end_frame - start_frame + 1
    
    # Apply frame selection offsets
    if frame_end_offset is None:
        frame_end_offset = total_frames_in_sequence - 1
    
    actual_start = start_frame + frame_start_offset
    actual_end = start_frame + frame_end_offset
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return 0
    
    # Verify frame count
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if actual_start >= total_video_frames:
        logging.error(f"Start frame {actual_start} exceeds total frames {total_video_frames}")
        cap.release()
        return 0
    
    # Adjust end frame if needed
    if actual_end >= total_video_frames:
        logging.warning(f"End frame {actual_end} exceeds total frames {total_video_frames}. Adjusting to {total_video_frames-1}")
        actual_end = total_video_frames - 1
    
    # Number of frames to extract
    num_frames = actual_end - actual_start + 1
    
    # Set position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)
    
    # Extract frames
    frames_extracted = 0
    with tqdm(total=num_frames, desc=f"Extracting from {os.path.basename(video_path)}") as pbar:
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame at position {actual_start + i}")
                break
            
            # Create frame directory if it doesn't exist (using i#### format)
            frame_dir = os.path.join(output_dir, f"i{frame_start_offset + i:04d}")
            os.makedirs(frame_dir, exist_ok=True)
            
            # Save frame with numeric filename
            frame_path = os.path.join(frame_dir, f"{camera_index+1:04d}.png")
            if cv2.imwrite(frame_path, frame):
                frames_extracted += 1
            else:
                logging.error(f"Failed to save frame to {frame_path}")
            
            pbar.update(1)
    
    cap.release()
    logging.info(f"Extracted {frames_extracted}/{num_frames} frames from {os.path.basename(video_path)}")
    return frames_extracted

def extract_frames_by_camera(video_path, camera_index, output_dir, start_frame, end_frame, frame_start_offset=0, frame_end_offset=None):
    """
    Extract frames from a video file and organize them by camera.
    (One folder per camera, with all frames inside)
    
    Args:
        video_path (str): Path to the video file
        camera_index (int): Numeric index of the camera (for folder naming)
        output_dir (str): Base output directory
        start_frame (int): First frame to extract from video (inclusive)
        end_frame (int): Last frame to extract from video (inclusive)
        frame_start_offset (int): Offset to start extracting from in the frame sequence
        frame_end_offset (int): Offset to end extracting at in the frame sequence
        
    Returns:
        int: Number of frames extracted
    """
    # Calculate actual video frame range
    total_frames_in_sequence = end_frame - start_frame + 1
    
    # Apply frame selection offsets
    if frame_end_offset is None:
        frame_end_offset = total_frames_in_sequence - 1
    
    actual_start = start_frame + frame_start_offset
    actual_end = start_frame + frame_end_offset
    
    # Create standardized camera directory name with c#### format
    camera_dir = os.path.join(output_dir, f"c{camera_index+1:04d}")
    os.makedirs(camera_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return 0
    
    # Verify frame count
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if actual_start >= total_video_frames:
        logging.error(f"Start frame {actual_start} exceeds total frames {total_video_frames}")
        cap.release()
        return 0
    
    # Adjust end frame if needed
    if actual_end >= total_video_frames:
        logging.warning(f"End frame {actual_end} exceeds total frames {total_video_frames}. Adjusting to {total_video_frames-1}")
        actual_end = total_video_frames - 1
    
    # Number of frames to extract
    num_frames = actual_end - actual_start + 1
    
    # Set position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)
    
    # Extract frames
    frames_extracted = 0
    with tqdm(total=num_frames, desc=f"Extracting from {os.path.basename(video_path)}") as pbar:
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame at position {actual_start + i}")
                break
            
            # Save frame with sequential filename based on offset
            frame_path = os.path.join(camera_dir, f"frame_{frame_start_offset + i:04d}.png")
            if cv2.imwrite(frame_path, frame):
                frames_extracted += 1
            else:
                logging.error(f"Failed to save frame to {frame_path}")
            
            pbar.update(1)
    
    cap.release()
    logging.info(f"Extracted {frames_extracted}/{num_frames} frames to {camera_dir}")
    return frames_extracted

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract synchronized frames from videos")
    parser.add_argument("folder", help="Folder containing video files and peak.json")
    parser.add_argument("--start-index", type=int, default=0, 
                       help="Start index for camera selection - which videos to process (0-based)")
    parser.add_argument("--end-index", type=int, default=None, 
                       help="End index for camera selection - which videos to process (exclusive, 0-based)")
    parser.add_argument("--by-camera", action="store_true", 
                       help="Organize output by camera instead of by iteration")
    
    # Frame selection options (mutually exclusive)
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument("--frame-range", type=str, 
                       help="Extract specific frame range (e.g., '10-50')")
    frame_group.add_argument("--specific-frame", type=int, 
                       help="Extract a specific frame number")
    
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    start_time = datetime.now()
    logging.info(f"Frame extraction started at {start_time}")
    logging.info(f"Organization mode: {'By camera' if args.by_camera else 'By iteration'}")
    
    try:
        # Set up paths
        folder_path = args.folder
        if not os.path.isdir(folder_path):
            logging.error(f"Folder not found: {folder_path}")
            return
            
        # Determine output directory based on organization mode
        output_subdir = "cameras" if args.by_camera else "iterations"
        output_dir = os.path.join(folder_path, output_subdir)
            
        # Check for peak.json file
        peak_json_path = os.path.join(folder_path, 'peak.json')
        if not os.path.exists(peak_json_path):
            logging.error(f"peak.json not found in {folder_path}")
            logging.error("Run audio_peak_detector_sync.py first to generate peak.json")
            return
        
        # Load peak.json
        logging.info(f"Loading peak.json from: {peak_json_path}")
        with open(peak_json_path, "r") as f:
            data = json.load(f)
        
        # Get video information
        videos = data.get("videos", [])
        if not videos:
            logging.error("No video data found in peak.json")
            return
        
        # Get frames per video
        frames_per_video = data.get("frames_per_video")
        if frames_per_video is None:
            logging.error("No frames_per_video found in peak.json")
            return
        
        logging.info(f"Found {len(videos)} videos in peak.json")
        logging.info(f"Frames per video: {frames_per_video}")
        
        # Parse frame selection options
        frame_start = 0
        frame_end = frames_per_video - 1
        
        if args.frame_range:
            try:
                if '-' in args.frame_range:
                    start_str, end_str = args.frame_range.split('-', 1)
                    frame_start = int(start_str)
                    frame_end = int(end_str)
                    
                    if frame_start < 0 or frame_end >= frames_per_video or frame_start > frame_end:
                        logging.error(f"Invalid frame range: {args.frame_range}. Must be within 0-{frames_per_video-1}")
                        return
                    
                    logging.info(f"Extracting frame range: {frame_start} to {frame_end}")
                else:
                    logging.error("Frame range must be in format 'start-end' (e.g., '10-50')")
                    return
            except ValueError:
                logging.error(f"Invalid frame range format: {args.frame_range}")
                return
        
        elif args.specific_frame is not None:
            if args.specific_frame < 0 or args.specific_frame >= frames_per_video:
                logging.error(f"Invalid specific frame: {args.specific_frame}. Must be within 0-{frames_per_video-1}")
                return
            
            frame_start = args.specific_frame
            frame_end = args.specific_frame
            logging.info(f"Extracting specific frame: {args.specific_frame}")
        
        else:
            logging.info(f"Extracting all frames: 0 to {frames_per_video-1}")
        
        # Calculate actual frames to extract
        actual_frames_to_extract = frame_end - frame_start + 1
        
        # Filter videos by index (choose which cameras to process)
        start_idx = args.start_index
        end_idx = args.end_index if args.end_index is not None else len(videos)
        
        if start_idx < 0 or start_idx >= len(videos):
            logging.error(f"Invalid start index: {start_idx}. Must be between 0 and {len(videos)-1}")
            return
        
        if end_idx <= start_idx or end_idx > len(videos):
            logging.error(f"Invalid end index: {end_idx}. Must be between {start_idx+1} and {len(videos)}")
            return
        
        selected_videos = videos[start_idx:end_idx]
        logging.info(f"Selected {len(selected_videos)} cameras (indices {start_idx} to {end_idx-1})")
        
        # Create output directory
        output_dir = os.path.join(folder_path, output_subdir)
        if os.path.exists(output_dir):
            logging.info(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")
        
        # Extract frames from each video
        successful_extractions = 0
        
        for i, video in enumerate(selected_videos):
            video_name = os.path.basename(video["filename"])
            video_path = video["full_path"]
            extraction = video.get("extraction")
            
            if not extraction:
                logging.error(f"No extraction data found for {video_name}")
                continue
            
            start_frame = extraction["start_frame"]
            end_frame = extraction["end_frame"]
            
            # Check if video file exists
            if not os.path.exists(video_path):
                # Try relative path
                alt_path = os.path.join(folder_path, video_name)
                if os.path.exists(alt_path):
                    video_path = alt_path
                    logging.info(f"Using alternative path for {video_name}: {video_path}")
                else:
                    logging.error(f"Video file not found: {video_path}")
                    logging.error(f"Alternative path not found: {alt_path}")
                    continue
            
            # Extract frames
            logging.info(f"Extracting frames from {video_name} (camera {i+1}/{len(selected_videos)})")
            logging.info(f"Range: {start_frame} to {end_frame} ({end_frame-start_frame+1} frames)")
            
            # Use appropriate extraction function based on organization mode
            if args.by_camera:
                frames_extracted = extract_frames_by_camera(
                    video_path, i + start_idx, output_dir, start_frame, end_frame,
                    frame_start, frame_end
                )
            else:
                frames_extracted = extract_frames_by_iteration(
                    video_path, i + start_idx, output_dir, start_frame, end_frame,
                    frame_start, frame_end
                )
            
            if frames_extracted == actual_frames_to_extract:
                successful_extractions += 1
            else:
                logging.warning(f"Expected {actual_frames_to_extract} frames, got {frames_extracted} from {video_name}")
        
        if successful_extractions == 0:
            logging.error("No frames were successfully extracted")
            return
            
        # Create metadata file
        organization_mode = "by_camera" if args.by_camera else "by_iteration"
        
        metadata = {
            "extraction_time": datetime.now().isoformat(),
            "source_folder": folder_path,
            "peak_json": peak_json_path,
            "output_folder": output_subdir,
            "cameras_processed": successful_extractions,
            "frames_per_camera": actual_frames_to_extract,
            "organization_mode": organization_mode,
            "selected_range": f"{start_idx}-{end_idx-1}",
            "frame_selection": {
                "start": frame_start,
                "end": frame_end,
                "total": actual_frames_to_extract
            },
            "total_cameras": len(videos),
            "total_files": actual_frames_to_extract * successful_extractions
        }
        
        with open(os.path.join(output_dir, "extraction_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Count directories
        if args.by_camera:
            directories = [d for d in os.listdir(output_dir) 
                          if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("c")]
            logging.info(f"Created {len(directories)} camera directories")
        else:
            frame_folders = [d for d in os.listdir(output_dir) if d.startswith("i")]
            logging.info(f"Created {len(frame_folders)} iteration directories")
        
        end_time = datetime.now()
        logging.info(f"Frame extraction completed in {end_time - start_time}")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Cameras processed: {successful_extractions}")
        logging.info(f"Organization mode: {organization_mode}")
        logging.info(f"Total files: {frames_per_video * successful_extractions}")
        
        print("\nFRAME EXTRACTION COMPLETE")
        print("=" * 65)
        print(f"Output directory: {output_dir}")
        print(f"Organization mode: {'By camera' if args.by_camera else 'By iteration'}")
        print(f"Cameras processed: {successful_extractions}/{len(selected_videos)}")
        print(f"Frames per camera: {frames_per_video}")
        print(f"Total files: {frames_per_video * successful_extractions}")
        print(f"Processing time: {end_time - start_time}")
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()