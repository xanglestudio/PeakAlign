# PeakAlign v1.4 - xanglestudio.com/PeakAlign
# Copyright (c) 2025 Xangle Studio - xanglestudio.com
# Licensed under the MIT License. See LICENSE file in the project root for full license text.

#!/usr/bin/env python3
"""
Video Batch Processor
------------------
This script runs audio peak detection and frame extraction on multiple folders,
processing them one at a time for easy batch processing of large datasets.

Requirements:
------------
Python 3.6 or higher
Required packages:
    All requirements from audio_peak_detector_sync.py and frame_extractor_sync.py

Usage:
------
1. Process a single root folder containing multiple video folders:
   python batch_processor.py --root /path/to/root_folder

2. Process a specific list of folders:
   python batch_processor.py --folders /path/to/folder1 /path/to/folder2 /path/to/folder3

3. Process with custom frame count:
   python batch_processor.py --root /path/to/root_folder --frames 500

4. Process with camera-based organization:
   python batch_processor.py --root /path/to/root_folder --by-camera

5. Process with both detection and extraction:
   python batch_processor.py --root /path/to/root_folder --detect --extract

6. Process with just detection:
   python batch_processor.py --root /path/to/root_folder --detect

7. Process with just extraction:
   python batch_processor.py --root /path/to/root_folder --extract
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('batch_processor.log', mode='w')
    ]
)

def is_video_folder(folder_path):
    """
    Check if a folder contains video files.
    
    Args:
        folder_path (str): Path to the folder
        
    Returns:
        bool: True if the folder contains video files, False otherwise
    """
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    if not os.path.isdir(folder_path):
        return False
        
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            return True
    
    return False

def find_video_folders(root_path):
    """
    Find all folders containing video files in a root folder.
    
    Args:
        root_path (str): Path to the root folder
        
    Returns:
        list: List of paths to folders containing video files
    """
    video_folders = []
    
    # Check if the root itself is a video folder
    if is_video_folder(root_path):
        return [root_path]
    
    # Otherwise, check all subdirectories
    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        if os.path.isdir(item_path) and is_video_folder(item_path):
            video_folders.append(item_path)
    
    return sorted(video_folders)

def run_detector(folder_path, frames, analyze_percent=100.0):
    """
    Run the audio peak detector on a folder.
    
    Args:
        folder_path (str): Path to the folder containing videos
        frames (int): Number of frames to extract
        analyze_percent (float): Percentage of video to analyze
        
    Returns:
        bool: True if successful, False otherwise
    """
    detector_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio_peak_detector_sync.py')
    if not os.path.exists(detector_path):
        logging.error(f"Detector script not found: {detector_path}")
        return False
    
    logging.info(f"Running audio peak detection on: {folder_path}")
    
    cmd = [
        sys.executable,
        detector_path,
        folder_path,
        '--frames-to-extract', str(frames),
        '--analyze-percent', str(analyze_percent)
    ]
    
    try:
        start_time = datetime.now()
        logging.info(f"Starting detector with command: {' '.join(cmd)}")
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        end_time = datetime.now()
        
        if process.returncode == 0:
            logging.info(f"Detection completed in {end_time - start_time}")
            return True
        else:
            logging.error(f"Detection failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        logging.error(f"Exception running detector: {str(e)}")
        return False

def run_extractor(folder_path, by_camera=False, start_index=0, end_index=None):
    """
    Run the frame extractor on a folder.
    
    Args:
        folder_path (str): Path to the folder containing videos and peak.json
        by_camera (bool): Whether to organize output by camera
        start_index (int): Start index for camera selection
        end_index (int): End index for camera selection
        
    Returns:
        bool: True if successful, False otherwise
    """
    extractor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frame_extractor_sync.py')
    if not os.path.exists(extractor_path):
        logging.error(f"Extractor script not found: {extractor_path}")
        return False
    
    # Check for peak.json
    peak_json_path = os.path.join(folder_path, 'peak.json')
    if not os.path.exists(peak_json_path):
        logging.error(f"peak.json not found in {folder_path}")
        logging.error("Run detection first")
        return False
    
    logging.info(f"Running frame extraction on: {folder_path}")
    
    cmd = [
        sys.executable,
        extractor_path,
        folder_path
    ]
    
    if by_camera:
        cmd.append('--by-camera')
    
    if start_index > 0:
        cmd.extend(['--start-index', str(start_index)])
    
    if end_index is not None:
        cmd.extend(['--end-index', str(end_index)])
    
    try:
        start_time = datetime.now()
        logging.info(f"Starting extractor with command: {' '.join(cmd)}")
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        end_time = datetime.now()
        
        if process.returncode == 0:
            logging.info(f"Extraction completed in {end_time - start_time}")
            return True
        else:
            logging.error(f"Extraction failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        logging.error(f"Exception running extractor: {str(e)}")
        return False

def process_folder(folder_path, args):
    """
    Process a single folder with detection and/or extraction.
    
    Args:
        folder_path (str): Path to the folder
        args (Namespace): Command-line arguments
        
    Returns:
        tuple: (detection_success, extraction_success)
    """
    folder_name = os.path.basename(folder_path)
    logging.info(f"\n{'='*30} Processing: {folder_name} {'='*30}")
    
    detection_success = True
    extraction_success = True
    
    # Run detection if requested
    if args.detect:
        detection_success = run_detector(
            folder_path,
            args.frames,
            args.analyze_percent
        )
        
        if not detection_success:
            logging.error(f"Detection failed for {folder_path}, skipping extraction")
            if args.extract:
                return False, False
    
    # Run extraction if requested and detection was successful
    if args.extract and detection_success:
        extraction_success = run_extractor(
            folder_path,
            args.by_camera,
            args.start_index,
            args.end_index
        )
    
    return detection_success, extraction_success

def main():
    parser = argparse.ArgumentParser(description="Batch process multiple video folders")
    
    # Folder selection arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--root', help='Root folder containing multiple video folders')
    group.add_argument('--folders', nargs='+', help='List of specific folders to process')
    
    # Processing mode
    parser.add_argument('--detect', action='store_true', help='Run audio peak detection')
    parser.add_argument('--extract', action='store_true', help='Run frame extraction')
    
    # Detector options
    parser.add_argument('--frames', type=int, default=300, help='Number of frames to extract per video')
    parser.add_argument('--analyze-percent', type=float, default=100.0, 
                      help='Only analyze the first X percent of each video')
    
    # Extractor options
    parser.add_argument('--by-camera', action='store_true', help='Organize output by camera')
    parser.add_argument('--start-index', type=int, default=0, help='Start index for camera selection')
    parser.add_argument('--end-index', type=int, default=None, help='End index for camera selection')
    
    args = parser.parse_args()
    
    # Default to both detection and extraction if neither is specified
    if not args.detect and not args.extract:
        args.detect = True
        args.extract = True
        logging.info("No processing mode specified, defaulting to both detection and extraction")
    
    start_time = datetime.now()
    
    # Get folders to process
    folders = []
    if args.root:
        if not os.path.isdir(args.root):
            logging.error(f"Root folder not found: {args.root}")
            return
            
        folders = find_video_folders(args.root)
        logging.info(f"Found {len(folders)} video folders in {args.root}")
    else:
        for folder in args.folders:
            if os.path.isdir(folder):
                if is_video_folder(folder):
                    folders.append(folder)
                else:
                    logging.warning(f"Folder does not contain videos: {folder}")
            else:
                logging.warning(f"Folder not found: {folder}")
        
        logging.info(f"Processing {len(folders)} specified folders")
    
    if not folders:
        logging.error("No valid video folders found to process")
        return
    
    # Process each folder
    successful_detections = 0
    successful_extractions = 0
    
    for folder in folders:
        detection_success, extraction_success = process_folder(folder, args)
        
        if detection_success:
            successful_detections += 1
        
        if extraction_success:
            successful_extractions += 1
    
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print("\nBATCH PROCESSING COMPLETE")
    print("=" * 65)
    print(f"Total folders processed: {len(folders)}")
    
    if args.detect:
        print(f"Successful detections: {successful_detections}/{len(folders)}")
    
    if args.extract:
        print(f"Successful extractions: {successful_extractions}/{len(folders)}")
    
    print(f"Total processing time: {total_time}")


if __name__ == "__main__":
    main()
