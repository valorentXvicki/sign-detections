"""
Utility functions for YOLO Sign Detection
"""

import os
from pathlib import Path
import json
from datetime import datetime
import cv2


class Logger:
    """Simple logger for detection events"""
    
    def __init__(self, log_file='detection.log'):
        self.log_file = log_file
    
    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')


class ResultsManager:
    """Manage detection results and statistics"""
    
    def __init__(self, output_dir='results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_detection_stats(self, filename, stats):
        """Save detection statistics to JSON"""
        output_path = self.output_dir / f"{Path(filename).stem}_stats.json"
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return output_path
    
    def save_annotated_image(self, image, output_name):
        """Save annotated image"""
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), image)
        return output_path


def validate_model_path(model_path):
    """Validate if model file exists"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return True


def validate_source(source):
    """Validate detection source"""
    if source == '0':
        return 'webcam'
    
    if Path(source).exists():
        ext = Path(source).suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return 'image'
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return 'video'
    
    raise ValueError(f"Invalid source: {source}")


def get_frame_info(cap):
    """Get information about video capture object"""
    return {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
