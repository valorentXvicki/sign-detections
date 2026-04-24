import argparse
import os
import sys
import cv2
from pathlib import Path
import torch
import numpy as np

# Set root directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO

class SignDetector:
    def __init__(self, weights='runs/detect/sign_detection/weights/best.pt', device=''):
        """Initialize the sign detector with a trained model"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(weights)
        self.model.to(self.device)
        self.class_names = {0: 'Stop', 1: 'Speed Limit', 2: 'Yield', 3: 'No Entry'}
        
    def detect_image(self, image_path, conf=0.5, save_result=True):
        """Detect signs in a single image"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        results = self.model.predict(image, conf=conf)
        
        for result in results:
            annotated_image = result.plot()
            
            if save_result:
                output_path = Path(image_path).stem + '_detected.jpg'
                cv2.imwrite(output_path, annotated_image)
                print(f"Detection result saved to {output_path}")
            
            cv2.imshow('Detection Result', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Print detection details
            self._print_detections(result)
        
        return annotated_image
    
    def detect_video(self, video_path, conf=0.5, save_result=True):
        """Detect signs in a video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video from {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if save_result:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = Path(video_path).stem + '_detected.mp4'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detections_summary = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model.predict(frame, conf=conf)
            
            for result in results:
                annotated_frame = result.plot()
                
                # Count detections per class
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.class_names.get(class_id, 'Unknown')
                        detections_summary[class_name] = detections_summary.get(class_name, 0) + 1
                
                frame = annotated_frame
            
            if save_result:
                out.write(frame)
            
            cv2.imshow('Video Detection', frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if save_result:
            out.release()
            print(f"Video detection result saved to {output_path}")
        
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n=== Detection Summary ===")
        for sign, count in detections_summary.items():
            print(f"{sign}: {count}")
    
    def detect_webcam(self, conf=0.5, save_result=False):
        """Real-time detection from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if save_result:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = 'webcam_detection.mp4'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Starting webcam detection. Press 'q' to quit.")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model.predict(frame, conf=conf)
            
            for result in results:
                annotated_frame = result.plot()
                frame = annotated_frame
            
            if save_result:
                out.write(frame)
            
            cv2.imshow('Webcam Detection', frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if save_result:
            out.release()
            print(f"Webcam detection saved to {output_path}")
        
        cv2.destroyAllWindows()
    
    def _print_detections(self, result):
        """Print detection details"""
        print("\n=== Detections ===")
        if len(result.boxes) == 0:
            print("No signs detected")
            return
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.class_names.get(class_id, 'Unknown')
            xyxy = box.xyxy[0].tolist()
            print(f"Class: {class_name}, Confidence: {confidence:.2f}, Location: {xyxy}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Road Sign Detection using YOLOv8')
    
    parser.add_argument('--weights', type=str, default='runs/detect/sign_detection/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--source', type=str, default='0',
                        help='Source: 0 for webcam, path to image, or path to video')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save', action='store_true',
                        help='Save detection results')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Initialize detector
    detector = SignDetector(weights=args.weights, device=args.device)
    
    # Determine source type and run detection
    source = args.source
    
    if source == '0':
        print("Starting webcam detection...")
        detector.detect_webcam(conf=args.conf, save_result=args.save)
    
    elif os.path.isfile(source):
        file_ext = Path(source).suffix.lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f"Detecting signs in image: {source}")
            detector.detect_image(source, conf=args.conf, save_result=args.save)
        
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"Detecting signs in video: {source}")
            detector.detect_video(source, conf=args.conf, save_result=args.save)
        
        else:
            print(f"Unsupported file format: {file_ext}")
    
    else:
        print(f"Error: Source not found: {source}")
        print("Usage:")
        print("  - Webcam: python main.py --source 0")
        print("  - Image: python main.py --source path/to/image.jpg")
        print("  - Video: python main.py --source path/to/video.mp4")

if __name__ == "__main__":
    main()
