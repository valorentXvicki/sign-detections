"""
Generic object detection module using YOLO framework
"""

import argparse
import cv2
from pathlib import Path
from models import ObjectDetector
from config import CLASS_NAMES
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenericObjectDetector:
    """Generic object detector for any object type"""
    
    def __init__(self, weights_path=None, model_size='n', device=''):
        """Initialize detector"""
        self.detector = ObjectDetector(weights_path=weights_path, model_size=model_size, device=device)
        self.class_names = CLASS_NAMES
    
    def detect_image(self, image_path, conf=0.5, save_result=True, display=True):
        """Detect objects in image"""
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None
        
        logger.info(f"Processing image: {Path(image_path).name}")
        results = self.detector.detect(image, conf=conf)
        
        if results:
            detections = self.detector.get_detections()
            annotated_image = self.detector.get_annotated_image()
            
            logger.info(f"Found {len(detections)} objects")
            
            if save_result:
                output_path = Path(image_path).stem + '_objects.jpg'
                cv2.imwrite(output_path, annotated_image)
                logger.info(f"Result saved to: {output_path}")
            
            if display:
                cv2.imshow('Object Detection', annotated_image)
                logger.info("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            self.detector.summary()
            return annotated_image
        
        return None
    
    def detect_video(self, video_path, conf=0.5, save_result=True, display=True):
        """Detect objects in video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video info - FPS: {fps}, Resolution: {width}x{height}")
        
        if save_result:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = Path(video_path).stem + '_objects.mp4'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = {}
        
        logger.info("Processing video frames...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.detector.detect(frame, conf=conf)
            
            if results:
                detections = self.detector.get_detections()
                
                for d in detections:
                    class_name = d['class_name']
                    total_detections[class_name] = total_detections.get(class_name, 0) + 1
                
                frame = self.detector.get_annotated_image()
            
            if save_result:
                out.write(frame)
            
            if display:
                cv2.imshow('Video Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames...")
        
        cap.release()
        if save_result:
            out.release()
            logger.info(f"Video saved to: {output_path}")
        
        if display:
            cv2.destroyAllWindows()
        
        logger.info(f"\nTotal frames: {frame_count}")
        logger.info("Objects detected per class:")
        for class_name, count in sorted(total_detections.items()):
            logger.info(f"  {class_name}: {count}")
    
    def detect_webcam(self, conf=0.5, save_result=False):
        """Real-time detection from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if save_result:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = 'webcam_objects.mp4'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info("Webcam detection started. Press 'q' to quit.")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.detector.detect(frame, conf=conf)
            
            if results:
                frame = self.detector.get_annotated_image()
            
            if save_result:
                out.write(frame)
            
            cv2.imshow('Webcam Object Detection - Press q to quit', frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if save_result:
            out.release()
            logger.info(f"Recording saved to: {output_path}")
        
        cv2.destroyAllWindows()
        logger.info(f"Total frames processed: {frame_count}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generic Object Detection using YOLO')
    parser.add_argument('--source', type=str, default='0',
                        help='Source: 0 for webcam, image path, or video path')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to custom weights')
    parser.add_argument('--model', type=str, default='n',
                        help='Model size: n/s/m/l/x')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='',
                        help='Device: cuda or cpu')
    parser.add_argument('--save', action='store_true',
                        help='Save results')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display results')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    detector = GenericObjectDetector(
        weights_path=args.weights,
        model_size=args.model,
        device=args.device
    )
    
    source = args.source
    
    if source == '0':
        logger.info("Starting webcam detection...")
        detector.detect_webcam(conf=args.conf, save_result=args.save)
    
    elif Path(source).exists():
        file_ext = Path(source).suffix.lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            logger.info(f"Detecting objects in image: {source}")
            detector.detect_image(source, conf=args.conf, save_result=args.save, display=not args.no_display)
        
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            logger.info(f"Detecting objects in video: {source}")
            detector.detect_video(source, conf=args.conf, save_result=args.save, display=not args.no_display)
        
        else:
            logger.error(f"Unsupported file format: {file_ext}")
    
    else:
        logger.error(f"Source not found: {source}")
        print("\nUsage:")
        print("  Webcam:  python object_detector.py --source 0")
        print("  Image:   python object_detector.py --source image.jpg")
        print("  Video:   python object_detector.py --source video.mp4")


if __name__ == "__main__":
    main()
