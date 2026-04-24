"""
Detection using trained HOG + SVM model
"""

import cv2
import argparse
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Optional

from hog_features import HOGFeatureExtractor
from svm_classifier import SVMClassifier, HOGSVMDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HOGSVMPredictor:
    """Predictor using trained HOG + SVM model"""
    
    def __init__(self, model_dir: str, window_size: Tuple[int, int] = (64, 128)):
        """
        Initialize predictor
        
        Args:
            model_dir: Directory with trained model
            window_size: Window size used in training
        """
        self.model_dir = Path(model_dir)
        self.window_size = window_size
        
        # Initialize HOG
        self.hog = HOGFeatureExtractor(win_size=window_size)
        
        # Initialize SVM
        self.svm = SVMClassifier()
        
        # Load model
        self.load_model()
        
        logger.info("Predictor initialized")
    
    def load_model(self):
        """Load trained model"""
        model_path = self.model_dir / 'svm_model.pkl'
        scaler_path = self.model_dir / 'scaler.pkl'
        
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return
        
        self.svm.load(str(model_path), str(scaler_path))
        logger.info(f"Model loaded from {self.model_dir}")
    
    def detect_image(self, image_path: str) -> Tuple[int, float]:
        """
        Detect object in image
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (class_label, confidence)
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None, 0.0
        
        # Extract features
        features = self.hog.extract(image).reshape(1, -1)
        
        # Predict
        label = self.svm.predict(features)[0]
        
        # Get probability
        try:
            proba = self.svm.predict_proba(features)
            confidence = np.max(proba)
        except:
            confidence = 1.0
        
        logger.info(f"Detection: label={label}, confidence={confidence:.4f}")
        return label, confidence
    
    def detect_batch(self, image_paths: List[str]) -> List[Tuple[int, float]]:
        """
        Detect objects in batch of images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of (label, confidence) tuples
        """
        results = []
        
        for i, img_path in enumerate(image_paths):
            label, confidence = self.detect_image(img_path)
            if label is not None:
                results.append((label, confidence))
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_paths)} images")
        
        return results
    
    def detect_directory(self, directory: str) -> List[Tuple[str, int, float]]:
        """
        Detect objects in all images in directory
        
        Args:
            directory: Directory path
            
        Returns:
            List of (filename, label, confidence) tuples
        """
        dir_path = Path(directory)
        results = []
        
        for ext in ['.jpg', '.png', '.bmp']:
            for img_file in sorted(dir_path.glob(f'*{ext}')):
                label, confidence = self.detect_image(str(img_file))
                if label is not None:
                    results.append((img_file.name, label, confidence))
        
        logger.info(f"Processed {len(results)} images from {directory}")
        return results
    
    def detect_video(self, video_path: str, output_path: Optional[str] = None) -> List[Tuple[int, float]]:
        """
        Detect objects in video
        
        Args:
            video_path: Path to video
            output_path: Optional path to save output video
            
        Returns:
            List of (label, confidence) tuples per frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        
        logger.info(f"Processing video: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect in frame
            features = self.hog.extract(frame).reshape(1, -1)
            label = self.svm.predict(features)[0]
            
            try:
                proba = self.svm.predict_proba(features)
                confidence = np.max(proba)
            except:
                confidence = 1.0
            
            results.append((label, confidence))
            
            # Draw detection
            if output_path or True:
                label_text = f"Label: {label}, Conf: {confidence:.2f}"
                cv2.putText(frame, label_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if output_path:
                out.write(frame)
            
            frame_count += 1
            
            if frame_count % fps == 0:
                logger.info(f"Processed {frame_count} frames")
        
        cap.release()
        if output_path:
            out.release()
            logger.info(f"Output video saved to {output_path}")
        
        return results
    
    def detect_webcam(self, confidence_threshold: float = 0.5):
        """
        Real-time detection from webcam
        
        Args:
            confidence_threshold: Minimum confidence for display
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return
        
        logger.info("Webcam detection started. Press 'q' to quit.")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect
            features = self.hog.extract(frame).reshape(1, -1)
            label = self.svm.predict(features)[0]
            
            try:
                proba = self.svm.predict_proba(features)
                confidence = np.max(proba)
            except:
                confidence = 1.0
            
            # Display
            if confidence >= confidence_threshold:
                label_text = f"Label: {label}, Conf: {confidence:.2f}"
                cv2.putText(frame, label_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('HOG + SVM Detection', frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Processed {frame_count} frames from webcam")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Detect using HOG + SVM')
    
    parser.add_argument('model_dir', help='Directory with trained model')
    parser.add_argument('--source', type=str, default='0',
                       help='Source: 0 (webcam), image path, video path, or directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (for video input)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--window-size', nargs=2, type=int, default=[64, 128],
                       help='Window size (height width)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Initialize predictor
    predictor = HOGSVMPredictor(
        args.model_dir,
        window_size=tuple(args.window_size)
    )
    
    source = args.source
    
    if source == '0':
        # Webcam
        predictor.detect_webcam(confidence_threshold=args.conf_threshold)
    
    elif Path(source).is_file():
        # Image or video file
        file_ext = Path(source).suffix.lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Image
            label, confidence = predictor.detect_image(source)
            print(f"Detection: label={label}, confidence={confidence:.4f}")
        
        elif file_ext in ['.mp4', '.avi', '.mov']:
            # Video
            results = predictor.detect_video(source, args.output)
            print(f"Processed {len(results)} frames")
    
    elif Path(source).is_dir():
        # Directory
        results = predictor.detect_directory(source)
        
        print("\n" + "="*60)
        print("DETECTION RESULTS")
        print("="*60)
        
        for filename, label, confidence in results:
            print(f"{filename}: label={label}, confidence={confidence:.4f}")
    
    else:
        print(f"Source not found: {source}")


if __name__ == "__main__":
    main()
