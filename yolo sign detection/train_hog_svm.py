"""
Training pipeline for HOG + SVM classifier
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Tuple, List

from hog_features import HOGFeatureExtractor, MultiScaleHOG
from svm_classifier import SVMClassifier, HOGSVMDetector, train_hog_svm_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HOGSVMTrainer:
    """Trainer for HOG + SVM classifier"""
    
    def __init__(self, positive_dir: str, negative_dir: str,
                 window_size: Tuple[int, int] = (64, 128)):
        """
        Initialize trainer
        
        Args:
            positive_dir: Directory with positive samples
            negative_dir: Directory with negative samples
            window_size: Window size for HOG
        """
        self.positive_dir = Path(positive_dir)
        self.negative_dir = Path(negative_dir)
        self.window_size = window_size
        
        self.hog = HOGFeatureExtractor(win_size=window_size)
        self.classifier = None
        self.metrics = None
        
        logger.info(f"Trainer initialized with window size: {window_size}")
    
    def load_images(self, directory: str, limit: int = None) -> List[np.ndarray]:
        """
        Load images from directory
        
        Args:
            directory: Directory path
            limit: Limit number of images (None for all)
            
        Returns:
            List of images
        """
        dir_path = Path(directory)
        images = []
        
        for ext in ['.jpg', '.png', '.bmp']:
            for img_file in sorted(dir_path.glob(f'*{ext}'))[:limit]:
                image = cv2.imread(str(img_file))
                if image is not None:
                    images.append(image)
        
        logger.info(f"Loaded {len(images)} images from {directory}")
        return images
    
    def train(self, kernel: str = 'rbf', C: float = 1.0,
              test_size: float = 0.2, 
              positive_limit: int = None,
              negative_limit: int = None) -> Dict:
        """
        Train HOG + SVM classifier
        
        Args:
            kernel: SVM kernel type
            C: Regularization parameter
            test_size: Test set ratio
            positive_limit: Limit positive samples
            negative_limit: Limit negative samples
            
        Returns:
            Training metrics
        """
        logger.info("Loading training data...")
        
        # Load images
        positive_images = self.load_images(str(self.positive_dir), positive_limit)
        negative_images = self.load_images(str(self.negative_dir), negative_limit)
        
        if not positive_images or not negative_images:
            logger.error("Could not load training data")
            return {}
        
        logger.info(f"Positive samples: {len(positive_images)}, "
                   f"Negative samples: {len(negative_images)}")
        
        # Train detector
        logger.info("Training detector...")
        self.classifier, self.metrics = train_hog_svm_detector(
            positive_images,
            negative_images,
            self.hog,
            kernel=kernel,
            test_size=test_size
        )
        
        return self.metrics
    
    def evaluate_on_directory(self, test_dir: str, label: int) -> Dict:
        """
        Evaluate classifier on directory
        
        Args:
            test_dir: Directory with test images
            label: Expected label
            
        Returns:
            Evaluation metrics
        """
        if self.classifier is None:
            logger.error("Classifier not trained")
            return {}
        
        logger.info(f"Evaluating on {test_dir}")
        
        # Load images
        images = self.load_images(test_dir)
        
        # Extract features
        X = self.hog.extract_batch(images)
        y = np.ones(len(X), dtype=int) * label
        
        # Evaluate
        metrics = self.classifier.evaluate(X, y)
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def save_model(self, output_dir: str):
        """Save trained model"""
        if self.classifier is None:
            logger.error("No model to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.classifier.save(
            str(output_path / 'svm_model.pkl'),
            str(output_path / 'scaler.pkl')
        )
        
        logger.info(f"Model saved to {output_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train HOG + SVM classifier')
    
    parser.add_argument('positive_dir', help='Directory with positive samples')
    parser.add_argument('negative_dir', help='Directory with negative samples')
    parser.add_argument('--output', default='hog_svm_model', help='Output directory')
    parser.add_argument('--kernel', default='rbf', help='SVM kernel (linear/rbf/poly)')
    parser.add_argument('--C', type=float, default=1.0, help='SVM C parameter')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--pos-limit', type=int, default=None, help='Limit positive samples')
    parser.add_argument('--neg-limit', type=int, default=None, help='Limit negative samples')
    parser.add_argument('--window-size', nargs=2, type=int, default=[64, 128],
                       help='Window size (height width)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create trainer
    trainer = HOGSVMTrainer(
        args.positive_dir,
        args.negative_dir,
        window_size=tuple(args.window_size)
    )
    
    # Train
    metrics = trainer.train(
        kernel=args.kernel,
        C=args.C,
        test_size=args.test_size,
        positive_limit=args.pos_limit,
        negative_limit=args.neg_limit
    )
    
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Save model
    trainer.save_model(args.output)
