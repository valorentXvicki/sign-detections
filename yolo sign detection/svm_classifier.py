"""
SVM Classifier for object detection using HOG features
"""

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import joblib
from pathlib import Path
import logging
from typing import Tuple, List, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVMClassifier:
    """SVM Classifier for object detection"""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, 
                 gamma: str = 'scale', class_weight: Optional[str] = None):
        """
        Initialize SVM classifier
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
            class_weight: Weight classes to handle imbalance
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        
        # Initialize SVM
        if kernel == 'linear':
            self.svm = LinearSVC(C=C, max_iter=2000, class_weight=class_weight)
        else:
            self.svm = SVC(kernel=kernel, C=C, gamma=gamma, 
                          probability=True, class_weight=class_weight)
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_dimension = None
        self.classes = None
        
        logger.info(f"SVM initialized with kernel={kernel}, C={C}, gamma={gamma}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            normalize: bool = True) -> 'SVMClassifier':
        """
        Train SVM classifier
        
        Args:
            X: Feature vectors (n_samples, n_features)
            y: Labels (n_samples,)
            normalize: Whether to normalize features
            
        Returns:
            Self for chaining
        """
        logger.info(f"Starting SVM training. Data shape: {X.shape}")
        
        # Normalize features
        if normalize:
            X = self.scaler.fit_transform(X)
            logger.info("Features normalized")
        
        # Train SVM
        self.svm.fit(X, y)
        self.is_trained = True
        self.feature_dimension = X.shape[1]
        self.classes = self.svm.classes_
        
        logger.info(f"SVM training completed. Classes: {self.classes}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Feature vectors
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = self.scaler.transform(X)
        predictions = self.svm.predict(X)
        
        logger.debug(f"Made predictions for {len(predictions)} samples")
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature vectors
            
        Returns:
            Probability estimates
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        if not hasattr(self.svm, 'predict_proba'):
            raise RuntimeError("Model does not support probability estimates. "
                             "Use kernel='rbf' or kernel='poly'.")
        
        X = self.scaler.transform(X)
        probabilities = self.svm.predict_proba(X)
        
        return probabilities
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classifier performance
        
        Args:
            X: Feature vectors
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y, predictions, average='weighted', zero_division=0)
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X: Feature vectors
            y: Labels
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        X_scaled = self.scaler.fit_transform(X)
        
        scores = cross_val_score(self.svm, X_scaled, y, cv=cv)
        
        cv_results = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
        
        logger.info(f"Cross-validation scores (cv={cv}): "
                   f"mean={cv_results['mean']:.4f}, std={cv_results['std']:.4f}")
        
        return cv_results
    
    def save(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Save trained model
        
        Args:
            model_path: Path to save SVM model
            scaler_path: Path to save scaler
        """
        if not self.is_trained:
            logger.warning("Model not trained. Cannot save.")
            return
        
        joblib.dump(self.svm, model_path)
        logger.info(f"Model saved to {model_path}")
        
        if scaler_path:
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
    
    def load(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Load trained model
        
        Args:
            model_path: Path to SVM model
            scaler_path: Path to scaler
        """
        self.svm = joblib.load(model_path)
        self.is_trained = True
        
        if Path(model_path).exists():
            logger.info(f"Model loaded from {model_path}")
        
        if scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")


class HOGSVMDetector:
    """Complete HOG + SVM detector"""
    
    def __init__(self, hog_extractor, svm_classifier: SVMClassifier):
        """
        Initialize HOG + SVM detector
        
        Args:
            hog_extractor: HOG feature extractor
            svm_classifier: Trained SVM classifier
        """
        self.hog = hog_extractor
        self.svm = svm_classifier
        logger.info("HOG + SVM detector initialized")
    
    def detect(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Detect object in image
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (class_label, confidence)
        """
        # Extract HOG features
        features = self.hog.extract(image)
        features = features.reshape(1, -1)
        
        # Predict
        label = self.svm.predict(features)[0]
        
        # Get probability if available
        if hasattr(self.svm.svm, 'predict_proba'):
            proba = self.svm.predict_proba(features)
            confidence = np.max(proba)
        else:
            confidence = 1.0
        
        return label, confidence
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Detect objects in batch of images
        
        Args:
            images: List of images
            
        Returns:
            List of (class_label, confidence) tuples
        """
        results = []
        
        for i, image in enumerate(images):
            label, confidence = self.detect(image)
            results.append((label, confidence))
            
            if (i + 1) % 10 == 0:
                logger.info(f"Detected {i + 1}/{len(images)} images")
        
        return results
    
    def save(self, model_dir: str):
        """Save complete detector"""
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        
        self.svm.save(
            str(model_dir / 'svm_model.pkl'),
            str(model_dir / 'scaler.pkl')
        )
        logger.info(f"Detector saved to {model_dir}")
    
    def load(self, model_dir: str):
        """Load complete detector"""
        model_dir = Path(model_dir)
        
        self.svm.load(
            str(model_dir / 'svm_model.pkl'),
            str(model_dir / 'scaler.pkl')
        )
        logger.info(f"Detector loaded from {model_dir}")


def train_hog_svm_detector(positive_images: List[np.ndarray],
                           negative_images: List[np.ndarray],
                           hog_extractor,
                           kernel: str = 'rbf',
                           test_size: float = 0.2) -> Tuple[SVMClassifier, Dict]:
    """
    Train complete HOG + SVM detector
    
    Args:
        positive_images: Positive training samples
        negative_images: Negative training samples
        hog_extractor: HOG feature extractor
        kernel: SVM kernel type
        test_size: Test set ratio
        
    Returns:
        Tuple of (trained classifier, metrics)
    """
    logger.info("Training HOG + SVM detector...")
    
    # Extract features
    logger.info("Extracting positive features...")
    X_pos = hog_extractor.extract_batch(positive_images)
    
    logger.info("Extracting negative features...")
    X_neg = hog_extractor.extract_batch(negative_images)
    
    # Create labels
    y_pos = np.ones(len(X_pos), dtype=int)
    y_neg = np.zeros(len(X_neg), dtype=int)
    
    # Combine
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])
    
    logger.info(f"Total samples: {len(X)}, "
               f"Positive: {np.sum(y_pos)}, Negative: {np.sum(y_neg)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Train SVM
    classifier = SVMClassifier(kernel=kernel)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    metrics = classifier.evaluate(X_test, y_test)
    metrics['train_score'] = classifier.evaluate(X_train, y_train)['accuracy']
    
    logger.info(f"Training completed. "
               f"Train accuracy: {metrics['train_score']:.4f}, "
               f"Test accuracy: {metrics['accuracy']:.4f}")
    
    return classifier, metrics
