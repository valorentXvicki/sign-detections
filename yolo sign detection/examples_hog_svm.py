"""
Examples for HOG + SVM feature extraction and classification
"""

import cv2
import numpy as np
from pathlib import Path

from hog_features import HOGFeatureExtractor, MultiScaleHOG, compute_hog_histogram
from svm_classifier import SVMClassifier, train_hog_svm_detector


def example_1_basic_hog_extraction():
    """Example 1: Basic HOG feature extraction"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic HOG Feature Extraction")
    print("="*60)
    
    # Initialize HOG
    hog = HOGFeatureExtractor(win_size=(64, 128))
    
    # Load image
    image = cv2.imread('image.jpg')
    if image is None:
        print("Image not found")
        return
    
    # Extract features
    features = hog.extract(image)
    print(f"Extracted {len(features)} features")
    print(f"Feature vector: {features[:10]}")  # First 10 features
    
    # Get feature dimension
    dim = hog.get_feature_dimension()
    print(f"Feature dimension: {dim}")


def example_2_batch_extraction():
    """Example 2: Batch HOG extraction"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch HOG Extraction")
    print("="*60)
    
    hog = HOGFeatureExtractor(win_size=(64, 128))
    
    # Load images
    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    images = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    
    if not images:
        print("No images found")
        return
    
    # Extract batch features
    features = hog.extract_batch(images)
    print(f"Extracted features from {len(features)} images")
    print(f"Features shape: {features.shape}")


def example_3_multiscale_hog():
    """Example 3: Multi-scale HOG extraction"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Multi-scale HOG Extraction")
    print("="*60)
    
    # Initialize multi-scale HOG
    scales = [(32, 64), (64, 128), (128, 256)]
    ms_hog = MultiScaleHOG(scales=scales)
    
    # Load image
    image = cv2.imread('image.jpg')
    if image is None:
        print("Image not found")
        return
    
    # Extract multi-scale features
    features = ms_hog.extract(image)
    print(f"Multi-scale features dimension: {len(features)}")


def example_4_hog_visualization():
    """Example 4: Visualize HOG features"""
    print("\n" + "="*60)
    print("EXAMPLE 4: HOG Visualization")
    print("="*60)
    
    hog = HOGFeatureExtractor(win_size=(64, 128))
    
    # Load image
    image = cv2.imread('image.jpg')
    if image is None:
        print("Image not found")
        return
    
    # Visualize HOG
    hog_viz = hog.visualize_hog(image, output_path='hog_visualization.jpg')
    print("HOG visualization saved to hog_visualization.jpg")


def example_5_svm_training():
    """Example 5: Train SVM classifier"""
    print("\n" + "="*60)
    print("EXAMPLE 5: SVM Training")
    print("="*60)
    
    # Create synthetic training data
    n_samples = 100
    n_features = 3780  # HOG feature dimension
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    # Initialize SVM
    svm = SVMClassifier(kernel='rbf', C=1.0)
    
    # Train
    svm.fit(X_train, y_train)
    print("SVM training completed")
    
    # Test
    X_test = np.random.randn(20, n_features)
    y_test = np.random.randint(0, 2, 20)
    
    # Evaluate
    metrics = svm.evaluate(X_test, y_test)
    print(f"Evaluation metrics: {metrics}")


def example_6_hog_svm_detector():
    """Example 6: Train HOG + SVM detector"""
    print("\n" + "="*60)
    print("EXAMPLE 6: HOG + SVM Detector Training")
    print("="*60)
    
    # Create synthetic training data
    hog = HOGFeatureExtractor(win_size=(64, 128))
    
    # Positive samples (synthetic)
    pos_images = [np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8) 
                  for _ in range(50)]
    
    # Negative samples (synthetic)
    neg_images = [np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8) 
                  for _ in range(50)]
    
    # Train detector
    classifier, metrics = train_hog_svm_detector(
        pos_images, neg_images, hog, kernel='rbf'
    )
    
    print(f"Training completed")
    print(f"Metrics: {metrics}")


def example_7_cross_validation():
    """Example 7: Cross-validation"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Cross-Validation")
    print("="*60)
    
    # Create synthetic data
    n_samples = 100
    n_features = 3780
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Initialize SVM
    svm = SVMClassifier(kernel='rbf')
    
    # Cross-validate
    cv_results = svm.cross_validate(X, y, cv=5)
    
    print(f"Cross-validation results:")
    print(f"  Mean: {cv_results['mean']:.4f}")
    print(f"  Std: {cv_results['std']:.4f}")
    print(f"  Scores: {cv_results['scores']}")


def example_8_hog_histogram():
    """Example 8: HOG histogram computation"""
    print("\n" + "="*60)
    print("EXAMPLE 8: HOG Histogram Computation")
    print("="*60)
    
    # Load image
    image = cv2.imread('image.jpg')
    if image is None:
        print("Image not found")
        return
    
    # Compute histogram
    histogram, shape = compute_hog_histogram(image, cell_size=8, nbins=9)
    
    print(f"Histogram shape: {histogram.shape}")
    print(f"Grid shape: {shape}")
    print(f"Sample histogram (first cell): {histogram[0, 0, :]}")


if __name__ == "__main__":
    print("\nHOG + SVM Feature Extraction Examples\n")
    print("Available examples:")
    print("1. example_1_basic_hog_extraction() - Basic HOG extraction")
    print("2. example_2_batch_extraction() - Batch HOG extraction")
    print("3. example_3_multiscale_hog() - Multi-scale HOG")
    print("4. example_4_hog_visualization() - HOG visualization")
    print("5. example_5_svm_training() - SVM classifier training")
    print("6. example_6_hog_svm_detector() - HOG + SVM detector")
    print("7. example_7_cross_validation() - Cross-validation")
    print("8. example_8_hog_histogram() - HOG histogram computation")
    print("\nRun any example:")
    print("python examples_hog_svm.py")
