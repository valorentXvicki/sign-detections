"""
HOG (Histogram of Oriented Gradients) Feature Extraction Module
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HOGFeatureExtractor:
    """HOG Feature Extractor for object detection"""
    
    def __init__(self, win_size: Tuple[int, int] = (64, 128), 
                 block_size: Tuple[int, int] = (16, 16),
                 block_stride: Tuple[int, int] = (8, 8),
                 cell_size: Tuple[int, int] = (8, 8),
                 nbins: int = 9):
        """
        Initialize HOG descriptor
        
        Args:
            win_size: Window size for detection
            block_size: Block size for HOG calculation
            block_stride: Block stride
            cell_size: Cell size in pixels
            nbins: Number of orientation bins
        """
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins
        
        # Create HOG descriptor
        self.hog = cv2.HOGDescriptor(
            win_size, 
            block_size, 
            block_stride, 
            cell_size, 
            nbins
        )
        
        logger.info(f"HOG initialized with window size: {win_size}, "
                   f"block size: {block_size}, cells: {nbins} bins")
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from image
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            HOG feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to window size
        if image.shape != self.win_size:
            image = cv2.resize(image, self.win_size)
        
        # Extract HOG features
        try:
            features = self.hog.compute(image)
            features = features.flatten()
            logger.debug(f"Extracted {len(features)} HOG features")
            return features
        except Exception as e:
            logger.error(f"Failed to extract HOG features: {e}")
            return np.array([])
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract HOG features from batch of images
        
        Args:
            images: List of images
            
        Returns:
            Array of feature vectors (n_samples, n_features)
        """
        features_list = []
        
        for i, image in enumerate(images):
            features = self.extract(image)
            if len(features) > 0:
                features_list.append(features)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(images)} images")
        
        if not features_list:
            logger.warning("No features extracted")
            return np.array([])
        
        features_array = np.array(features_list)
        logger.info(f"Extracted features from {len(features_list)} images. "
                   f"Shape: {features_array.shape}")
        return features_array
    
    def extract_from_files(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract HOG features from image files
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Array of feature vectors
        """
        images = []
        valid_paths = []
        
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is not None:
                images.append(image)
                valid_paths.append(path)
            else:
                logger.warning(f"Could not load image: {path}")
        
        logger.info(f"Loaded {len(images)}/{len(image_paths)} images")
        
        features = self.extract_batch(images)
        return features
    
    def extract_from_directory(self, directory: str, 
                              extensions: Tuple[str, ...] = ('.jpg', '.png', '.bmp')) -> Tuple[np.ndarray, List[str]]:
        """
        Extract HOG features from all images in directory
        
        Args:
            directory: Directory path
            extensions: Image extensions to process
            
        Returns:
            Tuple of (features array, image paths)
        """
        dir_path = Path(directory)
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(dir_path.glob(f'*{ext}'))
        
        logger.info(f"Found {len(image_paths)} images in {directory}")
        
        features = self.extract_from_files([str(p) for p in image_paths])
        
        return features, [str(p) for p in image_paths]
    
    def visualize_hog(self, image: np.ndarray, 
                     output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize HOG features
        
        Args:
            image: Input image
            output_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        
        # Resize to window size
        if image_gray.shape != self.win_size:
            image_gray = cv2.resize(image_gray, self.win_size)
        
        # Compute HOG
        hog_features = self.hog.compute(image_gray)
        
        # Create visualization
        h, w = image_gray.shape
        hog_viz = np.zeros((h, w), dtype=np.uint8)
        
        # Draw gradients on image
        cell_size_x, cell_size_y = self.cell_size
        bin_size = 180 / self.nbins
        
        for y in range(0, h, cell_size_y):
            for x in range(0, w, cell_size_x):
                cell = image_gray[y:y+cell_size_y, x:x+cell_size_x]
                
                # Compute gradients
                gx = cv2.Sobel(cell, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(cell, cv2.CV_32F, 0, 1, ksize=3)
                
                magnitude = np.sqrt(gx**2 + gy**2)
                hog_viz[y:y+cell_size_y, x:x+cell_size_x] = np.uint8(
                    np.mean(magnitude)
                )
        
        # Normalize and convert to color
        hog_viz = cv2.normalize(hog_viz, None, 0, 255, cv2.NORM_MINMAX)
        hog_color = cv2.applyColorMap(hog_viz, cv2.COLORMAP_JET)
        
        if output_path:
            cv2.imwrite(output_path, hog_color)
            logger.info(f"HOG visualization saved to {output_path}")
        
        return hog_color
    
    def get_feature_dimension(self) -> int:
        """Get dimension of feature vector"""
        dummy_image = np.zeros(self.win_size, dtype=np.uint8)
        features = self.hog.compute(dummy_image)
        return len(features.flatten())


class MultiScaleHOG:
    """Multi-scale HOG feature extraction"""
    
    def __init__(self, scales: List[Tuple[int, int]] = None):
        """
        Initialize multi-scale HOG
        
        Args:
            scales: List of window sizes for different scales
        """
        if scales is None:
            scales = [(32, 64), (64, 128), (128, 256)]
        
        self.extractors = [HOGFeatureExtractor(win_size=s) for s in scales]
        logger.info(f"Initialized multi-scale HOG with {len(self.extractors)} scales")
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract multi-scale HOG features
        
        Args:
            image: Input image
            
        Returns:
            Concatenated feature vector from all scales
        """
        features_list = []
        
        for i, extractor in enumerate(self.extractors):
            features = extractor.extract(image)
            features_list.append(features)
        
        # Concatenate all features
        combined_features = np.concatenate(features_list)
        logger.debug(f"Combined features dimension: {len(combined_features)}")
        
        return combined_features
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract multi-scale HOG from batch"""
        features_list = []
        
        for image in images:
            features = self.extract(image)
            features_list.append(features)
        
        return np.array(features_list)


def compute_hog_histogram(image: np.ndarray, 
                         cell_size: int = 8,
                         nbins: int = 9) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Compute HOG histogram for image
    
    Args:
        image: Input image
        cell_size: Size of each cell
        nbins: Number of bins
        
    Returns:
        Tuple of (histogram, shape)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute gradients
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute magnitude and angle
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 180
    
    # Compute histogram
    h, w = image.shape
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    
    histogram = np.zeros((n_cells_y, n_cells_x, nbins))
    
    for y in range(n_cells_y):
        for x in range(n_cells_x):
            cell_angle = angle[y*cell_size:(y+1)*cell_size, 
                              x*cell_size:(x+1)*cell_size]
            cell_mag = magnitude[y*cell_size:(y+1)*cell_size, 
                                x*cell_size:(x+1)*cell_size]
            
            # Create histogram
            for i in range(nbins):
                lower = i * 180 / nbins
                upper = (i + 1) * 180 / nbins
                
                mask = (cell_angle >= lower) & (cell_angle < upper)
                histogram[y, x, i] = np.sum(cell_mag[mask])
    
    return histogram, (n_cells_y, n_cells_x)
