"""
YOLO Framework - Complete object detection pipeline
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import cv2
from dataclasses import dataclass
from models import YOLOModel, ObjectDetector
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Data class for a single detection"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    area: float
    
    def to_dict(self):
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'area': self.area
        }


@dataclass
class DetectionResult:
    """Data class for detection results"""
    source: str
    detections: List[Detection]
    image_shape: Tuple[int, int, int]
    timestamp: str
    
    def to_dict(self):
        return {
            'source': self.source,
            'num_detections': len(self.detections),
            'detections': [d.to_dict() for d in self.detections],
            'image_shape': self.image_shape,
            'timestamp': self.timestamp
        }


class YOLOFramework:
    """Complete YOLO object detection framework"""
    
    def __init__(self, model_size='n', device='', custom_weights: Optional[str] = None):
        """
        Initialize YOLO framework
        
        Args:
            model_size: Model size (n/s/m/l/x)
            device: Device (cuda/cpu)
            custom_weights: Path to custom weights
        """
        self.model_size = model_size
        self.device = device
        self.detector = ObjectDetector(
            weights_path=custom_weights,
            model_size=model_size,
            device=device
        )
        self.results_history = []
        
        logger.info(f"YOLO Framework initialized with model size: {model_size}")
    
    def detect(self, source, conf=0.5, iou=0.45) -> Optional[DetectionResult]:
        """
        Run detection on source
        
        Args:
            source: Image array or file path
            conf: Confidence threshold
            iou: IOU threshold
            
        Returns:
            DetectionResult object
        """
        try:
            # Handle different source types
            if isinstance(source, str):
                if not Path(source).exists():
                    logger.error(f"Source file not found: {source}")
                    return None
                source_name = Path(source).name
            else:
                source_name = "array"
            
            logger.info(f"Running detection on: {source_name}")
            
            # Run inference
            self.detector.detect(source, conf=conf, iou=iou)
            detections_raw = self.detector.get_detections()
            
            # Convert to Detection objects
            detections = [
                Detection(
                    class_id=d['class_id'],
                    class_name=d['class_name'],
                    confidence=d['confidence'],
                    bbox=d['bbox'],
                    area=d['area']
                )
                for d in detections_raw
            ]
            
            # Get image shape
            result = self.detector.results[0]
            image_shape = result.orig_img.shape
            
            # Create result
            from datetime import datetime
            result_obj = DetectionResult(
                source=source_name,
                detections=detections,
                image_shape=image_shape,
                timestamp=datetime.now().isoformat()
            )
            
            self.results_history.append(result_obj)
            
            logger.info(f"Detection complete: {len(detections)} objects found")
            return result_obj
        
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return None
    
    def batch_detect(self, sources: List[str], conf=0.5) -> List[DetectionResult]:
        """
        Run detection on multiple sources
        
        Args:
            sources: List of source paths
            conf: Confidence threshold
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        for i, source in enumerate(sources):
            logger.info(f"Processing {i+1}/{len(sources)}: {source}")
            result = self.detect(source, conf=conf)
            if result:
                results.append(result)
        
        return results
    
    def filter_detections(self, result: DetectionResult, min_conf=0.5, 
                         min_area=0, class_names: Optional[List[str]] = None) -> List[Detection]:
        """
        Filter detections from result
        
        Args:
            result: DetectionResult object
            min_conf: Minimum confidence
            min_area: Minimum area
            class_names: Filter by specific classes
            
        Returns:
            Filtered detections
        """
        filtered = result.detections
        
        if min_conf > 0:
            filtered = [d for d in filtered if d.confidence >= min_conf]
        
        if min_area > 0:
            filtered = [d for d in filtered if d.area >= min_area]
        
        if class_names:
            filtered = [d for d in filtered if d.class_name in class_names]
        
        return filtered
    
    def get_statistics(self, result: DetectionResult) -> Dict:
        """
        Get detection statistics
        
        Args:
            result: DetectionResult object
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_objects': len(result.detections),
            'by_class': {},
            'confidence_stats': {
                'min': 0,
                'max': 0,
                'mean': 0
            },
            'area_stats': {
                'min': 0,
                'max': 0,
                'mean': 0
            }
        }
        
        if not result.detections:
            return stats
        
        # Count by class
        for det in result.detections:
            class_name = det.class_name
            if class_name not in stats['by_class']:
                stats['by_class'][class_name] = 0
            stats['by_class'][class_name] += 1
        
        # Confidence statistics
        confidences = [d.confidence for d in result.detections]
        stats['confidence_stats'] = {
            'min': float(min(confidences)),
            'max': float(max(confidences)),
            'mean': float(sum(confidences) / len(confidences))
        }
        
        # Area statistics
        areas = [d.area for d in result.detections]
        stats['area_stats'] = {
            'min': float(min(areas)),
            'max': float(max(areas)),
            'mean': float(sum(areas) / len(areas))
        }
        
        return stats
    
    def get_annotated_image(self) -> Optional[np.ndarray]:
        """Get annotated image from last detection"""
        if not self.detector.results:
            return None
        return self.detector.get_annotated_image()
    
    def save_result(self, result: DetectionResult, output_dir: str = 'results'):
        """
        Save detection result
        
        Args:
            result: DetectionResult object
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save JSON
        json_file = output_path / f"{Path(result.source).stem}_detections.json"
        with open(json_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to: {json_file}")
    
    def get_history(self) -> List[DetectionResult]:
        """Get detection history"""
        return self.results_history
    
    def clear_history(self):
        """Clear detection history"""
        self.results_history = []
        logger.info("Detection history cleared")
    
    def export_statistics(self, output_file: str = 'detection_stats.json'):
        """
        Export statistics for all detections
        
        Args:
            output_file: Output file path
        """
        stats_list = [
            {
                'source': result.source,
                'timestamp': result.timestamp,
                'statistics': self.get_statistics(result)
            }
            for result in self.results_history
        ]
        
        with open(output_file, 'w') as f:
            json.dump(stats_list, f, indent=2)
        
        logger.info(f"Statistics exported to: {output_file}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        info = self.detector.yolo.get_model_info()
        info['framework'] = 'YOLOv8'
        info['num_detections_total'] = sum(
            len(r.detections) for r in self.results_history
        )
        return info
