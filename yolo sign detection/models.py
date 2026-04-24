"""
YOLO Model wrapper and management
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import logging

logger = logging.getLogger(__name__)


class YOLOModel:
    """YOLO model wrapper for object detection"""
    
    def __init__(self, model_size='n', device=''):
        """
        Initialize YOLO model
        
        Args:
            model_size: Model size (n/s/m/l/x)
            device: Device to use ('cuda', 'cpu', or '' for auto)
        """
        self.model_size = model_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_name = f'yolov8{model_size}.pt'
        logger.info(f"Initializing YOLOv8{model_size} on {self.device}")
    
    def load_pretrained(self):
        """Load pretrained YOLOv8 model"""
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            logger.info(f"Pretrained model {self.model_name} loaded successfully")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_weights(self, weights_path):
        """Load custom trained weights"""
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        try:
            self.model = YOLO(weights_path)
            self.model.to(self.device)
            logger.info(f"Weights loaded from {weights_path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            raise
    
    def predict(self, source, conf=0.5, iou=0.45):
        """
        Run inference on source
        
        Args:
            source: Image, video, or stream
            conf: Confidence threshold
            iou: IOU threshold for NMS
            
        Returns:
            Detection results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_pretrained() or load_weights() first")
        
        try:
            results = self.model.predict(
                source=source,
                conf=conf,
                iou=iou,
                device=self.device
            )
            return results
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def train(self, data, epochs=100, imgsz=640, batch=16, **kwargs):
        """
        Train model
        
        Args:
            data: Path to dataset YAML
            epochs: Number of epochs
            imgsz: Image size
            batch: Batch size
            **kwargs: Additional training arguments
        """
        if self.model is None:
            self.load_pretrained()
        
        try:
            results = self.model.train(
                data=data,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=self.device,
                **kwargs
            )
            logger.info(f"Training completed. Results: {results}")
            return results
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return None
        
        return {
            'size': self.model_size,
            'device': self.device,
            'model_name': self.model_name,
            'parameters': sum(p.numel() for p in self.model.model.parameters()),
        }


class ObjectDetector:
    """High-level object detector using YOLO"""
    
    def __init__(self, weights_path=None, model_size='n', device=''):
        """
        Initialize object detector
        
        Args:
            weights_path: Path to custom weights (optional)
            model_size: Model size if using pretrained
            device: Device to use
        """
        self.yolo = YOLOModel(model_size=model_size, device=device)
        
        if weights_path:
            self.yolo.load_weights(weights_path)
        else:
            self.yolo.load_pretrained()
        
        self.results = None
    
    def detect(self, source, conf=0.5, iou=0.45):
        """
        Detect objects in source
        
        Args:
            source: Image, video, or stream
            conf: Confidence threshold
            iou: IOU threshold
            
        Returns:
            Processed results with annotations
        """
        self.results = self.yolo.predict(source, conf=conf, iou=iou)
        return self.results
    
    def get_detections(self, result_idx=0):
        """
        Get detections from results
        
        Args:
            result_idx: Index of result to process
            
        Returns:
            List of detections with class and confidence
        """
        if self.results is None:
            return []
        
        result = self.results[result_idx]
        detections = []
        
        for box in result.boxes:
            detection = {
                'class_id': int(box.cls[0]),
                'class_name': result.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist(),
                'area': self._calculate_area(box.xyxy[0])
            }
            detections.append(detection)
        
        return detections
    
    def _calculate_area(self, bbox):
        """Calculate bounding box area"""
        x1, y1, x2, y2 = bbox
        return float((x2 - x1) * (y2 - y1))
    
    def filter_detections(self, min_confidence=0.5, min_area=0):
        """
        Filter detections by confidence and area
        
        Args:
            min_confidence: Minimum confidence threshold
            min_area: Minimum bounding box area
            
        Returns:
            Filtered detections
        """
        detections = self.get_detections()
        filtered = [
            d for d in detections
            if d['confidence'] >= min_confidence and d['area'] >= min_area
        ]
        return filtered
    
    def get_annotated_image(self, result_idx=0):
        """Get annotated image from results"""
        if self.results is None:
            return None
        return self.results[result_idx].plot()
    
    def summary(self):
        """Print detection summary"""
        if self.results is None:
            print("No detections available")
            return
        
        detections = self.get_detections()
        print(f"\n{'='*50}")
        print(f"Total objects detected: {len(detections)}")
        
        class_counts = {}
        for d in detections:
            class_name = d['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\n{'Class':<20} {'Count':<10} {'Avg Conf'}")
        print(f"{'-'*40}")
        
        for class_name, count in sorted(class_counts.items()):
            confidences = [d['confidence'] for d in detections if d['class_name'] == class_name]
            avg_conf = sum(confidences) / len(confidences)
            print(f"{class_name:<20} {count:<10} {avg_conf:.2f}")
        
        print(f"{'='*50}\n")
