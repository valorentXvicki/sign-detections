"""
Advanced detection pipelines
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import cv2
from yolo_framework import YOLOFramework, DetectionResult
from collections import defaultdict

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Base detection pipeline"""
    
    def __init__(self, framework: YOLOFramework):
        self.framework = framework
        self.results = []
    
    def run(self, sources: List[str], **kwargs) -> List[DetectionResult]:
        """Run pipeline on sources"""
        raise NotImplementedError


class ImageDetectionPipeline(DetectionPipeline):
    """Pipeline for image detection"""
    
    def run(self, sources: List[str], conf=0.5, save_results=True) -> List[DetectionResult]:
        """
        Run detection on images
        
        Args:
            sources: List of image paths
            conf: Confidence threshold
            save_results: Save annotated images
            
        Returns:
            List of DetectionResult objects
        """
        logger.info(f"Starting image detection pipeline for {len(sources)} images")
        self.results = []
        
        for i, source in enumerate(sources):
            logger.info(f"[{i+1}/{len(sources)}] Processing: {Path(source).name}")
            
            result = self.framework.detect(source, conf=conf)
            if result:
                self.results.append(result)
                
                if save_results:
                    annotated = self.framework.get_annotated_image()
                    if annotated is not None:
                        output_file = Path(source).stem + '_detected.jpg'
                        cv2.imwrite(output_file, annotated)
                        logger.info(f"Saved: {output_file}")
        
        logger.info(f"Pipeline complete. Processed {len(self.results)} images")
        return self.results


class VideoDetectionPipeline(DetectionPipeline):
    """Pipeline for video detection"""
    
    def run(self, video_path: str, conf=0.5, save_video=True, 
            frame_skip=1, display=False) -> List[DetectionResult]:
        """
        Run detection on video
        
        Args:
            video_path: Path to video
            conf: Confidence threshold
            save_video: Save annotated video
            frame_skip: Process every nth frame
            display: Display while processing
            
        Returns:
            List of DetectionResult objects (one per frame)
        """
        logger.info(f"Starting video detection pipeline: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = Path(video_path).stem + '_detected.mp4'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.results = []
        frame_count = 0
        processed_frames = 0
        
        logger.info(f"Video info - FPS: {fps}, Size: {width}x{height}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                result = self.framework.detect(frame, conf=conf)
                if result:
                    self.results.append(result)
                    annotated = self.framework.get_annotated_image()
                    if save_video:
                        out.write(annotated)
                    if display:
                        cv2.imshow('Video Detection', annotated)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    processed_frames += 1
            else:
                if save_video:
                    out.write(frame)
            
            frame_count += 1
            
            if frame_count % (fps * 10) == 0:
                logger.info(f"Processed {frame_count} frames...")
        
        cap.release()
        if save_video:
            out.release()
            logger.info(f"Video saved: {output_path}")
        
        if display:
            cv2.destroyAllWindows()
        
        logger.info(f"Pipeline complete. Processed {processed_frames}/{frame_count} frames")
        return self.results


class RealtimeDetectionPipeline(DetectionPipeline):
    """Pipeline for real-time webcam detection"""
    
    def run(self, conf=0.5, save_video=False, fps_limit=None) -> List[DetectionResult]:
        """
        Run real-time detection from webcam
        
        Args:
            conf: Confidence threshold
            save_video: Save webcam recording
            fps_limit: Limit FPS (useful for testing)
            
        Returns:
            List of DetectionResult objects
        """
        logger.info("Starting real-time detection pipeline")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return []
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('webcam_realtime.mp4', fourcc, fps, (width, height))
        
        self.results = []
        frame_count = 0
        
        logger.info("Webcam detection started. Press 'q' to quit")
        
        import time
        last_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Limit FPS if specified
                if fps_limit:
                    current_time = time.time()
                    elapsed = current_time - last_time
                    wait_time = (1.0 / fps_limit) - elapsed
                    if wait_time > 0:
                        time.sleep(wait_time)
                    last_time = time.time()
                
                result = self.framework.detect(frame, conf=conf)
                if result:
                    self.results.append(result)
                    annotated = self.framework.get_annotated_image()
                    cv2.imshow('Real-time Detection', annotated)
                    
                    if save_video:
                        out.write(annotated)
                
                frame_count += 1
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if save_video:
                out.release()
                logger.info("Webcam recording saved: webcam_realtime.mp4")
            cv2.destroyAllWindows()
        
        logger.info(f"Pipeline complete. Processed {frame_count} frames")
        return self.results


class AnalyticsPipeline:
    """Pipeline for analyzing detection results"""
    
    def __init__(self, results: List[DetectionResult]):
        self.results = results
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of detected classes"""
        distribution = defaultdict(int)
        
        for result in self.results:
            for detection in result.detections:
                distribution[detection.class_name] += 1
        
        return dict(distribution)
    
    def get_confidence_statistics(self) -> Dict:
        """Get confidence statistics"""
        all_confidences = []
        
        for result in self.results:
            for detection in result.detections:
                all_confidences.append(detection.confidence)
        
        if not all_confidences:
            return {}
        
        return {
            'min': float(min(all_confidences)),
            'max': float(max(all_confidences)),
            'mean': float(sum(all_confidences) / len(all_confidences)),
            'median': float(sorted(all_confidences)[len(all_confidences) // 2])
        }
    
    def get_detection_frequency(self) -> Dict[str, int]:
        """Get how many frames had detections"""
        frequency = {
            'total_frames': len(self.results),
            'frames_with_detections': sum(1 for r in self.results if r.detections),
            'frames_without_detections': sum(1 for r in self.results if not r.detections)
        }
        return frequency
    
    def get_summary_report(self) -> str:
        """Generate summary report"""
        report = []
        report.append("="*60)
        report.append("DETECTION ANALYSIS REPORT")
        report.append("="*60)
        
        # Overall stats
        total_detections = sum(len(r.detections) for r in self.results)
        report.append(f"\nTotal Detections: {total_detections}")
        report.append(f"Total Frames: {len(self.results)}")
        
        # Class distribution
        class_dist = self.get_class_distribution()
        report.append(f"\nClass Distribution:")
        for class_name, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {class_name}: {count}")
        
        # Confidence stats
        conf_stats = self.get_confidence_statistics()
        if conf_stats:
            report.append(f"\nConfidence Statistics:")
            report.append(f"  Min: {conf_stats['min']:.2f}")
            report.append(f"  Max: {conf_stats['max']:.2f}")
            report.append(f"  Mean: {conf_stats['mean']:.2f}")
            report.append(f"  Median: {conf_stats['median']:.2f}")
        
        # Detection frequency
        freq = self.get_detection_frequency()
        report.append(f"\nDetection Frequency:")
        report.append(f"  Frames with detections: {freq['frames_with_detections']}")
        report.append(f"  Frames without detections: {freq['frames_without_detections']}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
