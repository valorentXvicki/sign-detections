"""
Examples of using the YOLO framework
"""

import logging
from yolo_framework import YOLOFramework
from pipelines import (
    ImageDetectionPipeline,
    VideoDetectionPipeline,
    RealtimeDetectionPipeline,
    AnalyticsPipeline
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_single_image():
    """Example: Detect objects in a single image"""
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: Single Image Detection")
    logger.info("=" * 60)
    
    framework = YOLOFramework(model_size='n')
    
    # Detect objects
    result = framework.detect('path/to/image.jpg', conf=0.5)
    
    if result:
        # Get statistics
        stats = framework.get_statistics(result)
        print(f"Total objects: {stats['total_objects']}")
        print(f"By class: {stats['by_class']}")
        
        # Get annotated image
        annotated = framework.get_annotated_image()
        
        # Save results
        framework.save_result(result)


def example_batch_images():
    """Example: Detect objects in multiple images"""
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: Batch Image Detection")
    logger.info("=" * 60)
    
    framework = YOLOFramework(model_size='n')
    
    # Run pipeline
    pipeline = ImageDetectionPipeline(framework)
    results = pipeline.run(
        sources=['image1.jpg', 'image2.jpg', 'image3.jpg'],
        conf=0.5,
        save_results=True
    )
    
    # Analyze results
    analytics = AnalyticsPipeline(results)
    print(analytics.get_summary_report())


def example_video():
    """Example: Detect objects in video"""
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: Video Detection")
    logger.info("=" * 60)
    
    framework = YOLOFramework(model_size='n')
    
    # Run pipeline
    pipeline = VideoDetectionPipeline(framework)
    results = pipeline.run(
        video_path='video.mp4',
        conf=0.5,
        save_video=True,
        frame_skip=1,
        display=False
    )
    
    # Analyze results
    analytics = AnalyticsPipeline(results)
    print(analytics.get_summary_report())


def example_realtime():
    """Example: Real-time webcam detection"""
    logger.info("=" * 60)
    logger.info("EXAMPLE 4: Real-time Webcam Detection")
    logger.info("=" * 60)
    
    framework = YOLOFramework(model_size='n')
    
    # Run real-time pipeline
    pipeline = RealtimeDetectionPipeline(framework)
    results = pipeline.run(
        conf=0.5,
        save_video=False,
        fps_limit=30
    )
    
    # Print summary
    print(f"Total frames processed: {len(results)}")


def example_custom_weights():
    """Example: Using custom trained weights"""
    logger.info("=" * 60)
    logger.info("EXAMPLE 5: Custom Weights")
    logger.info("=" * 60)
    
    framework = YOLOFramework(
        model_size='n',
        custom_weights='runs/detect/sign_detection/weights/best.pt'
    )
    
    # Get model info
    info = framework.get_model_info()
    print(f"Model info: {info}")
    
    # Run detection
    result = framework.detect('image.jpg', conf=0.5)
    if result:
        framework.save_result(result)


def example_filtering():
    """Example: Filtering detections"""
    logger.info("=" * 60)
    logger.info("EXAMPLE 6: Filtering Detections")
    logger.info("=" * 60)
    
    framework = YOLOFramework(model_size='n')
    
    # Run detection
    result = framework.detect('image.jpg', conf=0.5)
    
    if result:
        # Filter by confidence
        high_conf = framework.filter_detections(
            result,
            min_conf=0.7
        )
        print(f"High confidence detections: {len(high_conf)}")
        
        # Filter by class
        specific_class = framework.filter_detections(
            result,
            class_names=['person', 'car']
        )
        print(f"Specific class detections: {len(specific_class)}")
        
        # Filter by area
        large_objects = framework.filter_detections(
            result,
            min_area=10000
        )
        print(f"Large object detections: {len(large_objects)}")


def example_history():
    """Example: Using detection history"""
    logger.info("=" * 60)
    logger.info("EXAMPLE 7: Detection History")
    logger.info("=" * 60)
    
    framework = YOLOFramework(model_size='n')
    
    # Run multiple detections
    images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    for img in images:
        framework.detect(img, conf=0.5)
    
    # Get history
    history = framework.get_history()
    print(f"Total detections in history: {len(history)}")
    
    # Get statistics
    for i, result in enumerate(history):
        stats = framework.get_statistics(result)
        print(f"Image {i+1}: {stats['total_objects']} objects")
    
    # Export statistics
    framework.export_statistics('detection_stats.json')
    
    # Clear history
    framework.clear_history()


if __name__ == "__main__":
    print("\nYOLO Framework Examples\n")
    print("Available examples:")
    print("1. example_single_image() - Single image detection")
    print("2. example_batch_images() - Batch image detection")
    print("3. example_video() - Video detection")
    print("4. example_realtime() - Real-time webcam detection")
    print("5. example_custom_weights() - Using custom weights")
    print("6. example_filtering() - Filtering detections")
    print("7. example_history() - Detection history")
    print("\nRun any example by calling the function:")
    print("python examples.py")
