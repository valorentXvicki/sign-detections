# YOLO Framework - Complete Object Detection System

Advanced YOLOv8-based object detection framework with multiple detection modes, pipelines, and analytics.

## Features

### Core Components

- **YOLOModel**: Low-level YOLO model wrapper with inference capabilities
- **ObjectDetector**: High-level detector with filtering and statistics
- **YOLOFramework**: Complete framework with history tracking and export
- **Detection Pipelines**: Pre-built pipelines for different detection scenarios
- **Analytics**: Statistical analysis of detection results

### Supported Detection Modes

1. **Single Image Detection** - Detect objects in a single image
2. **Batch Image Processing** - Process multiple images efficiently
3. **Video Detection** - Frame-by-frame detection with video output
4. **Real-time Webcam** - Live detection from webcam feed
5. **Custom Models** - Support for custom-trained weights

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Detection

```python
from yolo_framework import YOLOFramework

# Initialize framework
framework = YOLOFramework(model_size='n')

# Detect objects
result = framework.detect('image.jpg', conf=0.5)

# Get statistics
stats = framework.get_statistics(result)
print(f"Found {stats['total_objects']} objects")
```

### Using Pipelines

```python
from yolo_framework import YOLOFramework
from pipelines import ImageDetectionPipeline, AnalyticsPipeline

framework = YOLOFramework(model_size='n')

# Batch image detection
pipeline = ImageDetectionPipeline(framework)
results = pipeline.run(
    sources=['img1.jpg', 'img2.jpg', 'img3.jpg'],
    conf=0.5,
    save_results=True
)

# Analyze results
analytics = AnalyticsPipeline(results)
print(analytics.get_summary_report())
```

### Real-time Detection

```python
from yolo_framework import YOLOFramework
from pipelines import RealtimeDetectionPipeline

framework = YOLOFramework(model_size='n')

pipeline = RealtimeDetectionPipeline(framework)
results = pipeline.run(conf=0.5, save_video=True)
```

### Video Processing

```python
from yolo_framework import YOLOFramework
from pipelines import VideoDetectionPipeline

framework = YOLOFramework(model_size='n')

pipeline = VideoDetectionPipeline(framework)
results = pipeline.run(
    video_path='video.mp4',
    conf=0.5,
    save_video=True,
    frame_skip=1,
    display=False
)
```

## Module Documentation

### models.py

**YOLOModel**
- `load_pretrained()` - Load pretrained model
- `load_weights(path)` - Load custom weights
- `predict(source, conf, iou)` - Run inference
- `train(data, epochs, batch)` - Train model
- `get_model_info()` - Get model information

**ObjectDetector**
- `detect(source, conf, iou)` - Detect objects
- `get_detections()` - Get detection details
- `filter_detections(conf, area)` - Filter by criteria
- `get_annotated_image()` - Get annotated output
- `summary()` - Print detection summary

### yolo_framework.py

**YOLOFramework**
- `detect(source, conf)` - Single detection
- `batch_detect(sources, conf)` - Multiple detections
- `filter_detections(result, min_conf, min_area, classes)` - Filter results
- `get_statistics(result)` - Get detection stats
- `save_result(result, output_dir)` - Save detections
- `get_history()` - Get detection history
- `export_statistics(output_file)` - Export all stats

**Detection Classes**
- `Detection` - Individual detection data
- `DetectionResult` - Result for one source

### pipelines.py

**Pipelines**
1. `ImageDetectionPipeline` - Batch image processing
2. `VideoDetectionPipeline` - Video frame processing
3. `RealtimeDetectionPipeline` - Webcam detection
4. `AnalyticsPipeline` - Result analysis

### object_detector.py

**GenericObjectDetector**
- `detect_image()` - Image detection
- `detect_video()` - Video detection
- `detect_webcam()` - Real-time detection

## Command-Line Usage

### Object Detection

```bash
# Webcam detection
python object_detector.py --source 0

# Image detection
python object_detector.py --source image.jpg --conf 0.5 --save

# Video detection
python object_detector.py --source video.mp4 --conf 0.5 --save

# With custom weights
python object_detector.py --source image.jpg --weights custom_model.pt
```

### Options

- `--source` - Input source (0, image path, or video path)
- `--weights` - Path to custom weights
- `--model` - Model size: n/s/m/l/x (default: n)
- `--conf` - Confidence threshold (default: 0.5)
- `--device` - Device: cuda or cpu
- `--save` - Save results
- `--no-display` - Don't display results

## Detection Result Format

```python
{
    'source': 'image.jpg',
    'num_detections': 5,
    'timestamp': '2024-01-15T10:30:00',
    'image_shape': (640, 480, 3),
    'detections': [
        {
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.92,
            'bbox': [100, 150, 250, 400],
            'area': 37500
        },
        ...
    ]
}
```

## Statistics Output

```python
{
    'total_objects': 15,
    'by_class': {
        'person': 8,
        'car': 5,
        'bicycle': 2
    },
    'confidence_stats': {
        'min': 0.51,
        'max': 0.98,
        'mean': 0.75
    },
    'area_stats': {
        'min': 1000,
        'max': 50000,
        'mean': 15000
    }
}
```

## Model Sizes

- `n` - Nano (fastest, lowest accuracy)
- `s` - Small
- `m` - Medium
- `l` - Large
- `x` - Extra Large (slowest, highest accuracy)

## Performance Tips

1. Use model size `n` or `s` for real-time applications
2. Use GPU for faster inference: `device='cuda'`
3. Increase confidence threshold for fewer false positives
4. Use frame skipping in video processing for faster speeds
5. Filter detections by area to remove noise

## Troubleshooting

- **CUDA errors**: Install compatible NVIDIA drivers or use CPU
- **Out of memory**: Reduce batch size or use smaller model
- **Low accuracy**: Use larger model or collect more training data
- **Webcam not opening**: Check device permissions or try device index 1, 2, etc.

## File Structure

```
yolo sign detection/
├── models.py              # Core YOLO model classes
├── yolo_framework.py      # Main framework
├── object_detector.py     # Generic object detector
├── pipelines.py           # Detection pipelines
├── examples.py            # Usage examples
├── config.py              # Configuration
├── utils.py               # Utility functions
├── train.py               # Training script
├── detect.py              # Image detection script
├── video_detect.py        # Video detection script
├── webcam_detect.py       # Webcam detection script
├── main.py                # Unified detection
└── data.yaml              # Dataset configuration
```

## Advanced Usage

### Custom Detection Pipeline

```python
from yolo_framework import YOLOFramework
from pipelines import DetectionPipeline

class CustomPipeline(DetectionPipeline):
    def run(self, sources, **kwargs):
        # Your custom detection logic
        pass

framework = YOLOFramework()
pipeline = CustomPipeline(framework)
results = pipeline.run(sources)
```

### Export for Different Formats

```python
framework = YOLOFramework()
result = framework.detect('image.jpg')

# Export as JSON
framework.save_result(result, 'json_results')

# Export statistics
framework.export_statistics('stats.json')
```

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [COCO Dataset](https://cocodataset.org)
