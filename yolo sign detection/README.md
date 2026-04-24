# YOLO Sign Detection

Complete real-time traffic sign detection project using YOLOv8.

## Setup

### Installation

```bash
pip install -r requirements.txt
```

### Dataset Preparation

Create the following directory structure:

```
dataset/
├── images/
│   ├── train/     (training images)
│   ├── val/       (validation images)
│   └── test/      (test images)
└── labels/
    ├── train/     (YOLO format annotations)
    ├── val/
    └── test/
```

Update `data.yaml` with correct dataset path and class names.

## Usage

### 1. Training

Train a new model:

```bash
python train.py --data data.yaml --epochs 100 --model n
```

**Options:**
- `--data`: Path to dataset YAML file (default: data.yaml)
- `--epochs`: Number of training epochs (default: 100)
- `--imgsz`: Image size (default: 640)
- `--device`: GPU device index or -1 for CPU (default: 0)
- `--model`: Model size - n/s/m/l/x (default: n)
- `--batch`: Batch size (default: 16)

### 2. Image Detection

Detect signs in a single image:

```bash
python detect.py path/to/image.jpg --conf 0.5
```

**Options:**
- `image`: Path to image file (required)
- `--model`: Path to model weights (default: runs/detect/sign_detection/weights/best.pt)
- `--conf`: Confidence threshold (default: 0.5)
- `--no-save`: Do not save detection result
- `--no-display`: Do not display result

### 3. Video Detection

Detect signs in a video file:

```bash
python video_detect.py path/to/video.mp4 --conf 0.5
```

**Options:**
- `video`: Path to video file (required)
- `--model`: Path to model weights
- `--conf`: Confidence threshold (default: 0.5)
- `--no-save`: Do not save output video
- `--no-display`: Do not display while processing

### 4. Real-time Webcam Detection

Run real-time detection from webcam:

```bash
python webcam_detect.py --conf 0.5
```

**Options:**
- `--model`: Path to model weights
- `--conf`: Confidence threshold (default: 0.5)
- `--save`: Save webcam recording

### 5. Complete Pipeline

Use `main.py` for unified detection:

```bash
# Webcam
python main.py --source 0

# Image
python main.py --source path/to/image.jpg

# Video
python main.py --source path/to/video.mp4
```

**Options:**
- `--source`: 0 (webcam), image path, or video path (default: 0)
- `--weights`: Path to model weights
- `--conf`: Confidence threshold (default: 0.5)
- `--device`: Device to use (cuda or cpu)
- `--save`: Save detection results

## Model Classes

- **0**: Stop Sign
- **1**: Speed Limit Sign
- **2**: Yield Sign
- **3**: No Entry Sign

## Output

- **Images**: Saves annotated images as `{filename}_detected.jpg`
- **Videos**: Saves annotated videos as `{filename}_detected.mp4`
- **Webcam**: Saves recordings as `webcam_detection.mp4` (with --save flag)

## Performance Metrics

Training results are saved in:
```
runs/detect/sign_detection/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── confusion_matrix.png
└── ...
```

## Advanced Frameworks

### YOLO Framework

Complete object detection framework with:
- Multiple detection modes (image, video, real-time)
- Advanced filtering and statistics
- Detection history and export
- Multiple pipelines for different use cases

See `FRAMEWORK.md` for complete documentation.

### Object Detection Modules

- **models.py** - Core YOLO model wrapper
- **yolo_framework.py** - Complete framework
- **pipelines.py** - Detection pipelines
- **object_detector.py** - Generic detector

## Usage Examples

See `examples.py` for 7 detailed examples:
1. Single image detection
2. Batch image processing
3. Video detection
4. Real-time webcam
5. Custom weights
6. Filtering detections
7. Detection history

## Troubleshooting

- **Webcam not opening**: Check if device is available or try device index 1, 2, etc.
- **CUDA errors**: Install compatible NVIDIA drivers or use CPU with `--device cpu`
- **Low detection accuracy**: Increase epochs during training or collect more training data
- **Out of memory**: Reduce batch size with `--batch 8` or use smaller model size with `--model n`

## System Requirements

- Python 3.8+
- NVIDIA GPU (CUDA 11.0+) - optional but recommended
- 8GB+ RAM
- Sufficient disk space for dataset and models

