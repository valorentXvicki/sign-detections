"""
Configuration file for YOLO Sign Detection Project
"""

# Model Configuration
MODEL_SIZE = 'n'  # nano, small, medium, large, xlarge
MODEL_CONFIDENCE = 0.5
MODEL_IOU = 0.45

# Training Configuration
TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 16
TRAIN_IMAGE_SIZE = 640
TRAIN_DEVICE = 0  # GPU device index or -1 for CPU
TRAIN_PATIENCE = 20
TRAIN_SEED = 42

# Detection Classes
CLASS_NAMES = {
    0: 'Stop',
    1: 'Speed Limit',
    2: 'Yield',
    3: 'No Entry'
}

# Paths
MODEL_WEIGHTS_PATH = 'runs/detect/sign_detection/weights/best.pt'
DATASET_YAML = 'data.yaml'
RUNS_DIR = 'runs/detect'

# Output Configuration
SAVE_RESULTS = True
DISPLAY_RESULTS = True
OUTPUT_FORMAT = 'mp4'  # mp4 or avi

# Performance
USE_GPU = True
NUM_WORKERS = 4
PIN_MEMORY = True
