import argparse
from ultralytics import YOLO
import os
from pathlib import Path

def train_model(data='data.yaml', epochs=100, imgsz=640, device=0, model_size='n', batch=16):
    """Train YOLOv8 model for sign detection"""
    
    # Model size options: n (nano), s (small), m (medium), l (large), x (xlarge)
    model_name = f'yolov8{model_size}.pt'
    
    # Load a pretrained model
    model = YOLO(model_name)
    
    print(f"Starting training with {model_name}...")
    print(f"Dataset: {data}")
    print(f"Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}")
    
    # Train the model
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        patience=20,
        save=True,
        project='runs/detect',
        name='sign_detection',
        verbose=True,
        seed=42
    )
    
    print(f"Training completed!")
    print(f"Best model saved to: runs/detect/sign_detection/weights/best.pt")
    print(f"Results: {results}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 for Sign Detection')
    
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device index (0 for first GPU, -1 for CPU)')
    parser.add_argument('--model', type=str, default='n',
                        help='Model size: n/s/m/l/x')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        model_size=args.model,
        batch=args.batch
    )
