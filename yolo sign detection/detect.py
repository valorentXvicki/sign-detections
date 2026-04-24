import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

class ImageDetector:
    def __init__(self, model_path='runs/detect/sign_detection/weights/best.pt'):
        """Initialize detector with model"""
        self.model = YOLO(model_path)
        self.class_names = {0: 'Stop', 1: 'Speed Limit', 2: 'Yield', 3: 'No Entry'}
    
    def detect(self, image_path, conf_threshold=0.5, save_result=True, display=True):
        """Detect signs in image"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        results = self.model.predict(image, conf=conf_threshold)
        
        for result in results:
            annotated_image = result.plot()
            
            # Print detections
            print(f"\n=== Detections in {Path(image_path).name} ===")
            if len(result.boxes) == 0:
                print("No signs detected")
            else:
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.class_names.get(class_id, 'Unknown')
                    print(f"Detection {i+1}: {class_name} (Confidence: {confidence:.2f})")
            
            if save_result:
                output_path = Path(image_path).stem + '_detected.jpg'
                cv2.imwrite(output_path, annotated_image)
                print(f"Result saved to: {output_path}")
            
            if display:
                cv2.imshow('Detection Result', annotated_image)
                print("Press any key to close window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return annotated_image
        
        return None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Detect signs in images')
    parser.add_argument('image', help='Path to image for detection')
    parser.add_argument('--model', default='runs/detect/sign_detection/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save result')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display result')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    detector = ImageDetector(model_path=args.model)
    detector.detect(
        image_path=args.image,
        conf_threshold=args.conf,
        save_result=not args.no_save,
        display=not args.no_display
    )
