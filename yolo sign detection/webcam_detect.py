import argparse
import cv2
from ultralytics import YOLO

class WebcamDetector:
    def __init__(self, model_path='runs/detect/sign_detection/weights/best.pt'):
        """Initialize detector with model"""
        self.model = YOLO(model_path)
        self.class_names = {0: 'Stop', 1: 'Speed Limit', 2: 'Yield', 3: 'No Entry'}
    
    def detect(self, conf_threshold=0.5, save_result=False):
        """Real-time detection from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if save_result:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = 'webcam_detection.mp4'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Webcam detection started. Press 'q' to quit.")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model.predict(frame, conf=conf_threshold)
            
            for result in results:
                annotated_frame = result.plot()
                frame = annotated_frame
            
            if save_result:
                out.write(frame)
            
            cv2.imshow('Webcam Detection - Press q to quit', frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if save_result:
            out.release()
            print(f"Webcam recording saved to: {output_path}")
        
        cv2.destroyAllWindows()
        print(f"Total frames processed: {frame_count}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Real-time sign detection from webcam')
    parser.add_argument('--model', default='runs/detect/sign_detection/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--save', action='store_true',
                        help='Save webcam recording')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    detector = WebcamDetector(model_path=args.model)
    detector.detect(
        conf_threshold=args.conf,
        save_result=args.save
    )
