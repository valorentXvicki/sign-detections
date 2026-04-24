import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

class VideoDetector:
    def __init__(self, model_path='runs/detect/sign_detection/weights/best.pt'):
        """Initialize detector with model"""
        self.model = YOLO(model_path)
        self.class_names = {0: 'Stop', 1: 'Speed Limit', 2: 'Yield', 3: 'No Entry'}
    
    def detect(self, video_path, conf_threshold=0.5, save_result=True, display=True):
        """Detect signs in video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video from {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if save_result:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = Path(video_path).stem + '_detected.mp4'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detections_summary = defaultdict(int)
        
        print(f"Processing video: {Path(video_path).name}")
        print("Press 'q' to stop processing...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model.predict(frame, conf=conf_threshold)
            
            for result in results:
                annotated_frame = result.plot()
                
                # Count detections per class
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.class_names.get(class_id, 'Unknown')
                        detections_summary[class_name] += 1
                
                frame = annotated_frame
            
            if save_result:
                out.write(frame)
            
            if display:
                cv2.imshow('Video Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        if save_result:
            out.release()
            print(f"Video saved to: {output_path}")
        
        if display:
            cv2.destroyAllWindows()
        
        print(f"\n=== Detection Summary ===")
        print(f"Total frames processed: {frame_count}")
        for sign, count in sorted(detections_summary.items()):
            print(f"{sign}: {count}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Detect signs in videos')
    parser.add_argument('video', help='Path to video for detection')
    parser.add_argument('--model', default='runs/detect/sign_detection/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save result video')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display video while processing')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    detector = VideoDetector(model_path=args.model)
    detector.detect(
        video_path=args.video,
        conf_threshold=args.conf,
        save_result=not args.no_save,
        display=not args.no_display
    )
