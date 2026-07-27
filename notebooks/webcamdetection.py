import cv2
from ultralytics import YOLO
import time
import numpy as np

class TrafficSignDetector:
    def __init__(self, model_path='results/yolov8n.pt', conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the traffic sign detector
        
        Args:
            model_path: Path to the model file (.pt, .onnx, .engine)
            conf_threshold: Confidence threshold (0.0-1.0)
            iou_threshold: IoU threshold for NMS (0.0-1.0)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = self.model.names
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()
        
        print(f"Model loaded: {model_path}")
        print(f"Classes: {len(self.class_names)}")
        print(f"Classes list: {list(self.class_names.values())}")
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Returns:
            processed_frame: Frame with detections drawn
            detections: List of detection dictionaries
        """
        # Run inference
        results = self.model(
            frame, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        # Extract detections
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]
                
                detections.append({
                    'class': cls_name,
                    'confidence': float(conf),
                    'bbox': [x1, y1, x2, y2],
                    'center': [(x1+x2)//2, (y1+y2)//2]
                })
        
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.prev_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.prev_time = current_time
        
        # Draw detections on frame
        processed_frame = results.plot()
        
        # Add FPS counter
        cv2.putText(
            processed_frame, 
            f"FPS: {self.fps}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        return processed_frame, detections
    
    def run_webcam(self, camera_id=0, window_name="Traffic Sign Detection"):
        """
        Run real-time detection on webcam
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return
        
        print(f"Press 'q' to quit, 's' to save frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Process frame
            processed_frame, detections = self.process_frame(frame)
            
            # Display detections in console (optional)
            if detections:
                print(f"Detected {len(detections)} signs: {[d['class'] for d in detections]}")
            
            # Show frame
            cv2.imshow(window_name, processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('s'):  # Save frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"detection_{timestamp}.jpg", processed_frame)
                print(f"Frame saved as detection_{timestamp}.jpg")
            elif key == ord('c'):  # Toggle confidence threshold
                self.conf_threshold = 0.5 if self.conf_threshold == 0.25 else 0.25
                print(f"Confidence threshold: {self.conf_threshold}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_video_file(self, input_path, output_path=None, show_preview=True):
        """
        Process a video file
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video (None to not save)
            show_preview: Show real-time preview
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {input_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {input_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, detections = self.process_frame(frame)
            
            # Write to output video
            if output_path:
                out.write(processed_frame)
            
            # Show preview
            if show_preview:
                cv2.imshow("Processing Video", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_idx / elapsed
                print(f"Processed {frame_idx}/{total_frames} frames ({fps_actual:.1f} fps)")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
            print(f"✅ Output saved to: {output_path}")
        if show_preview:
            cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"Processing complete: {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")

# Usage
if __name__ == "__main__":
    # Initialize detector
    detector = TrafficSignDetector(
        model_path='best.pt',  # or 'best.onnx'
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # Option 1: Run webcam
    detector.run_webcam(camera_id=0)
    
    # Option 2: Process video file
    # detector.process_video_file(
    #     input_path="input_video.mp4",
    #     output_path="output_video.mp4",
    #     show_preview=True
    # )
