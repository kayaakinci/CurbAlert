# Adapting my object_detection_COCOdataset.py code to not use pyrealsense2

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time


class Camera_object_detection:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5, camera_id=0):
        self.confidence_threshold = confidence_threshold
        self.camera_id = camera_id
        
        # Initialize the camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera with ID {self.camera_id}")
            
        # Set resolution if needed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Loading YOLOv8 model
        self.model = YOLO(model_path)
        self.model_class_names = self.model.names  # classes from the COCO dataset

        # Window
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection", 1280, 720)

    # Function to process camera frames
    def process_stream(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    # function to apply YOLOv8 model to stream images
    # Gets classification and bounding box data
    def object_detection(self, rgb_image):
        results = self.model(rgb_image, conf=self.confidence_threshold)[0]

        # Processing results
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result

            width = x2 - x1
            height = y2 - y1

            detections.append({
                'class_id': int(class_id),
                'class_name': self.model_class_names[int(class_id)],
                'confidence': confidence,
                'bbox': (int(x1), int(y1), int(width), int(height))
            })

        return detections
    
    # Drawing the bounding boxes on the image of the frame of the stream
    def draw_bounding_boxes(self, image, detections):
        for detection in detections:
            x, y, w, h = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Use a fixed color scheme since we don't have depth data
            color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            label = f"{class_name} ({confidence:.2f})"
            
            # Draw background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                image, 
                (x, y - label_size[1] - 5), 
                (x + label_size[0], y), 
                color, 
                -1
            )
            
            # Draw label
            cv2.putText(
                image, 
                label, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
        
        return image

    def run_detection(self):
        try:
            while True:
                start_time = time.time()
                
                # Process frames
                color_image = self.process_stream()
                if color_image is None:
                    print("Failed to capture frame")
                    continue
                
                # Detect objects
                detections = self.object_detection(color_image)
                
                # Draw detections
                annotated_image = self.draw_bounding_boxes(color_image.copy(), detections)
                
                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(
                    annotated_image, 
                    f"FPS: {fps:.2f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                cv2.imshow("Object Detection", annotated_image)
                
                # Exit on ESC key
                key = cv2.waitKey(1)
                if key == 27:  # esc
                    break
                
        finally:  # happens whether an exception is raised or not
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create object detection instance
    detector = Camera_object_detection(
        model_path="yolov8n.pt",  # can use yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, or yolov8x.pt
        confidence_threshold=0.5,
        camera_id=0  # Use camera ID 0 (default), change if necessary
    )

    detector.run_detection()