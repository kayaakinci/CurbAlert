import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time


class Camera_object_detection:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5, color_camera_id=0, depth_camera_id=1):
        self.confidence_threshold = confidence_threshold
        self.color_camera_id = color_camera_id
        self.depth_camera_id = depth_camera_id
        
        # Initialize the color camera
        self.color_cap = cv2.VideoCapture(self.color_camera_id)
        if not self.color_cap.isOpened():
            raise Exception(f"Could not open color camera with ID {self.color_camera_id}")
            
        # Try to initialize the depth camera
        # For Intel RealSense, we'll try different approaches
        self.depth_cap = None
        self.has_depth = False
        
        # Attempt 1: Try with CAP_INTEL_REALSENSE flag
        try:
            self.depth_cap = cv2.VideoCapture(self.depth_camera_id, cv2.CAP_INTEL_REALSENSE)
            if self.depth_cap.isOpened():
                self.has_depth = True
                print("Depth camera opened with CAP_INTEL_REALSENSE flag")
        except Exception as e:
            print(f"Failed to open depth camera with CAP_INTEL_REALSENSE: {e}")
        
        # Attempt 2: Try with regular VideoCapture if first attempt failed
        if not self.has_depth:
            try:
                self.depth_cap = cv2.VideoCapture(self.depth_camera_id)
                if self.depth_cap.isOpened():
                    self.has_depth = True
                    print("Depth camera opened with regular VideoCapture")
                else:
                    print("Failed to open depth camera with regular VideoCapture")
            except Exception as e:
                print(f"Failed to open depth camera: {e}")
        
        # Set resolution for color camera
        self.color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Set resolution for depth camera if available
        if self.has_depth:
            self.depth_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
            self.depth_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        
        # Loading YOLOv8 model
        self.model = YOLO(model_path)
        self.model_class_names = self.model.names  # classes from the COCO dataset

        # Window
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection", 2560, 720)  # Wider to accommodate both streams

    # Function to process camera frames
    def process_stream(self):
        ret_color, color_frame = self.color_cap.read()
        
        depth_frame = None
        if self.has_depth:
            ret_depth, depth_frame = self.depth_cap.read()
            if not ret_depth:
                self.has_depth = False
                print("Failed to read from depth camera, disabling depth stream")
        
        if not ret_color:
            return None, None
        
        # Convert depth frame to a color visualization if available
        depth_colormap = None
        if depth_frame is not None:
            # Check if it's already a color image (RGB/BGR)
            if len(depth_frame.shape) == 3:
                depth_colormap = depth_frame
            else:
                # If it's a single-channel depth image, apply a color map
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_frame, alpha=0.03),
                    cv2.COLORMAP_JET
                )
        
        return color_frame, depth_colormap

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
            
            # Use a fixed color scheme
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
                color_image, depth_colormap = self.process_stream()
                if color_image is None:
                    print("Failed to capture color frame")
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
                
                # Prepare display output
                if depth_colormap is not None:
                    # Resize depth colormap to match color image height
                    h, w = color_image.shape[:2]
                    depth_h, depth_w = depth_colormap.shape[:2]
                    
                    # Calculate scaling factor to make heights match
                    scale = h / depth_h
                    new_depth_w = int(depth_w * scale)
                    
                    depth_colormap_resized = cv2.resize(depth_colormap, (new_depth_w, h))
                    
                    # Stack color image and depth colormap side by side
                    stacked_image = np.hstack((annotated_image, depth_colormap_resized))
                    
                    # Add a label to identify the depth stream
                    cv2.putText(
                        stacked_image, 
                        "Depth Stream", 
                        (annotated_image.shape[1] + 10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (255, 255, 255), 
                        2
                    )
                else:
                    # If no depth data is available, just display the color image
                    stacked_image = annotated_image
                    # Add a message that depth stream is not available
                    cv2.putText(
                        stacked_image, 
                        "Depth stream not available", 
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 0, 255), 
                        2
                    )
                
                cv2.imshow("Object Detection", stacked_image)
                
                # Exit on ESC key
                key = cv2.waitKey(1)
                if key == 27:  # esc
                    break
                
        finally:  # happens whether an exception is raised or not
            self.color_cap.release()
            if self.depth_cap:
                self.depth_cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create object detection instance
    detector = Camera_object_detection(
        model_path="yolov8n.pt",  # can use yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, or yolov8x.pt
        confidence_threshold=0.5,
        color_camera_id=0,  # Color camera index
        depth_camera_id=1   # Depth camera index (may need adjustment)
    )

    detector.run_detection()