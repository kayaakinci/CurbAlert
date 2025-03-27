import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import time


class L515_object_detection:
    def __init__(self, 
                 stairs_model_path="/content/drive/MyDrive/stairs_detection_final/best.pt", 
                 coco_model_path="yolov8n.pt",  # Original COCO model
                 confidence_threshold=0.5, 
                 camera_id="/dev/video6"):
                 
        self.confidence_threshold = confidence_threshold
        
        # Initialize pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable streams (RGB and depth from L515)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

        # Set camera ID if provided
        if camera_id is not None:
            self.config.enable_device(camera_id)
            print(f"Using camera with ID: {camera_id}")

        # Starting pipeline
        self.profile = self.pipeline.start(self.config)

        # Configuring depth for camera
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print(f"Depth Scale: {self.depth_scale}")
        self.align = rs.align(rs.stream.color) # for alignment

        # Loading two models:
        # 1. The fine-tuned stairs model
        print(f"Loading stairs model from: {stairs_model_path}")
        self.stairs_model = YOLO(stairs_model_path)
        self.stairs_model_classes = self.stairs_model.names
        print(f"Stairs model loaded with {len(self.stairs_model_classes)} classes")
        
        # 2. The original COCO model for all other objects
        print(f"Loading COCO model from: {coco_model_path}")
        self.coco_model = YOLO(coco_model_path)
        self.coco_model_classes = self.coco_model.names
        print(f"COCO model loaded with {len(self.coco_model_classes)} classes")
        
        # Combined class names
        self.model_class_names = self.coco_model_classes.copy()
        # Add stairs class if not already in COCO classes
        if 'stairs' not in self.model_class_names.values():
            new_idx = max(self.model_class_names.keys()) + 1
            self.model_class_names[new_idx] = 'stairs'
            self.stairs_class_id = new_idx
        else:
            # Find the existing stairs class ID
            self.stairs_class_id = next(id for id, name in self.model_class_names.items() 
                                       if name == 'stairs')
        
        print(f"Combined model has {len(self.model_class_names)} classes")
        print(f"Stairs class ID: {self.stairs_class_id}")

        # Window
        cv2.namedWindow("Object and Stair Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object and Stair Detection", 1280, 720)

    # Function to process camera frames and detect objects
    def process_stream(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03),
            cv2.COLORMAP_JET)

        return color_image, depth_frame, depth_colormap

    # Gets distance to bounding box of an object using depth data
    def get_distance(self, depth_frame, x, y, width, height):
        center_x = int(x + width/2)
        center_y = int(y + height/2)
        
        size = 5
        x_min = max(0, center_x - size)
        x_max = min(depth_frame.width, center_x + size)
        y_min = max(0, center_y - size)
        y_max = min(depth_frame.height, center_y + size)
        
        depth_region = []
        for rx in range(x_min, x_max):
            for ry in range(y_min, y_max):
                dist = depth_frame.get_distance(rx, ry)
                if dist > 0:
                    depth_region.append(dist)
        
        if depth_region:
            return np.mean(depth_region)
        else:
            return 0

    # Detect objects using both models
    def object_detection(self, rgb_image, depth_frame):
        # Run both models
        stairs_results = self.stairs_model(rgb_image, conf=self.confidence_threshold)[0]
        coco_results = self.coco_model(rgb_image, conf=self.confidence_threshold)[0]
        
        # Process stairs results
        detections = []
        
        # First, process stairs detections (priority)
        for result in stairs_results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            
            # All detections from the stairs model are treated as stairs
            # regardless of what class ID it reports
            
            width = x2 - x1
            height = y2 - y1
            distance = self.get_distance(depth_frame, x1, y1, width, height)
            
            detections.append({
                'class_id': self.stairs_class_id,  # Use our stairs class ID
                'class_name': 'stairs',
                'confidence': confidence,
                'bbox': (int(x1), int(y1), int(width), int(height)),
                'distance': distance
            })
        
        # Then process COCO model results
        for result in coco_results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            class_id = int(class_id)
            
            # Skip if this is a stair detection from COCO model
            # (we'll use our fine-tuned model for stairs)
            if self.coco_model_classes.get(class_id) == 'stairs':
                continue
                
            width = x2 - x1
            height = y2 - y1
            distance = self.get_distance(depth_frame, x1, y1, width, height)
            
            detections.append({
                'class_id': class_id,
                'class_name': self.coco_model_classes.get(class_id, 'unknown'),
                'confidence': confidence,
                'bbox': (int(x1), int(y1), int(width), int(height)),
                'distance': distance
            })
            
        return detections
    
    # Drawing the bounding boxes 
    def draw_bounding_boxes(self, image, detections):
        # Track object counts
        object_counts = {}
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            distance = detection['distance']
            
            # Update count
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            # Choose color based on class
            if class_name == 'stairs':
                # Blue for stairs
                color = (255, 0, 0)
            else:
                # Distance-based color for other objects
                if distance > 0:
                    max_distance = 2.286  # meters (7.5 feet)
                    normalized_distance = min(distance / max_distance, 1.0)
                    color = (
                        0,  # B
                        int(255 * normalized_distance),  # G
                        int(255 * (1 - normalized_distance))  # R
                    )
                else:
                    color = (0, 0, 0)  # Black for invalid distances
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            label = f"{class_name} ({confidence:.2f}) - {distance:.2f}m"
            
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
                (255, 255, 255), 
                2
            )
        
        # Show object counts
        y_offset = 30
        for i, (class_name, count) in enumerate(object_counts.items()):
            if class_name == 'stairs':
                text_color = (255, 0, 0)  # Blue for stairs
            else:
                text_color = (0, 255, 0)  # Green for other objects
                
            cv2.putText(
                image,
                f"{class_name}: {count}",
                (10, y_offset + i*30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2
            )
        
        return image

    def run_hazard_detection(self):
        try:
            while True:
                start_time = time.time()
                
                # Process frames
                color_image, depth_frame, depth_colormap = self.process_stream()
                if color_image is None:
                    continue
                
                # Detect objects
                detections = self.object_detection(color_image, depth_frame)
                
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
                
                # Stack color image and depth colormap side by side
                stacked_image = np.hstack((annotated_image, depth_colormap))
                
                cv2.imshow("Object and Stair Detection", stacked_image)
                
                # Exit on ESC key
                key = cv2.waitKey(1)
                if key == 27:  # esc
                    break
                
        finally: # happens whether an exception is raised or not
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create object detection instance with both models
    detector = L515_object_detection(
        stairs_model_path="/content/drive/MyDrive/stairs_detection_final/best.pt",  # Fine-tuned stairs model
        coco_model_path="yolov8n.pt",  # Original COCO model
        confidence_threshold=0.4,
        camera_id='/dev/video6'  # Set your camera ID if needed
    )

    detector.run_hazard_detection()