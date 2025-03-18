import tensorflow as tf
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import time


class L515_object_detection:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
      self.confidence_threshold = confidence_threshold
      # self.model = YOLO(model_path)

      # Initialize pipeline
      self.pipeline = rs.pipeline()
      self.config = rs.config()

      # Enable streams (RGB and depth from L515)
      self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
      self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

      # Starting pipeline
      self.profile = self.pipeline.start(self.config)

      # Configuring depth for camera
      # This is to later get distance from object/bounding boxes
      self.depth_sensor = self.profile.get_device().first_depth_sensor()
      self.depth_scale = self.depth_sensor.get_depth_scale()
      print(f"Depth Scale: {self.depth_scale}")
      self.align = rs.align(rs.stream.color) # for alignment

      # Loading YOLOv8 model
      self.model = YOLO(model_path)
      self.model_class_names = self.model.names # classes from the COCO dataset

      # Window
      cv2.namedWindow("Hazard Detection", cv2.WINDOW_NORMAL)
      cv2.resizeWindow("Hazard Detection", 1280, 720) # idk maybe edit

    # Function to process camera frames and detect objects
    def process_stream(self):
      frames = self.pipeline.wait_for_frames()

      # Aligning both the depth and RGB frames
      aligned_frames = self.align.process(frames)
      color_frame = aligned_frames.get_color_frame()
      depth_frame = aligned_frames.get_depth_frame()

      if not color_frame or not depth_frame:
          return None, None

      # Converting to arrays
      color_image = np.asanyarray(color_frame.get_data())
      depth_colormap = cv2.applyColorMap(
          cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03),
          cv2.COLORMAP_JET)

      return color_image, depth_frame, depth_colormap


    # Gets distance to bounding box of an object using depth data from 3D LiDAR
    #   (x1, y1) is the point on the frame we are getting distance of
    def get_distance(self, depth_frame, x, y, width, height):
      # Get center point of the bounding box
      center_x = int(x + width/2)
      center_y = int(y + height/2)
      
      # Create small region around center to average depth
      size = 5
      x_min = max(0, center_x - size)
      x_max = min(depth_frame.width, center_x + size)
      y_min = max(0, center_y - size)
      y_max = min(depth_frame.height, center_y + size)
      
      # Calculate average distance from the region
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

    # function to apply YOLOv8 model to stream images
    #   Gets classification and bounding box data
    def object_detection(self, rgb_image, depth_frame):

      results = self.model(rgb_image, conf=self.confidence_threshold)[0]

      # Processing results
      detections = []
      for result in results.boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = result

        width = x2 - x1
        height = y2 - y1

        # Distance to object
        distance_to_hazard = self.get_distance(depth_frame, x1, y1, width, height)

        detections.append({
          'class_id': int(class_id),
          'class_name': self.model_class_names[int(class_id)],
          'confidence': confidence,
          'bbox': (int(x1), int(y1), int(width), int(height)),
          'distance': distance_to_hazard
        })

        return detections
    
    # Drawing the bounding boxes on the image of the frame of the stream
    def draw_bounding_boxes(self, image, detections):
      for detection in detections:
        x, y, w, h = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        distance = detection['distance']
            
        # Define color based on distance (closer = more red)
        if distance > 0:
            # Scale from green to red based on distance (0-3 meters)
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
            (0, 0, 0), 
            2
        )
    
      return image

    def run_hazard_detection(self):
      try:
          while True:
              start_time = time.time()
              
              # Process frames
              color_image, depth_frame, depth_colormap = self.process_frames()
              if color_image is None:
                  continue
              
              # Detect objects
              detections = self.detect_objects(color_image, depth_frame)
              
              # Draw detections
              annotated_image = self.draw_detections(color_image.copy(), detections)
              
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
              
              cv2.imshow("Hazard Detection", stacked_image)
              
              # Exit on ESC key
              key = cv2.waitKey(1)
              if key == 27:  # esc
                  break
              
      finally: # happens whether an exection is raised or not
          self.pipeline.stop()
          cv2.destroyAllWindows()



if __name__ == "__main__":
    # Create hazard detection object
    detector = L515_object_detection(
        model_path="yolov8n.pt",  # can use yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, or yolov8x.pt
        confidence_threshold=0.5
    )

    detector.run_hazard_detection()