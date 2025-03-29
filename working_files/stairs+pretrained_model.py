import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import pyrealsense2 as rs
import serial # for haptics

class Camera_object_detection:
    def __init__(self, 
                 stairs_model_path="best.pt", 
                 coco_model_path="yolov8n.pt", 
                 confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold

        # start pylibrealse stream pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        # Loading two models:
        # 1. Fine-tuned stairs model
        print(f"Loading stairs model: {stairs_model_path}")
        self.stairs_model = YOLO(stairs_model_path)
        self.stairs_model_classes = self.stairs_model.names
        print(f"Stairs model loaded with {len(self.stairs_model_classes)} classes")
        
        # 2. Original COCO model for all other objects
        print(f"Loading COCO model: {coco_model_path}")
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
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection", 1280, 720)
        
    # Function to process camera frames
    def process_stream(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # Convert color image to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame

    def get_distance_from_point(self, depth_frame, center_x, center_y):
        # convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # scale depth measurement points to color
        x_scale = 1280 / 640
        y_scale = 720 / 480

        scaled_x = int((1/x_scale) * center_x)
        scaled_y = int((1/y_scale) * center_y)

        distance_to_centerpoint = depth_frame.get_distance(scaled_x, scaled_y)
        return distance_to_centerpoint # in meters

    
    # def object_detection(self, rgb_image):
    # Now returns list of objects detected along with list of distances to the center points of the objects
    def object_detection(self, rgb_image, depth_frame):
        # Run both models
        stairs_results = self.stairs_model(rgb_image, conf=0.8)[0]
        coco_results = self.coco_model(rgb_image, conf=self.confidence_threshold)[0]

        # Process results
        detections = []
        distances = []
        
        # First, process stairs detections (priority)
        for result in stairs_results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            
            # All detections from the stairs model are treated as stairs
            # regardless of what class ID it reports
            
            width = x2 - x1
            height = y2 - y1
            
            detections.append({
                'class_id': self.stairs_class_id,  # Use our stairs class ID
                'class_name': 'stairs',
                'confidence': confidence,
                'bbox': (int(x1), int(y1), int(width), int(height))
            })

            center_x = int(x1 + (width//2))
            center_y = int(y1 + (height//2))
            distances.append(((center_x, center_y), self.get_distance_from_point(depth_frame, center_x, center_y)))
        
        
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
            
            detections.append({
                'class_id': class_id,
                'class_name': self.coco_model_classes.get(class_id, 'unknown'),
                'confidence': confidence,
                'bbox': (int(x1), int(y1), int(width), int(height))
            })

            center_x = int(x1 + (width//2))
            center_y = int(y1 + (height//2))
            distances.append(((center_x, center_y), self.get_distance_from_point(depth_frame, center_x, center_y)))
            
        return detections, distances

    def get_depth_distances(self, depth_frame):
        # convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())


         # create 9 measurement points (3 columns, 3 rows)
        height, width = depth_image.shape
        thirds_x = [width // 4, width // 2, (3 * width) // 4]
        thirds_y = [height // 4, height // 2, (3 * height) // 4]
        measurement_points = [(x, y) for x in thirds_x for y in thirds_y]

    

        # scale depth measurement points to color
        x_scale = 1280 / 640
        y_scale = 720 / 480
        # store depths in dict
        measurement_points = {(x,y): depth_frame.get_distance(x, y) for x,y in measurement_points}
        # calculate final distances by scaling the depth distance to color distance
        final_distances = {(int(x *x_scale), int(y *y_scale)): distance for (x, y), distance in measurement_points.items()}

        return final_distances


    
     # Drawing the bounding boxes on the image of the frame of the stream
    def draw_bounding_boxes(self, image, detections, distances):
        # Track object counts
        object_counts = {}
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Update count
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            # Use different colors for different classes
            if class_name == 'stairs':
                color = (255, 0, 0)  # Blue for stairs
            else:
                color = (0, 255, 0)  # Green for other objects
            
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

        # draw distance circles
        for (x, y), distance in distances:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, f"{distance:.2f}m", (x - 30, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Show object counts
        y_offset = 60  # Start below FPS counter
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

    def run_detection(self):
        command = "none"
        prev_command = "n/a"
        try:
            while True:
                start_time = time.time()

                # Process frames
                color_image, depth_image = self.process_stream()
                if color_image is None:
                    print("Failed to capture frame")
                    continue

                # Detect objects
                # detections = self.object_detection(color_image)
                detections, distances = self.object_detection(color_image, depth_image)
                # distances = self.get_depth_distances(depth_image)

                if (detections != []):
                    prev_command = command
                    command = decide_haptic_response(detections, distances)
                else:
                    prev_command = command
                    command = "none"
                send_haptic_command(command, prev_command)

                # Draw detections
                annotated_image = self.draw_bounding_boxes(color_image.copy(), detections, distances)

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
                

        finally:
            # happens whether an exception is raised or not
            self.pipeline.stop()
            cv2.destroyAllWindows()

''' Different commands:
"wall turn left"
"wall turn right"
"object turn left"
"object turn right"
"stairs"
'''
def send_haptic_command(command, prev_command):
    print("Starting stair haptic...\n")
    if (prev_command != command):
        ser.write((command + "\n").encode())
        print("Haptic command sent")
    else:
        print("no haptics")

def decide_object_action(nearest_hazard, second_hazard):
    action = "none"

    # Using global CENTER_REGION
    x_nearest, y_nearest, w_nearest, h_nearest = nearest_hazard['bbox']
    x_center_nearest = x_nearest + (x_nearest + w_nearest)//2
    x_right_nearest = x_nearest + w_nearest
    x_left_nearest = x_nearest

    if (second_hazard == None):
        if (x_center_nearest <= CENTER_REGION[0]):
            return "object turn right"
        else:
            return "object turn left"
    
    x_second, y_second, w_second, h_second = second_hazard['bbox']
    x_center_second = x_second + (x_second + w_second)//2
    x_right_second = x_second + w_second
    x_left_second = x_second
    
    if (x_center_nearest <= CENTER_REGION[0]):
        if (x_center_second <= CENTER_REGION[0]):
            action = "object turn right"
        else:
            # action = "object turn left"
            return "stairs"
    else:
        if (x_center_second >= CENTER_REGION[0]):
            action = "object turn left"
        else:
            action = "object turn right"
    return action


""" Deciding haptic response (need to get wall data and distance data for this function)
based on the x location of nearest hazard in frame (in center third of frame)
"""
def decide_haptic_response(detections, distances):
    nearest_hazard = None
    second_nearest = None
    nearest_distance = None
    second_distance = None
    for i in range(len(detections)):
        detection = detections[i]
        curr_distance = distances[i]
    # for detection in detections:
        x, y, w, h = detection['bbox']
        x_center = x + (x + w)//2
        x_right = x + w
        x_left = x
        
        if (CENTER_REGION[0] < x_right or x_left < CENTER_REGION[1]):
            if (curr_distance != 0 and (nearest_distance == None or curr_distance >= nearest_distance)):
                second_distance = nearest_distance
                nearest_distance = curr_distance
                second_nearest = nearest_hazard
                nearest_hazard = detection 
            elif (curr_distance != 0 and (second_distance == None or curr_distance >= second_distance)):
                second_distance = curr_distance
                second_nearest = detection
        else:
            if (curr_distance != 0 and (second_distance == None or curr_distance >= second_distance)):
                second_distance = curr_distance
                second_nearest = detection
    
    # Deciding object haptic response to send
    if (detection["class_id"] == "stairs"):
        return "stairs"
    else:
        return decide_object_action(nearest_hazard, second_nearest)



if __name__ == "__main__":
    # Connecting to the haptics motor controller
    ser = serial.Serial('/dev/ttyACM0', 9600)
    time.sleep(2)

    # middle of screen region where objects are hazards
    CENTER_REGION = (int(cv2.CAP_PROP_FRAME_WIDTH * 0.33), int(cv2.CAP_PROP_FRAME_WIDTH * 0.66))
    
    detector = Camera_object_detection(
        stairs_model_path="best.pt",
        coco_model_path="yolov8n.pt",
        confidence_threshold=0.5
    )
    detector.run_detection()
