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
                 confidence_threshold=0.65):
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
    def object_detection(self, rgb_image, depth_frame, CENTER_REGION, bottom_y):
        # Run both models
        stairs_results = self.stairs_model(rgb_image, conf=0.45, verbose=False)[0]
        coco_results = self.coco_model(rgb_image, conf=self.confidence_threshold, verbose=False)[0]

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

            center_x = int(x1 + (width//2))
            center_y = int(y1 + (height//2))
            if (y2 > bottom_y):
                y2 = bottom_y
            hazard_dist = self.get_distance_from_point(depth_frame, center_x, y2)
            if (class_id == self.stairs_class_id):
                print("STAIRS DETECTIONS ARE: ", result)
                print("\n")
                # if (hazard_dist > 0 and hazard_dist <= 2.2):
                    # Checking if in center of frame
                #if ((x1 >= CENTER_REGION[0] and x1 < CENTER_REGION[1]) or (x2 <= CENTER_REGION[1] and x2 > CENTER_REGION[0])):
                print("x: ", center_x, "y2 ", y2, "hazard_distance ", hazard_dist, "\n")
                center_x = int(center_x)
                y2 = int(y2)
                hazard_dist = int(hazard_dist)

                if hazard_dist == 0:
                    # used to get rid of error with stairs distance, and this gives stairs the highest priority
                    hazard_dist = 0.1
                distances.append(((center_x, y2), hazard_dist))
                detections.append({
                        'class_id': self.stairs_class_id,  # Use our stairs class ID
                        'class_name': 'stairs',
                        'confidence': confidence,
                        'bbox': (int(x1), int(y1), int(width), int(height))
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

            center_x = int(x1 + (width//2))
            center_y = int(y1 + (height//2))
            hazard_dist = self.get_distance_from_point(depth_frame, center_x, center_y)
            
            if (hazard_dist >= 0.6 and hazard_dist <= 2.2):
                if ((x1 >= CENTER_REGION[0] and x1 < CENTER_REGION[1]) or (x2 <= CENTER_REGION[1] and x2 > CENTER_REGION[0])):
                    distances.append(((center_x, center_y), hazard_dist))
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.coco_model_classes.get(class_id, 'unknown'),
                        'confidence': confidence,
                        'bbox': (int(x1), int(y1), int(width), int(height))
                    })
            # distances.append(self.get_distance_from_point(depth_frame, center_x, center_y))
            
        return detections, distances

    def find_wall_detections(self, list_of_points):
        walls = []
        #print("In wall detections \n")
        #print( "\nList of points ", list_of_points)

        #print("\n Length is: ", len(list_of_points))
        for i in range(len(list_of_points) -1 ):
            (p1, d1) = list_of_points[i]
            #print("\n point 1 :", (p1, d1)) 
            if d1 == 0:
                # skip invalid distances of 0
                continue
            (p2, d2) = list_of_points[i + 1]
            #print(" \nsecond point: ", (p2, d2), "\n\n")
            if d2 == 0:
                continue
                    # skip invalid distances of 0
                    # only compare vertically aligned points
            if abs(d1 - d2) < 0.05:
                wall_location = {
                    "mid_point": (p1, p2), 
                    "class_name": "walls",
                    "distance": d2}
                if (wall_location["distance"] > 0.6 and wall_location["distance"] <= 2.2):
                    walls.append(wall_location)
        return walls
    
                    
    # function for getting the depth distances of walls and determining if there is a wall
    def get_depth_distances(self, depth_frame):
        # convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())


         # create 9 measurement points (3 columns, 3 rows)
        height, width = depth_image.shape

        offset1 = int(.2 * width)  # offset off of the width/height
        offset2 = int(.4 * width) 

        points = [(offset1, height // 2),
                  (offset1, 100),
                  (offset2, height // 2),
                   (offset2, 100),
                  (width - offset1, height // 2),
                  (width - offset1, 100),
                  (width - offset2, height // 2),
                   (width - offset2,100)]
                  
    
        # scale depth measurement points to color
        x_scale = 1280 / 640
        y_scale = 720 / 480
        # store depths in dict
        measurement_points = {(x,y): depth_frame.get_distance(x, y) for x,y in points}

        # calculate final distances by scaling the depth distance to color distance
        final_distances = {(int(x *x_scale), int(y *y_scale)): distance for (x, y), distance in measurement_points.items()}

        # check for close distance wall pairs
        list_of_points = list(final_distances.items())

        wall_detections = self.find_wall_detections(list_of_points)
       # print("\n\n Wall Detections are: ", wall_detections) 
                    
        return final_distances, wall_detections


    
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

            if (class_name == "oven"):
                label = f"stairs ({confidence: .2f})"
                color = (255, 0, 0)
            
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
        hazard = None
        hazard_dist = None
        prev_command = "n/a"
        prev = None
        prev_dist = None
        state = "OFF"  # initialize state to be off
        
        # Add state tracking variables
        previous_fsr_state = "OFF"  # Track previous FSR state
        current_hazard_id = None    # Track unique hazard instances
        notification_sent = False   # Track if we've already sent a notification for this placement
        
        try:
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    
                    if line in ["ON", "OFF"]:
                        # FSR state has changed
                        if line == "ON" and previous_fsr_state == "OFF":
                            # Cane just placed - reset notification flag
                            notification_sent = False
                            print("Cane placed - ready for new hazard detection")
                        elif line == "OFF" and previous_fsr_state == "ON":
                            # Cane just lifted - reset hazard tracking
                            current_hazard_id = None
                            print("Cane lifted - hazard tracking reset")
                        
                        previous_fsr_state = line
                        state = line  # Keep using state for display/pausing
                
                start_time = time.time()
    
                # Process frames
                color_image, depth_image = self.process_stream()
                if color_image is None:
                    print("Failed to capture frame")
                    continue
                    
                if state == "ON":
                    # Define regions of interest
                    CENTER_REGION = (int(color_image.shape[1] * 0.4), int(color_image.shape[1] * 0.6))
                    CENTER_X = color_image.shape[1] // 2
                    bottom_y = color_image.shape[0] - 5
                    
                    # Detect objects
                    object_detections, distances = self.object_detection(color_image, depth_image, CENTER_REGION, bottom_y)
    
                    # Wall Detection
                    wall_distances, wall_detections = self.get_depth_distances(depth_image)
                    
                    # Draw distance points
                    for (x,y), distance in wall_distances.items():
                        cv2.circle(color_image, (x, y), 6, (0, 0, 255), -1)
                        cv2.putText(color_image, f"{distance:.2f}m", (x+5, y-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    
                    # Draw wall detections
                    for item in wall_detections:
                        (p1, p2) = item["mid_point"]
                        cv2.circle(color_image, p1, 10, (255, 0, 255), -1)
                        cv2.circle(color_image, p2, 10, (255, 0, 255), -1)
    
                    # Downstairs Detection
                    downstairs_x = color_image.shape[1]// 2
                    downstairs_y = int((3 * color_image.shape[0]) / 4)
    
                    distance_downstairs1 = self.get_distance_from_point(depth_image, downstairs_x - 50, downstairs_y - 100)
                    distance_downstairs2 = self.get_distance_from_point(depth_image, downstairs_x + 50, downstairs_y - 100)
                    distance_downstairs3 = self.get_distance_from_point(depth_image, downstairs_x - 50, downstairs_y)
                    distance_downstairs4 = self.get_distance_from_point(depth_image, downstairs_x + 50, downstairs_y)
                    
                    # Determine if there's a hazard and what kind
                    if ((distance_downstairs1 == 0 and distance_downstairs2 == 0 and 
                         distance_downstairs3 == 0 and distance_downstairs4 == 0) or 
                        (distance_downstairs1 > 1.3 and distance_downstairs2 > 1.3 and 
                         distance_downstairs3 > 0.7 and distance_downstairs4 > 0.7)):
                        # Downstairs detected
                        command = "stairs down"
                        hazard = {"class_name": "stairs down"}
                        hazard_dist = min(filter(lambda x: x > 0, [distance_downstairs1, distance_downstairs2, 
                                                                  distance_downstairs3, distance_downstairs4] + [999]))
                        hazard_id = "stairs_down"
                    elif object_detections or wall_detections:
                        # Object or wall detected
                        command, hazard, hazard_dist = decide_haptic_response(object_detections, distances, 
                                                                             CENTER_X, CENTER_REGION, wall_detections)
                        
                        # Generate a unique identifier for this hazard
                        if hazard is not None:
                            if "class_name" in hazard:
                                hazard_class = hazard["class_name"]
                                # Create a unique ID based on class and approximate position
                                if "bbox" in hazard:
                                    x, y, w, h = hazard["bbox"]
                                    hazard_id = f"{hazard_class}_{x//50}_{y//50}"
                                elif "mid_point" in hazard:  # For walls
                                    center = get_center_point(hazard["mid_point"])
                                    hazard_id = f"{hazard_class}_{center[0]//50}_{center[1]//50}"
                            else:
                                hazard_id = None
                    else:
                        # No hazard detected
                        command = "none"
                        hazard = None
                        hazard_dist = None
                        hazard_id = None
                    
                    # Determine whether to send haptic feedback
                    should_send = False
                    
                    # Only send a notification if:
                    # 1. FSR state is ON (cane is placed)
                    # 2. We haven't sent a notification for this placement yet
                    # 3. There is a hazard to report
                    if previous_fsr_state == "ON" and not notification_sent and hazard is not None:
                        should_send = True
                        notification_sent = True  # Mark that we've sent a notification for this placement
                        current_hazard_id = hazard_id
                        print(f"New hazard detected: {command}")
                    
                    # Send the haptic command if needed
                    if should_send:
                        send_haptic_command(command, prev_command, hazard, prev, hazard_dist, prev_dist)
                        prev_command = command
                        prev = hazard
                        prev_dist = hazard_dist
                    
                    # Draw detections
                    annotated_image = self.draw_bounding_boxes(color_image.copy(), object_detections, distances)
    
                    # Draw downstairs dots
                    cv2.circle(annotated_image, (downstairs_x - 50, downstairs_y - 100), 5, (255, 255, 255), -1)
                    cv2.putText(annotated_image, f"{distance_downstairs1:.2f}m", (downstairs_x - 50 - 10, downstairs_y - 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.circle(annotated_image, (downstairs_x + 50, downstairs_y - 100), 5, (255, 255, 255), -1)
                    cv2.putText(annotated_image, f"{distance_downstairs2:.2f}m", (downstairs_x + 50 - 10, downstairs_y - 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.circle(annotated_image, (downstairs_x - 50, downstairs_y), 5, (255, 255, 255), -1)
                    cv2.putText(annotated_image, f"{distance_downstairs3:.2f}m", (downstairs_x - 50 - 10, downstairs_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.circle(annotated_image, (downstairs_x + 50, downstairs_y), 5, (255, 255, 255), -1)
                    cv2.putText(annotated_image, f"{distance_downstairs4:.2f}m", (downstairs_x + 50 - 10, downstairs_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Calculate FPS
                    fps = 1.0 / (time.time() - start_time)
                    
                    # Add FSR state and current command to display
                    cv2.putText(
                        annotated_image, 
                        f"FPS: {fps:.2f}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
                    
                    cv2.putText(
                        annotated_image, 
                        f"FSR: {previous_fsr_state}", 
                        (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
                    
                    cv2.putText(
                        annotated_image, 
                        f"CMD: {command if should_send else 'none'}", 
                        (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
                    
                    cv2.imshow("Object Detection", annotated_image)
                else:
                    cv2.putText(color_image, "Model Paused", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    cv2.imshow("Object Detection", color_image)
                    
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
def send_haptic_command(command, prev_command, hazard, prev, hazard_dist, prev_dist):
    if command != "none":
        ser.write((command + "\n").encode())
        print(f"Haptic command {command} sent")

def decide_object_action(nearest_hazard, second_hazard, CENTER_X):
    action = "none"

    # Using global CENTER_REGION
    if (nearest_hazard == None): return 'none'
    x_nearest, y_nearest, w_nearest, h_nearest = nearest_hazard['bbox']
    x_center_nearest = (x_nearest + (x_nearest + w_nearest))//2
    x_right_nearest = x_nearest + w_nearest
    x_left_nearest = x_nearest

    if (second_hazard == None):
        if (x_center_nearest <= CENTER_X):
            return "object turn right"
        else:
            return "object turn left"
    
    x_second, y_second, w_second, h_second = second_hazard['bbox']
    x_center_second = (x_second + (x_second + w_second))//2
    x_right_second = x_second + w_second
    x_left_second = x_second
    
    if (x_center_nearest <= CENTER_X):
        if (x_center_second <= CENTER_X):
            action = "object turn right"
        else:
            action = "object turn left"
    else: 
        if (x_center_second >= CENTER_X):
            action = "object turn left"
        else:
            action = "object turn right"
    return action


""" Deciding haptic response (need to get wall data and distance data for this function)
based on the x location of nearest hazard in frame (in center third of frame)
"""
def decide_haptic_response(detections, distances, CENTER_X, CENTER_REGION, wall_detections):
    nearest_hazard = None
    second_nearest = None
    nearest_distance = None
    second_distance = None
    # detections [((),())]
    for i in range(len(detections)):
        detection = detections[i]
        curr_distance = distances[i][1]
    # for detection in detections:
        x, y, w, h = detection['bbox']
        x_center = (x + (x + w))//2
        x_right = x + w
        x_left = x
        
        if (CENTER_REGION[0] < x_right or x_left < CENTER_REGION[1]):
            if (detection["class_name"] == "stairs"):
                return "stairs", detection, curr_distance
            elif (detection["class_name"] == "oven"):
                return "stairs", detection, curr_distance
            if (curr_distance != 0 and (nearest_distance == None or curr_distance <= nearest_distance)):
                second_distance = nearest_distance
                nearest_distance = curr_distance
                second_nearest = nearest_hazard
                nearest_hazard = detection 
            elif (curr_distance != 0 and (second_distance == None or curr_distance <= second_distance)):
                second_distance = curr_distance
                second_nearest = detection
        else:
            if (curr_distance != 0 and (second_distance == None or curr_distance <= second_distance)):
                second_distance = curr_distance
                second_nearest = detection

    print("Nearest Hazard is: \n", nearest_hazard, "\n")
    print("Nearest Distance is : \n", nearest_distance, "\n")
    print("Wall Detections ", wall_detections)
    # look through wall detections
    for wall in wall_detections:
        wall_dist = wall["distance"]
        if (nearest_hazard is None or wall_dist < nearest_distance):
            if (not wall_in_obj(wall, nearest_hazard)):
                nearest_distance = wall_dist
                nearest_hazard = wall
    print("   After wall search, nearest hazard is", nearest_hazard)
            
    # Deciding object haptic response to send
    if (nearest_hazard == None):
        return "none", None, None
    if (nearest_hazard["class_name"] == "stairs"):
        return "stairs", nearest_hazard, nearest_distance
    elif (nearest_hazard["class_name"] == "walls"):
        print("\n \n\nWall action function...         ")
        return decide_wall_action(nearest_hazard, CENTER_X), nearest_hazard, nearest_distance
    else:
        return decide_object_action(nearest_hazard, second_nearest, CENTER_X), nearest_hazard, nearest_distance

def wall_in_obj(wall, nearest_hazard):
    if (wall == None or nearest_hazard == None):
        return False
    elif (nearest_hazard["class_name"] == "walls"):
        return False
    wall_x, wall_y = get_center_point(wall["mid_point"])
    x1, y1, w, h = nearest_hazard["bbox"]
    x2 = x1 + w
    y2 = y1 + h

    if (wall_x >= x1 and wall_x <= x2):
        if (wall_y >= y1 and wall_y <= y2):
            return True
    return False

# decides the proper move after detecting a wall
def decide_wall_action(nearest_hazard, CENTER_X):
    center_pt = get_center_point(nearest_hazard["mid_point"])
    detection_x = center_pt[0]

    print("CENTER_X", CENTER_X)
    print("\n detection_x : ", detection_x)
    if (detection_x < CENTER_X):
        return "wall turn right"
    else:
        return "wall turn left"

# gets the center of two wall detection points 
def get_center_point(two_points):
    p1, p2 = two_points
    x1, y1 = p1
    x2, y2 = p2
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return (center_x,center_y) 

def wait_for_qtpy():
    while True:
        try:
            ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            return ser
        except serial.SerialException:
            time.sleep(1)

if __name__ == "__main__":
    # Connecting to the haptics motor controller
    # ser = serial.Serial('/dev/ttyACM0', 9600)
    ser = wait_for_qtpy()
    time.sleep(2)
        

    detector = Camera_object_detection(
        stairs_model_path="best_aug.pt",
        coco_model_path="yolov8n.pt",
        confidence_threshold=0.55
    )
    detector.run_detection()

