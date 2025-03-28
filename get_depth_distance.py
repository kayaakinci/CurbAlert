import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # start the depth stream
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # start streaming
    pipeline.start(config)
    
    try:
        while True:
            # Wait for the depth frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                  
                # convert depth frame to numpy array
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # create 9 measurement points (3 columns, 3 rows)
                height, width = depth_image.shape
                thirds_x = [width // 4, width // 2, (3 * width) // 4]
                thirds_y = [height // 4, height // 2, (3 * height) // 4]
                measurement_points = [(x, y) for x in thirds_x for y in thirds_y]
                
                # Get distances at the selected points
                distances = {point: depth_frame.get_distance(point[0], point[1]) for point in measurement_points}
                
                # Display distances on image
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                for (x, y), distance in distances.items():
                    cv2.circle(depth_colormap, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(depth_colormap, f"{distance:.2f}m", (x - 30, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Show depth image
                cv2.imshow('Depth Image', depth_colormap)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit(1)
    
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
