from ultralytics import YOLO
import cv2 as cv
import numpy as np
import serial
import time

# Initialize serial communication with Arduino
try:
    ser = serial.Serial('COM3', 9600, timeout=1)  # COM3 for Arduino Uno
    time.sleep(2)  # Wait for Arduino to initialize
except serial.SerialException as e:
    print(f"Error: Could not open serial port: {e}")
    exit()

# Load the custom YOLOv8 model
model = YOLO('51ep-16-GPU.pt')

# Initialize the USB webcam
cam = cv.VideoCapture(1)  # Webcam index 1
if not cam.isOpened():
    print("Error: Could not open webcam. Check if the webcam is connected and not in use by another application.")
    print("Try changing the webcam index or test with another device.")
    ser.close()
    exit()

# Get webcam resolution (default)
width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
if width == 0 or height == 0:
    print("Error: Invalid webcam resolution. Check webcam compatibility.")
    cam.release()
    ser.close()
    exit()
print(f"Webcam initialized: {width}x{height}")

# Save a test frame to verify capture
ret, test_frame = cam.read()
if ret and test_frame is not None:
    cv.imwrite("test_frame.jpg", test_frame)
    print("Test frame saved as test_frame.jpg")
else:
    print("Warning: Could not save test frame. Capture may be failing.")

def map_to_servo(x, y, img_width, img_height):
    # Map pixel coordinates to servo angles
    # X servo: 0 to 180 degrees (left to right)
    # Y servo: 25 to 100 degrees (bottom to top)
    x_servo = int(np.clip((x / img_width) * 180, 0, 180))
    y_servo = int(np.clip(25 + ((img_height - y) / img_height) * (100 - 25), 25, 100))
    return x_servo, y_servo

try:
    frame_count = 0
    while True:
        ret, frame = cam.read()
        if not ret or frame is None or len(np.shape(frame)) == 0:
            print("Error: Failed to capture frame. Check webcam connection or try another index.")
            continue

        frame_count += 1
        # Process every other frame to reduce load
        if frame_count % 2 == 0:
            # Run YOLO model on the frame
            results = model(frame)
            detect = results[0].plot()

            # Initialize default state and angles
            state = 'S'  # SEARCHING
            x_servo, y_servo = 0, 25  # Default servo positions

            # Process detections
            if len(results[0].boxes) > 0:
                # Get the first detected drone (highest confidence)
                box = results[0].boxes[0]
                confidence = box.conf.item()
                print(f"Detection: Confidence={confidence:.2f}, Box={box.xyxy[0].tolist()}")
                if confidence > 0.3:  # Lowered threshold
                    # Calculate bounding box center
                    x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                    y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                    print(f"Center: x={x_center:.1f}, y={y_center:.1f}")
                    # Map to servo angles
                    x_servo, y_servo = map_to_servo(x_center, y_center, width, height)
                    # Set state to LOCKED_ON if centered, else TARGETING
                    state = 'L' if abs(x_center - width/2) < 50 and abs(y_center - height/2) < 50 else 'T'
                    print(f"State: {state}, Servo: x={x_servo}, y={y_servo}")

            # Send state and angles to Arduino
            command = f"{state},{x_servo},{y_servo}\n"
            print(f"Sending command: {command.strip()}")
            ser.write(command.encode())
            time.sleep(0.01)  # Small delay to prevent buffer overflow

            # Display the frame with detections
            try:
                cv.imshow("Drone Detection", detect)
                if cv.getWindowProperty("Drone Detection", cv.WND_PROP_VISIBLE) <= 0:
                    print("Warning: Display window is not visible. It may be minimized or off-screen.")
            except cv.error as e:
                print(f"Error displaying frame: {e}")

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stream interrupted.")

finally:
    # Send SEARCHING command before closing
    ser.write("S,0,25\n".encode())
    # Release resources
    cam.release()
    ser.close()
    cv.destroyAllWindows()
    print("Webcam feed and serial connection terminated.")