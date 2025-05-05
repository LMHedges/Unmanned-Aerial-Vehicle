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

# Set webcam resolution to 640x480 for better image clarity
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
if width == 0 or height == 0:
    print("Error: Invalid webcam resolution. Check webcam compatibility.")
    cam.release()
    ser.close()
    exit()
print(f"Webcam initialized: {width}x{height}")

def map_to_servo(x, y, img_width, img_height):
    # Map pixel coordinates to servo angles
    # X servo: 0 to 180 degrees (left to right)
    # Y servo: 25 to 100 degrees (bottom to top)
    x_servo = int(np.clip((x / img_width) * 180, 0, 180))
    y_servo = int(np.clip(25 + ((img_height - y) / img_height) * (100 - 25), 25, 100))
    return x_servo, y_servo

def clear_frame_buffer(cam):
    # Clear webcam buffer by reading the latest frame
    for _ in range(30):  # Read up to 30 frames to ensure buffer is cleared
        cam.grab()  # Grab without decoding to clear buffer
    ret, frame = cam.read()  # Read the latest frame
    return ret, frame

# Variables to persist detection state
last_state = 'S'
last_x_servo = 0
last_y_servo = 25
no_detection_count = 0
MAX_NO_DETECTION_FRAMES = 1  # Revert to SEARCHING after 1 frame of no detection
frame_count = 0
waiting_for_center_check = False  # Flag to check centering after moving in TARGETING

try:
    while True:
        ret, frame = cam.read()
        if not ret or frame is None or len(np.shape(frame)) == 0:
            print("Error: Failed to capture frame. Check webcam connection or try another index.")
            continue

        frame_count += 1
        # Process every 4th frame to reduce backlog
        process_frame = (frame_count % 4 == 0)

        if process_frame:
            # Run YOLO model on the frame
            results = model(frame)
            detect = results[0].plot()

            # Initialize state and angles
            state = last_state
            x_servo, y_servo = last_x_servo, last_y_servo

            print(f"Processing frame {frame_count}, Resolution: {width}x{height}, Current state: {state}")

            # Process detections
            if len(results[0].boxes) > 0:
                # Get the first detected drone (highest confidence)
                box = results[0].boxes[0]
                confidence = box.conf.item()
                print(f"Initial Detection: Confidence={confidence:.2f}, Box={box.xyxy[0].tolist()}")
                if confidence >= 0.2:  # High confidence for TARGETING/LOCKED_ON
                    # Store initial detection coordinates as fallback
                    x_center_initial = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                    y_center_initial = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                    x_servo_initial, y_servo_initial = map_to_servo(x_center_initial, y_center_initial, width, height)

                    # Clear frame buffer to ensure fresh frame
                    print("Clearing frame buffer due to detection")
                    ret, frame = clear_frame_buffer(cam)
                    if not ret:
                        print("Error: Failed to clear frame buffer")
                        continue
                    # Re-run YOLO on the fresh frame
                    results = model(frame)
                    detect = results[0].plot()
                    if len(results[0].boxes) == 0:
                        print("No detection in fresh frame after buffer clear, using initial detection")
                        # Fallback to initial detection
                        x_servo, y_servo = x_servo_initial, y_servo_initial
                        if last_state == 'S':
                            state = 'T'
                            waiting_for_center_check = True
                            print("Transitioning to TARGETING using initial detection")
                        elif last_state == 'T':
                            if waiting_for_center_check:
                                if abs(x_center_initial - width/2) < 100 and abs(y_center_initial - height/2) < 100:
                                    state = 'L'
                                    print("Transitioning to LOCKED_ON using initial detection")
                                else:
                                    state = 'T'
                                    print("Staying in TARGETING: Not centered (initial detection)")
                                waiting_for_center_check = False
                            else:
                                state = 'T'
                                waiting_for_center_check = True
                                print("Moving to new center in TARGETING (initial detection)")
                        elif last_state == 'L':
                            state = 'L'
                            print("Holding LOCKED_ON (initial detection)")
                        print(f"State updated to: {state}, Servo: x={x_servo}, y={y_servo}")
                        no_detection_count = 0
                    else:
                        # Process the fresh detection
                        box = results[0].boxes[0]
                        confidence = box.conf.item()
                        print(f"Fresh Detection: Confidence={confidence:.2f}, Box={box.xyxy[0].tolist()}")
                        if confidence >= 0.2:
                            # Calculate bounding box center
                            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                            y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                            print(f"Center: x={x_center:.1f}, y={y_center:.1f}")
                            # Map to servo angles
                            x_servo, y_servo = map_to_servo(x_center, y_center, width, height)

                            # State logic
                            if last_state == 'S':
                                state = 'T'
                                waiting_for_center_check = True
                                print("Transitioning to TARGETING")
                            elif last_state == 'T':
                                if waiting_for_center_check:
                                    if abs(x_center - width/2) < 100 and abs(y_center - height/2) < 100:
                                        state = 'L'
                                        print("Transitioning to LOCKED_ON")
                                    else:
                                        state = 'T'
                                        print("Staying in TARGETING: Not centered")
                                    waiting_for_center_check = False
                                else:
                                    state = 'T'
                                    waiting_for_center_check = True
                                    print("Moving to new center in TARGETING")
                            elif last_state == 'L':
                                state = 'L'
                                print("Holding LOCKED_ON")
                            print(f"State updated to: {state}, Servo: x={x_servo}, y={y_servo}")
                            no_detection_count = 0
                        else:
                            print(f"Fresh Confidence too low ({confidence:.2f} < 0.2), using initial detection")
                            # Fallback to initial detection
                            x_servo, y_servo = x_servo_initial, y_servo_initial
                            if last_state == 'S':
                                state = 'T'
                                waiting_for_center_check = True
                                print("Transitioning to TARGETING using initial detection")
                            elif last_state == 'T':
                                if waiting_for_center_check:
                                    if abs(x_center_initial - width/2) < 100 and abs(y_center_initial - height/2) < 100:
                                        state = 'L'
                                        print("Transitioning to LOCKED_ON using initial detection")
                                    else:
                                        state = 'T'
                                        print("Staying in TARGETING: Not centered (initial detection)")
                                    waiting_for_center_check = False
                                else:
                                    state = 'T'
                                    waiting_for_center_check = True
                                    print("Moving to new center in TARGETING (initial detection)")
                            elif last_state == 'L':
                                state = 'L'
                                print("Holding LOCKED_ON (initial detection)")
                            print(f"State updated to: {state}, Servo: x={x_servo}, y={y_servo}")
                            no_detection_count = 0
                else:
                    print(f"Initial Confidence too low ({confidence:.2f} < 0.2)")
                    no_detection_count += 1
                    if no_detection_count >= MAX_NO_DETECTION_FRAMES:
                        state = 'S'
                        print("Reverting to SEARCHING: No detection for MAX_NO_DETECTION_FRAMES")
                    else:
                        state = last_state
                        print(f"Staying in {state}: Low confidence")
            else:
                print("No detection in this frame")
                no_detection_count += 1
                if last_state == 'L':
                    state = 'S'  # Revert to SEARCHING immediately if no detection in LOCKED_ON
                    print("Reverting to SEARCHING: No detection in LOCKED_ON")
                elif no_detection_count >= MAX_NO_DETECTION_FRAMES:
                    state = 'S'
                    print("Reverting to SEARCHING: No detection for MAX_NO_DETECTION_FRAMES")
                else:
                    state = last_state
                    print(f"Staying in {state}: No detection")

            # Update last known state and angles
            last_state = state
            last_x_servo = x_servo
            last_y_servo = y_servo

            # Send commands only for TARGETING or LOCKED_ON
            if state in ['T', 'L']:
                command = f"{state},{x_servo},{y_servo}\n"
                print(f"Sending command: {command.strip()}")
                ser.flush()
                ser.write(command.encode())
                time.sleep(0.1)  # Delay for reliable processing
            else:
                print("No command sent: State is SEARCHING")

            # Read Arduino response (if any)
            if ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                if response:
                    print(f"Arduino response: {response}")

            # Video feed disabled to improve performance with higher resolution
            # To re-enable, uncomment the following:
            # display_frame = cv.resize(detect, (0, 0), fx=0.5, fy=0.5)
            # try:
            #     cv.imshow("Drone Detection", display_frame)
            #     if cv.getWindowProperty("Drone Detection", cv.WND_PROP_VISIBLE) <= 0:
            #         print("Warning: Display window is not visible.")
            # except cv.error as e:
            #     print(f"Error displaying frame: {e}")

            # Cap frame rate with fixed delay
            time.sleep(0.1)

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