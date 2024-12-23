from ultralytics import YOLO
import cv2 as cv
import os
import numpy as np

# Load the custom YOLOv8 model
model = YOLO('51ep-16-GPU.pt') 

count = 0

for item in os.listdir("test-img"):
    
    img = cv.imread("test/"+item)
    
    if len(np.shape(img)) == 0:
        
        continue

    print("Image -------",item)

    results = model(img)

    annotated_frame = results[0].plot()

    cv.imshow("Detected", annotated_frame)

    cv.waitKey()
    
    count += 1
    
print(count, "<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>")