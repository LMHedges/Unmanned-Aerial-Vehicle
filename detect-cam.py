from ultralytics import YOLO
import cv2 as cv
import os
import numpy as np

# Load the custom YOLOv8 model
model = YOLO('51ep-16-GPU.pt')

count = 0

for item in os.listdir("test-video/"):
    
    if item == ".DS_Store":
    
        continue
    
    cam = cv.VideoCapture("test-video/"+item)
    
    width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    
    height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    fps = cam.get(cv.CAP_PROP_FPS)
            
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    
    out = cv.VideoWriter(item, fourcc, fps, (width, height))

    while True:
        
        ret, frame = cam.read()
        
        if len(np.shape(frame)) == 0:
            
            continue
        
        result = model(frame)
        
        detect = result[0].plot()
        
        cv.imshow("detect",detect)
        
        out.write(detect)
        
        if cv.waitKey(1) & 0xFF == ord("q"):
            
            break
            
print(count, "<<<<<<<<<<<<<<<<<<<<-END->>>>>>>>>>>>>>>>>>>")