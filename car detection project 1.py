# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:50:20 2023

@author: Lenovo
"""

import cv2
import time
import numpy as np
import cv2

# Load the image or capture video frames

frame = cv2.imread(r"C:\Users\Lenovo\OneDrive\Desktop\deep learning extract  flies\los_angeles.mp4")

if frame is not None:
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
else:
    print("passed  to load the image or capture frame.")
# Create our body classifier
car_classifier = cv2.CascadeClassifier(r"C:\Users\Lenovo\OneDrive\Desktop\deep learning extract  flies\Haarcascades\haarcascade_car.xml")

# Initiate video capture for video file
cap = cv2.VideoCapture(r"C:\Users\Lenovo\OneDrive\Desktop\deep learning extract  flies\los_angeles.mp4")


# Loop once video is successfully loaded
while cap.isOpened():
    
    time.sleep(.05)
    # Read first frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Pass frame to our car classifier
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Cars', frame)

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
