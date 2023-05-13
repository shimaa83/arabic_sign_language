import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

model = YOLO('D:/Omdena/sign/new500image/best.pt')

picture = st.camera_input("First, take a picture...")

if picture:
    with open ('D:/Omdena/sign/new500image/test.jpg','wb') as file:
        file.write(picture.getbuffer())
result=model.predict('D:/Omdena/sign/new500image/test.jpg', imgsz=320, conf=0.10,show=True)
for r in result:
    for c in r.boxes.cls:
          st.title(model.names[int(c)]) 



    