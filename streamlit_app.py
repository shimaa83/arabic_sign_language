
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Input,Dense ,Dropout,Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.applications.vgg19 import VGG19
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import os
import cv2
import streamlit as st
from PIL import Image
import numpy as np
model= load_model('D://Omdena//sign//keras_code//sign.h5',compile=False)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
st.title("Convert sign language to text")
    # Create a VideoCapture object
cap = cv2.VideoCapture(0) 
    
    
    # Set dimensions of the camera frame (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Add a button to take a photo
if st.button("Take Photo"):
        ret, frame = cap.read()
        if ret:
            # Save the photo
            cv2.imwrite('D://Omdena//sign//keras_code//my_photo.jpg', frame)
            image = Image.open("D://Omdena//sign//keras_code//my_photo.jpg")
            st.image(image, caption="your image ")

            z=[]
            img_path = 'D:/Omdena/sign/keras_code/my_photo.jpg' 
            orignal_image = cv2.imread(img_path)
            image = cv2.cvtColor(orignal_image, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(image, (224,224))
            z.append(resized_image)
            z= np.array(z)
            z=z/255.0
            predictions = model.predict(z)

# Get the index of the predicted label
            predicted_label_index = tf.argmax(predictions, axis=-1).numpy()[0]
            mydict={'شهر': 0, 'جد': 1,'يشرب':2,'صفر':3,'اسف':4,'ام':5,'ينام':6,'اب':7,'ياكل':8,'اليوم':9}
            st.write(':الكلمة المتوقعة هى ')
            st.write(list(mydict.keys())[list(mydict.values()).index(predicted_label_index)])

