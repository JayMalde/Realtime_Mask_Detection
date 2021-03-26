import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model('model.h5')

class Camera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        global model
        success, image = self.video.read()
        image=cv2.resize(image,(150,150),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.1,6)
        for (x,y,w,h) in face_rects:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            prediction = model.predict(np.expand_dims(image , axis=0))
            if (prediction == 1):
               cv2.rectangle(image , (x,y) , (x+w,y+h) , (0,0,255) , 2)     # No Masks
               cv2.putText(image, text='NO MASK', org=(x,y-2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)
            else:
               cv2.rectangle(image , (x,y) , (x+w,y+h) , (0,255,0) , 2)     # Masks
               cv2.putText(image, text='MASK', org=(x,y-2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
