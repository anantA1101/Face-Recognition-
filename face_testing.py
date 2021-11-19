import cv2
from keras.models import load_model
from keras.preprocessing import image 
import numpy as np
from PIL import Image 

model= load_model(r'D:\projects\projects recognition\Face-Recognition-Using-Transfer-Learning-master\face_recognition.h5')
face_cascade = cv2.CascadeClassifier(r'D:\projects\projects recognition\Face-Recognition-Using-Transfer-Learning-master\haarcascade_frontalface_default.xml')

def face_extractor(img):
    faces= face_cascade.detectMultiScale(img,1.3,5)

    if faces is():
        return None

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)

        cropped_face=img[y:y+h,x:x+w]

    return cropped_face

video_capture=cv2.VideoCapture(0)

while True:
    _,frame=video_capture.read()
    face=face_extractor(frame)
    if type(face) is np.ndarray:
        
        face=cv2.resize(face,(224,224))
        im= Image.fromarray(face,'RGB')
        img_array=np.array(im)
        img_array=np.expand_dims(img_array,axis=0)
        pred=model.predict(img_array)
        print(pred)
        name='None Matching'

        if (pred[0][0]>0.5):
            name=" Anant"
        cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        if (pred[0][1]>0.5):
            name="Avi"
        cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        if (pred[0][2]>0.5):
            name="Devesh"
        cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        if (pred[0][3]>0.5):
            name="Gathra"
        cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        """ if (pred[0][4]>0.5):
            name="Monica"
        cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) """
        if (pred[0][4]>0.5):
            name="Neha"
        cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    else:
        cv2.putText(frame,"No Face Found",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
         break


video_capture.release()
cv2.destroyAllWindows()
