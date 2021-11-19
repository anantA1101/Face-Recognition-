import numpy as np
import cv2

#haar cascase algorithim has to be loaded for the detection of the face etc 
face_classifier = cv2.CascadeClassifier(r'D:\projects\projects recognition\Face-Recognition-Using-Transfer-Learning-master\haarcascade_frontalface_default.xml')

# now we create functions for feature extractors 

# fist function : detecting face and returning cropped face if no face detected then returns the input face 

def face_extractor(img):
    faces= face_classifier.detectMultiScale(img,1.3,5)

    if faces is():
        return None

    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face=img[y:y+h+50,x:x+w+50]

    return cropped_face


#initializing web cam 0: if internal web cam; 1: if exteral webcam 

cap=cv2.VideoCapture(0)
count = 0

# now we read the camera input and store it in return frame as shown 

while True:
    ret,frame=cap.read()                                                    # getting frames from the camera from webcam

    if face_extractor(frame) is not None:                                   # this takes those frames and checks for a face in the face_extractor function 
        count +=1                                                           # if face is found then the count value goes up 
        face=cv2.resize(face_extractor(frame),(400,400))                    # and a frame shot is taken and then we resize it accordingly
        
        # now we have to store the taken frame for that we create a path for storing and then using cv2.imwrite we store the face into that path 

        file_name_path= r'D:\projects\projects recognition\Face-Recognition-Using-Transfer-Learning-master\images\validation\anant/' + str(count) + '.jpg' 
        cv2.imwrite(file_name_path,face)

        #next we put count on images and display live count 
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)          # to put a text on the photo   
        cv2.imshow('Face Cropper', face )                                                    # this commad is to display the face to us 

    else:
        print ('Face Not Found ')
        pass

    if cv2.waitKey(1)==13 or count ==100:               # this is when we want to stop the program that is when we press 'enter' which has the code 13 or when there are 100 images taken  
        break
 
cap.release()                                           #This is to reease the webcam  
cv2.destroyAllWindows()                                 # this is a safety command to close the windows later 
print ("collecting Samples Complete ")             
     
