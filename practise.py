import cv2
import numpy as np 

object_tracking = cv2.createBackgroundSubtractorMOG2()

def nothing():
    pass


cap= cv2.VideoCapture(0)
# creating a tracker 
cv2.namedWindow("Tracking")

cv2.createTrackbar("LH", "Tracking", 0,255, nothing)
cv2.createTrackbar("LS", "Tracking", 0,255, nothing)
cv2.createTrackbar("LV", "Tracking", 0,255, nothing)
cv2.createTrackbar("UH", "Tracking", 255,255, nothing)
cv2.createTrackbar("US", "Tracking", 255,255, nothing)
cv2.createTrackbar("UV", "Tracking", 255,255, nothing)


while True:
    
    ret, frame = cap.read()
    hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    l_h  = cv2.getTrackbarPos("LH", "Tracking")
    l_s  = cv2.getTrackbarPos("LS", "Tracking")
    l_v  = cv2.getTrackbarPos("LV", "Tracking")

    u_h  = cv2.getTrackbarPos("UH", "Tracking")
    u_s  = cv2.getTrackbarPos("US", "Tracking")
    u_v  = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s , l_v])
    u_b = np.array([u_h, u_s , u_v])

    mask = cv2.inRange(hsv, l_b, u_b)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    mask_1=object_tracking.apply(res)
    contours,_ = cv2.findContours(mask_1 , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area= cv2.contourArea(cnt)
        if area>500:
            cv2.drawContours(mask, [cnt] , -1, (0,0,255),2)
            cv2.drawContours(frame, [cnt] , -1, (0,0,255),2)
            x,y,w,h= cv2.boundingRect(cnt)
            print ("x= "+ str(x)+" y= "+ str(y)+" w= "+ str(w)+" h= "+ str(h))
            cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0,3))

    cv2.imshow("cam", frame)
    cv2.imshow("cam_hsv", hsv)
    cv2.imshow("with mask", res )
    cv2.imshow("Object TRACKING ", mask_1)
    

    key= cv2.waitKey(40)
    if key== 27:
        break

cv2.release()
cv2.destroyAllWindows()





























""" img= cv2.imread("D:\projects\practise\sample.jpg",1)
print (img.shape)
resized_img= cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
cv2.imshow("sample",resized_img)
cv2.waitKey(2000)
cv2.destroyAllWindows()

img_1=cv2.imread("D:\projects\practise\sample.jpg",0)
print (img_1.shape)
resized_img_1= cv2.resize(img_1,(int(img_1.shape[1]/2),int(img_1.shape[0]/2)))
cv2.imshow("sample_in_B&W",resized_img_1)
cv2.waitKey(2000)
cv2.destroyAllWindows()
 """