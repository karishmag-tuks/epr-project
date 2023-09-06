# import libraries
import numpy as np
import cv2

capture_video = cv2.VideoCapture(0)
 
while(True):
    ret, frame = capture_video.read() 
    if ret==True: 
        face = frame[:300,200:500]
        cv2.imshow('frame', face)
        eyes = face[100:200,:]
        cv2.imshow("eyes", eyes)
        cv2.imshow("left eye", eyes[:,:150])
        cv2.imshow("right eye", eyes[:,150:])
        # cv2.imshow("right eye", face[:150,160:] )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_video.release()
cv2.destroyAllWindows()
