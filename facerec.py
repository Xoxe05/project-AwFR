import numpy as np 
import cv2
from PIL import ImageGrab


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

a= 0 

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        img_item = "Photos\opencv_frame_{}.png".format(a)
        cv2.imwrite(img_item, roi_gray)
        a+=1

        color = (0, 255, 0)
        stroke = 1
        end_cord_x = x + w
        end_cord_y = y + h 
        cv2.rectangle(frame,(x,y), (end_cord_x, end_cord_y), color, stroke )
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
