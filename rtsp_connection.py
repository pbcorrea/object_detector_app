import cv2
import numpy as np

stream_ip=str('rtsp://admin:12345678@10.10.10.10/stream0/mobotix.h264')

while True:
    cap = cv2.VideoCapture(stream_ip)
    ret, frame = cap.read()
    if ret == False:
        print("Frame is empty")
        break
    else:
        cv2.imshow('Recording',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
print('Video OK')
cap.release()
cv2.destroyAllWindows()
    