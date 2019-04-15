import cv2

stream_ip=str('rtsp://admin:12345678@10.10.10.10/stream0/mobotix.mjpeg')

while True:
    cap = cv.VideoCapture(stream_ip)
    ret, frame = cap.read()
    cv2.imshow('Recording',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('Video OK')
cap.release()
cv2.destroyAllWindows()
    