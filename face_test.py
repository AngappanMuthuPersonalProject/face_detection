from VideoCamera import VideoCamera
from FaceDetector import FaceDetector
from face_utils import *
from time import sleep as delay
import cv2

webcam = VideoCamera('rtsp://admin:123456@192.168.1.22/H264?ch=1&subtype=0')
detector = FaceDetector('Cascades/haarcascade_frontalface_default.xml')
while True:
    frame = webcam.get_frame(True)
    faces_coord = detector.detect(frame)
    if len(faces_coord):
        faces = normalize_faces(frame,faces_coord)
        cv2.imshow('frame',faces[0])
    k = cv2.waitKey(10) & 0xFF
    if(k ==27):
        break
del webcam
cv2.destroyAllWindows()



