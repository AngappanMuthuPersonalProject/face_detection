import glob
import cv2
from face_utils import *
from FaceDetector import FaceDetector

detector = FaceDetector('Cascades/haarcascade_frontalface_default.xml')

class VideoCamera(object):
    def __init__(self, index=0):  # 'rtsp://admin:123456@192.168.1.22/H264?ch=1&subtype=0'
        self.index = index
        self.video = cv2.VideoCapture(index)
        self.video.set(3,1280)
        self.video.set(4,720)
        print(str(self.video.isOpened()))

    def __del__(self):
        self.video.release()

    def get_frame(self, in_grayScale=False):
        ret, frame = self.video.read()
        if in_grayScale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def Image2Face(self,foldername):
        for img in glob.glob(foldername + '/*.*'):
            frame = cv2.imread(img)
            faces_coord = detector.detect(frame)
            faces = normalize_faces(frame, faces_coord)
            if len(faces_coord):
                crop_image = faces[0]
                cv2.imwrite("ouputs/" + str(img) + ".jpg", crop_image)
                cv2.imshow('image', crop_image)
                cv2.waitKey(10)

