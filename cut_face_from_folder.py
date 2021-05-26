import glob
import cv2
from FaceDetector import FaceDetector
from VideoCamera import VideoCamera
from face_utils import *

detector = FaceDetector('Cascades/haarcascade_frontalface_default.xml')


def Image2Face(foldername):
    count = 0
    for img in glob.glob(foldername + '/*.*'):
        frame = cv2.imread(img)
        faces_coord = detector.detect(frame)
        faces = pre_normalize_faces(frame, faces_coord)
        if len(faces_coord):
            count += 1
            crop_image = faces[0]
            cv2.imwrite("outputs/" + str(count) + ".jpg", crop_image)
            print("\n"+str(count) + ".jpg")
            cv2.imshow('image', crop_image)
            cv2.waitKey(10)

foldername = input("Enter the folder name : ")

Image2Face(foldername)
