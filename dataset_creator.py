from VideoCamera import VideoCamera
from FaceDetector import FaceDetector
from face_utils import *
from time import sleep as delay
import cv2

webcam = VideoCamera('sample_videos/prathisha.mp4')#'sample_videos/mariyappan.3gp')#'rtsp://admin:123456@192.168.1.22/H264?ch=1&subtype=0')
detector = FaceDetector('Cascades/haarcascade_frontalface_default.xml')

folder = "people/" + input('Person: ').lower() # input name
counter  = 0
gama = 0.1
if not os.path.exists(folder):
    os.makedirs(folder)
    while True:
        frame = webcam.get_frame(True)
        faces_coord = detector.detect(frame)
        if len(faces_coord):
            counter += 1
            gama += 0.006
            print(gama)
            faces = pre_normalize_faces(frame, faces_coord)
            cv2.imwrite(folder+"/"+str(counter)+".jpg",faces[0])
            cv2.imshow('frame', faces[0])
            #delay(1)
            print(folder+"/"+str(counter)+".jpg exported")
        else:
            print("Face not found")
        if counter > 101:
            break
        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            break
    del webcam
    cv2.destroyAllWindows()
else:
    print("person already exist ....")



