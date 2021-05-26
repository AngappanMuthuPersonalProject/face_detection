
from VideoCamera import VideoCamera
from FaceDetector import FaceDetector
from face_utils import *

images, labels, labels_dic = collect_database()

print(type(images))
print(type(labels))
print(type(labels_dic))

# print(images)
# print(labels)
# print(labels_dic)

rec_eig = cv2.face.EigenFaceRecognizer_create()
rec_eig.train(images,labels)

# need atleast two peoples
rec_fisher = cv2.face.FisherFaceRecognizer_create()
rec_fisher.train(images,labels)

rec_lbph = cv2.face.LBPHFaceRecognizer_create()
rec_lbph.train(images,labels)

print("Models Trained Successfully")

webcam = VideoCamera('rtsp://admin:123456@192.168.1.22/H264?ch=1&subtype=0')#'sample_videos/angappan1.3gp')#'rtsp://admin:123456@192.168.1.22/H264?ch=1&subtype=0')
detector = FaceDetector('Cascades/haarcascade_frontalface_default.xml')

cv2.namedWindow("prediction", cv2.WINDOW_AUTOSIZE)
while (True):
    frame = webcam.get_frame()
    faces_coord = detector.detect(frame)
    if len(faces_coord):
        faces = pre_normalize_faces(frame, faces_coord)
        for i,face in enumerate(faces):
            collector = cv2.face.StandardCollector_create()
            pred, conf = rec_lbph.predict(face)
            threshold = 180
            print("Predicton: " + labels_dic[pred].capitalize() + "\nConfidence:  " + str(round(conf)))
            draw_label(frame,labels_dic[pred],faces_coord,conf,threshold,i)
        cv2.imshow("face", faces[0])
        draw_rectangle(frame, faces_coord)
    cv2.imshow("prediction", frame)

    print(cv2.imshow("prediction", frame))
    if cv2.waitKey(10) & 0xFF == 27:
        break
del webcam
cv2.destroyAllWindows()



