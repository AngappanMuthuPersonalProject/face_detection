import cv2
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