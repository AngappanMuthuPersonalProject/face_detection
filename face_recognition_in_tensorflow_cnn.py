import cv2
import os
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPool2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from VideoCamera import VideoCamera
from FaceDetector import FaceDetector
from face_utils import *


person_name = os.listdir('people')
person_name.append('Unknown')

# NAME = 'Cat-vs-dog-64x2{}'.format(int(time.time()))

# tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

X = pickle.load(open('X.pickle','rb'))
Y = pickle.load(open('Y.pickle','rb'))
# print(X[0][0][0])


X = X / 255.0
Y = np.array(Y)

model = Sequential()

model.add(Conv2D(64,(3,3),input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(len(person_name)+1))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.fit(X,Y,batch_size=32,epochs = 10,validation_split=0.1)
model.save('faceModel')
# print(Y)

# model = Sequential()
# model.add(Flatten())
# model.add(Dense(128,activation=tf.nn.relu))
# model.add(Dense(128,activation=tf.nn.relu))
# model.add(Dense(10,activation=tf.nn.softmax))
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(X,Y,epochs=100,validation_split=0.1)

x_faces = []
x_lables = []
model = tf.
webcam = VideoCamera('sample_videos/surya.mp4')#'sample_videos/mariyappan.3gp')#'rtsp://admin:123456@192.168.1.22/H264?ch=1&subtype=0')#'sample_videos/angappan1.3gp')#'rtsp://admin:123456@192.168.1.22/H264?ch=1&subtype=0')
detector = FaceDetector('Cascades/haarcascade_frontalface_default.xml')
try:
    while True:
        count = 0
        frame = webcam.get_frame()
        faces_coord = detector.detect(frame)
        if len(faces_coord):
            faces = pre_normalize_faces(frame, faces_coord)
            faces = reshape_np(faces)
#             print(faces.shape)

            predictions = model.predict([faces])
            pred = np.argmax(predictions)
            # x_faces = faces[0]
            # x_lables.append(pred)
            # if count > 50:
            #     x_faces.clear()
            #     x_lables.clear()

            # test_loss, test_acc = model.evaluate(x_faces,  x_lables, verbose=2)
            # print("Test loss: ",test_loss,"Test acc: ",test_acc)
            print(predictions)
            print(person_name[pred])
            print(pred)

            # break
            cv2.imshow('face',faces[0])

            for i,face in enumerate(faces):

                # pred = round(predictions[round(faces[i])])
                cv2.putText(frame, person_name[pred].capitalize(), (faces_coord[i][0], faces_coord[i][1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)

            # plt.imshow(frame,cmap='gray')
            # plt.show()
            # clear_output(wait=True)
            draw_rectangle(frame, faces_coord)


        cv2.imshow('frame',frame)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    del webcam
    cv2.destroyAllWindows()
except KeyboardInterrupt as e:
    del webcam
