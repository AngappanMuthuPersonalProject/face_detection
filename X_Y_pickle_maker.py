import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers,models
import os
from sklearn.model_selection import train_test_split
from random import shuffle
import pickle

person_name = os.listdir('people')
dir_name = 'people'
print(person_name)


data = []
for person in person_name:
    path = os.path.join(dir_name,person)
    class_num = person_name.index(person)
    print(person,class_num)
    for img in os.listdir(path):
        try:
            photo = cv2.imread(os.path.join(path,img),-1)
            new_photo = cv2.resize(photo,(50,50))
            data.append([new_photo,class_num])
        except Exception as e:
            pass


# print(len(data))
# plt.imshow(data[0][0],cmap='gray')
# plt.show()

shuffle(data)

for sample in data[:10]:
    print(sample[1])

X = []
Y = []

for features, labels in data:
    X.append(features)
    Y.append(labels)

X = np.array(X).reshape(-1,50,50,1)

pickle_out = open('X.pickle','wb')
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open('Y.pickle','wb')
pickle.dump(Y,pickle_out)
pickle_out.close()

X = pickle.load(open('X.pickle','rb'))
Y = pickle.load(open('Y.pickle','rb'))



