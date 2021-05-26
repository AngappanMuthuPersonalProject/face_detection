
import cv2
import numpy as np
import os

from numpy import random


def cut_faces(image, faces_coord):
    faces = []

    for (x, y, w, h) in faces_coord:
        w_rm = int(0.2 * w / 2)
        faces.append(image[y:y + h, x + w_rm:x + w - w_rm])
    return faces

def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

def resize(images, size=(50, 50)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

        images_norm.append(image_norm)

    return images_norm

def normalize_faces(frame, faces_coord,gama):
    faces = cut_faces(frame, faces_coord)
    faces = adjust_gamma(faces,gama)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces

def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2)
        cv2.rectangle(image, (x + w_rm, y), (x + w - w_rm, y + h), (150, 150, 0), 8)

def collect_database():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/" + person):
            images.append(cv2.imread("people/" + person + "/" + image, 0))
            labels.append(i)
    return (images, np.array(labels), labels_dic)

def draw_label(image, text, coord, conf, threshold,i):
    if conf < threshold:
        cv2.putText(image, text.capitalize(), (coord[i][0], coord[i][1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
    else:
        cv2.putText(image, "Unknown", (coord[i][0], coord[i][1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)

def adjust_gamma(images, gamma=1.0):
    images_gama = []
    for image in images:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
          for i in np.arange(0, 256)]).astype("uint8")
        images_gama.append(cv2.LUT(image, table))
    return images_gama

def pre_normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces

def reshape_np(X):
    X = np.array(X).reshape(-1,50,50,1)
    return X
