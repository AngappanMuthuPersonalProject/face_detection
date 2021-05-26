
from face_utils import *
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np


def adjust_gamma(image, gamma=1.0):
    faces = []
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

    for f in range(0,6):
        faces.append(cv2.LUT(image, table))
    return faces

def radius2stdev(radius):
    """
    Return std deviation corresponding to a given radius.
    I got this from: https://gitlab.gnome.org/GNOME/gimp/blob/master/plug-ins/common/edge-dog.c
    """
    stdev  = math.sqrt (-(radius * radius) / (2 * math.log (1.0 / 255.0)))
    return stdev

def DoG_img(img):
    # Load image, make float and scale to range 0..1
    im = img.astype(np.float)
    im = im/255.0

    stdev1  = radius2stdev(22.0)
    stdev2  = radius2stdev(5.0)

    g1 = cv2.GaussianBlur(im,(0,0),stdev1,stdev1)
    g2 = cv2.GaussianBlur(im,(0,0),stdev2,stdev2)
    result = g1-g2
    result = (result * 255).astype(np.uint8)
    resultn = cv2.normalize(result,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)

    return result,resultn

x = 'people/angappan/50.jpg'  #location of the image
original = cv2.imread(x, 1)
cv2.imshow('original',original)

gamma = 3.0                         # change the value here to get different result
adjusted = adjust_gamma(original, gamma=gamma)
cv2.imshow("gammam image 1", adjusted[0])
print(type(gamma))


cv2.waitKey(0)
cv2.destroyAllWindows()
