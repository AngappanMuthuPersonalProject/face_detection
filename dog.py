#!/usr/bin/env python3

import numpy as np
import math
import cv2

def radius2stdev(radius):
    """
    Return std deviation corresponding to a given radius.
    I got this from: https://gitlab.gnome.org/GNOME/gimp/blob/master/plug-ins/common/edge-dog.c
    """
    stdev  = math.sqrt (-(radius * radius) / (2 * math.log (1.0 / 255.0)))
    return stdev

def DoG_img(img):
    # Load image, make float and scale to range 0..1
    #im = cv2.imread(img,cv2.IMREAD_COLOR).astype(np.float)
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


r,r1 = DoG_img("people/angappan/13.jpg")
cv2.imshow("result", r)

# Normalize and save normalised too
resultn = cv2.normalize(r1,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
cv2.imshow("result-n", resultn)

cv2.waitKey(0)
cv2.destroyAllWindows()
