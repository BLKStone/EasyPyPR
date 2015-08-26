# -*- coding: utf-8 -*-

import numpy as np
import cv2

im = cv2.imread('test.png')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

#imgray = np.float64(imgray)
ret,thresh =  cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#print im[0:10,0:10]
#print imgray[0:10,0:10]

#Python: cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) → contours, hierarchy
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print len(contours)
print contours[2]
image = cv2.drawContours(im, contours, 2, (0,255,0), 3)


cv2.imshow('source image',im)
cv2.imshow('gray image',imgray)
cv2.imshow('thres',thresh)
cv2.imshow('img',image)

cv2.waitKey(0)
cv2.destroyAllWindows()