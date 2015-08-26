# -*- coding: utf-8 -*-

import numpy as np
import cv2
import PlateLocater

global m_debug
m_debug = False

# 直方图均衡
def histeq(inMat):
	rows,cols,channels = inMat.shape

	if channels == 3:
		hsv = cv2.cvtColor(res,cv2.COLOR_BGR2HSV)
		h,s,v = cv2.split(hsv)
		v_equalHist = cv2.equalizeHist(v)
		hsv = cv2.merge((h,s,v_equalHist))
		outMat = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	if channels == 1:
		outMat = cv2.equalizeHist(inMat)

	return outMat


imgPlate = cv2.imread('plate_judge.jpg',cv2.IMREAD_COLOR)

PlateLocater.m_debug = False
Result = PlateLocater.fuzzyLocate(imgPlate)

res = Result[0]






# imgGray = cv2.cvtColor(imgPlate,cv2.COLOR_BGR2GRAY)
# cv2.imshow('src',imgGray)
# imgEqulhist = cv2.equalizeHist(imgGray)
# cv2.imshow('equal',imgEqulhist)
cv2.waitKey(0)
cv2.destroyAllWindows()

