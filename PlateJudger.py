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

# 获取垂直和水平的直方图图值
def getHistogramFeatures(img):
	# 灰度化
	imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# 大津算法 二值化
	retval,imgThres = cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# 获取 水平投影，垂直投影 的特征
	features = getProjectFeatures(imgThres)

	return features


# 按行或按列 统计比 阈值 threshold大 的像素数量
def countOfBigValue(oneLine,project_type,threshold):

	count = 0

	if project_type == 0:
		# vertical
		for line in oneLine:
			for pixel in line:
				if pixel > threshold:
					count = count + 1
	else:
		# horizontal
		for pixel in oneLine:
			if pixel > threshold:
				count = count + 1
	
	return count


# 获取垂直和水平方向直方图
def ProjectedHistogram(img,project_type):

	if project_type == 0:
		# vertical
		size = img.shape[1]
	else:
		# horizontal
		size = img.shape[0]

	mhist = np.zeros((1,size))
	threshold = 20

	for i in range(0,size):
		if project_type == 0:
			# vertical
			oneLine = img[:,i]

			mhist[0,i] = countOfBigValue(oneLine,project_type,threshold)

		else:
			# horizontal
			oneLine = img[i,:]

			mhist[0,i] = countOfBigValue(oneLine,project_type,threshold)


	# 归一化直方图
	# Normalize histogram

	# 先获取最大值和最小值
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mhist)

	# 用mhist直方图中的最大值，归一化直方图

	mhist = np.float64(mhist)
	mhist = mhist / max_val

	return mhist 

# inMat 2维矩阵 只有 0与255
def getProjectFeatures(inMat):

	VERTICAL = 0
	HORIZONTAL = 1 

	vhist = ProjectedHistogram(inMat, VERTICAL)
	hhist = ProjectedHistogram(inMat, HORIZONTAL)

	numCols = vhist.shape[1] + hhist.shape[1]

	outMat = np.zeros((1, numCols));
	index = 0
	for i in range(0,vhist.shape[1]):
		outMat[0,index] = vhist[0,i]
		index = index + 1
	for i in range(0,vhist.shape[1]):
		outMat[0,index] = hhist[0,i]
		index = index + 1

	return outMat

# 对多幅图像进行SVM判断
def platesJudge(inMats):

	resultVec = []

	for inMat in inMats:
		response = -1
		response = plateJudge(inMat)

		if (response == 1):
			resultVec.append(inMat)

	return resultVec


# 了解features reshape以后的形式
# 1*(rows+cols)的矩阵 0~1之间的float

# 对单幅图像进行SVM判断
def plateJudge(inMat):

	svm = cv2.SVM()
	svm.load('model/svm.xml')

	m_getFeatures = getProjectFeatures

	# 获取特征
	features = m_getFeatures(inMat)
	# 使用 svm 预测
	response = svm.predict(features)
	response = int(response)
	print 'debug',response

	return response




# 
imgPlate = cv2.imread('plate_judge.jpg',cv2.IMREAD_COLOR)

PlateLocater.m_debug = False
Result = PlateLocater.fuzzyLocate(imgPlate)

print cv2.__version__
platesJudge(Result)






# imgGray = cv2.cvtColor(imgPlate,cv2.COLOR_BGR2GRAY)
# cv2.imshow('src',imgGray)
# imgEqulhist = cv2.equalizeHist(imgGray)
# cv2.imshow('equal',imgEqulhist)
cv2.waitKey(0)
cv2.destroyAllWindows()



# svm 参考
# http://answers.opencv.org/question/5713/save-svm-in-python/
#

# 遇到的问题
# http://answers.opencv.org/question/55152/unable-to-find-knearest-and-svm-functions-in-cv2/