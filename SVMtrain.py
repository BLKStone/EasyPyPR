# -*- coding: utf-8 -*-

import numpy as np
import cv2
from sklearn import svm

from sklearn.externals import joblib

import os
import sys
import time

import PlateLocater
import PlateJudger

# OpenCV 示例中的类
class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

# OpenCV 示例中的类
class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model = cv2.ml.SVM_create()
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
    	print type(self.model.predict(samples))
    	print self.model.predict(samples)
        return self.model.predict(samples)[1][0].ravel()

# 正负样本读取，提取图像特征
def preLoadPicture():
	platePath = 'train/svm/has'
	traindata,label = loadPositiveTrainData(platePath)
	# print traindata.shape
	# print label.shape
	joblib.dump(traindata, 'data/data_has.pkl')
	joblib.dump(label,'data/label_has.pkl')

	platePath = 'train/svm/no'
	traindata,label = loadNegativeTrainData(platePath)
	# print traindata.shape
	# print label.shape
	joblib.dump(traindata, 'data/data_no.pkl')
	joblib.dump(label,'data/label_no.pkl')

# 数据预处理
# 区分训练集和数据集
def dataPreProcess():

	#训练集所占比例
	rate = 0.7

	# 数据预处理
	label_has = joblib.load('data/label_has.pkl')
	data_has = joblib.load('data/data_has.pkl')

	label_no = joblib.load('data/label_no.pkl')
	data_no = joblib.load('data/data_no.pkl')

	slice_index = int(data_has.shape[0]*rate)

	data_has_train = data_has[0:slice_index,:]
	label_has_train = label_has[0:slice_index]
	data_has_test = data_has[slice_index:,:]
	label_has_test = label_has[slice_index:]


	slice_index = int(data_no.shape[0]*rate)

	data_no_train = data_no[0:slice_index,:]
	label_no_train = label_no[0:slice_index]
	data_no_test = data_no[slice_index:,:]
	label_no_test = label_no[slice_index:]

	# 合并
	data_train = np.vstack([data_has_train,data_no_train])
	label_train = np.hstack([label_has_train,label_no_train])

	data_test = np.vstack([data_has_test,data_no_test])
	label_test = np.hstack([label_has_test,label_no_test])

	print '测试集数量',data_test.shape

	joblib.dump(data_train, 'data/data_train.pkl')
	joblib.dump(label_train, 'data/label_train.pkl')

	joblib.dump(data_test, 'data/data_test.pkl')
	joblib.dump(label_test, 'data/label_test.pkl')


# K-folder 交叉验证
def KFolderCrossValidation():

	clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.1, degree=0.1, gamma=1.0,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=True)

	# K折
	K = 5

	# 读取数据
	data_train = joblib.load('data/data_train.pkl')
	label_train = joblib.load('data/label_train.pkl')

	# 乱序化数据
	np.random.seed(0)
	# np.random.seed(int(time.time()))
	indices = np.random.permutation(data_train.shape[0])

	data_train = data_train[indices,:]
	label_train = label_train[indices]

	# K折数据
	data_folds = np.array_split(data_train, K)
	label_folds = np.array_split(label_train, K)


	scores = list()

	for i in range(0,K):
		
		x_train = list(data_folds)
		x_test = x_train.pop(i)
		x_train = np.concatenate(x_train)

		y_train = list(label_folds)
		y_test  = y_train.pop(i)
		y_train = np.concatenate(y_train)

		clf.fit(x_train, y_train)
		evaluateModel(clf)
		joblib.dump(clf, 'model/svm'+str(i)+'.pkl')
		scores.append(clf.score(x_test, y_test))

	print scores

# 模型评估
def evaluateModel(clf):

	# 测试数据读取
	data_test = joblib.load('data/data_test.pkl')
	label_test = joblib.load('data/label_test.pkl')

	# 预测分类
	predict = clf.predict(data_test)

	testset_size = predict.shape[0]

	ptrue_rtrue = 0.
	ptrue_rfalse = 0.
	pfalse_rtrue = 0.
	pfalse_rfalse = 0.

	# 统计结果
	for i in range(0,testset_size):
		if label_test[i] == 1 and predict[i] == 1:
			ptrue_rtrue += 1
		elif label_test[i] == 1 and predict[i] == 0:
			pfalse_rtrue += 1
		elif label_test[i] == 0 and predict[i] == 1:
			ptrue_rfalse += 1
		elif label_test[i] == 0 and predict[i] == 0:
			pfalse_rfalse += 1

	print 'ptrue_rtrue:',int(ptrue_rtrue)
	print 'ptrue_rfalse:',int(ptrue_rfalse)
	print 'pfalse_rtrue:',int(pfalse_rtrue)
	print 'pfalse_rfalse',int(pfalse_rfalse)

	# 计算 准确率与查全率
	precise = ptrue_rtrue / (ptrue_rtrue + ptrue_rfalse)
	recall = ptrue_rtrue / (ptrue_rtrue + pfalse_rtrue)

	print 'precise:',precise
	print 'recall:',recall

	# 计算 Fscore
	Fsocre = 2 * (precise * recall) / (precise + recall)
	print 'Fscore:',Fsocre

	return





# 标签定义 1 has plate
# 标签定义 0 no plate
# sklearn
def trainModel():

	# 数据预处理
	data_train = joblib.load('data/data_train.pkl')
	label_train = joblib.load('data/label_train.pkl')

	print data_train.shape

	clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.1, degree=0.1, gamma=1.0,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=True)

	#clf.set_params(kernel='rbf')

	print clf

	print data_train.shape
	print label_train.shape

	print 'begin training....'
	clf.fit(data_train,label_train)
	print 'finish training....'
	print clf
	joblib.dump(clf, 'model/svm.pkl')
	
	return None


# platePath = 'train/svm/has'
# loadTrainData(platePath) -> traindata,label
def loadPositiveTrainData(platePath):

	traindata = np.ones((1,172))
	label = np.array([1])
	isFirst = True

	for root, dirs, files in os.walk( platePath ):
		print len(files)	
		for file in files:

			path = os.path.join( root, file )
			print 'loading....',path
			imgPlate = cv2.imread(path,cv2.IMREAD_COLOR)
			feature = PlateJudger.getHistogramFeatures(imgPlate)

			if isFirst:
				traindata = feature
				isFirst = False
			else:
				traindata = np.vstack([traindata,feature])
				label = np.hstack([label,1])

	return traindata,label


# demo to load data

# platePath = 'train/svm/no'
# traindata,label = loadNegativeTrainData(platePath)
# joblib.dump(traindata, 'data/traindata_no.pkl')
# joblib.dump(label,'data/label_no.pkl')
def loadNegativeTrainData(platePath):

	traindata = np.ones((1,172))
	label = np.array([0])
	isFirst = True

	for root, dirs, files in os.walk( platePath ):
		print len(files)	
		for file in files:

			path = os.path.join( root, file )
			print 'loading....',path
			imgPlate = cv2.imread(path,cv2.IMREAD_COLOR)
			feature = PlateJudger.getHistogramFeatures(imgPlate)

			if isFirst:
				traindata = feature
				isFirst = False
			else:
				traindata = np.vstack([traindata,feature])
				label = np.hstack([label,0])

	return traindata,label

def CV_trainModel():

	# 数据预处理
	label_has = joblib.load('data/label_has.pkl')
	traindata_has = joblib.load('data/data_has.pkl')

	label_no = joblib.load('data/label_no.pkl')
	traindata_no = joblib.load('data/data_no.pkl')

	traindata = np.vstack([traindata_has,traindata_no])
	labels = np.hstack([label_has,label_no])

	# print traindata.dtype
	# print labels.dtype

	traindata = np.float32(traindata)
	labels = np.int32(labels)

	model = SVM(C=1.0, gamma=1.0)

	model.train(traindata,labels)
	model.save('model/svm.dat')
	#model.load('model/svm.dat')

	return model

def CV_dirPicTest():
	model = CV_trainModel()

	platePath = 'resource/general_test'
	for root, dirs, files in os.walk( platePath ):
		print len(files)	
		for file in files:
			path = os.path.join( root, file )
			print 'loading....',path
			imgPlate = cv2.imread(path,cv2.IMREAD_COLOR)
			PlateLocater.m_debug = False
			Result = PlateLocater.fuzzyLocate(imgPlate)

			for plate in Result:
				
				feature = PlateJudger.getHistogramFeatures(plate)

				print feature.dtype

				feature = np.float32(feature)
				res = model.predict(feature)
				
				a = (res==1)
				print a[0]

				if a[0]:
					cv2.imshow('test',plate)
					cv2.waitKey(0)
					cv2.destroyAllWindows()


def CV_PicTest():
	model = CV_trainModel()

	imgPlate = cv2.imread('plate_judge.jpg',cv2.IMREAD_COLOR)

	PlateLocater.m_debug = False
	Result = PlateLocater.fuzzyLocate(imgPlate)

	for plate in Result:
		cv2.imshow('test',plate)
		feature = PlateJudger.getHistogramFeatures(plate)
		feature = np.float32(feature)
		res = model.predict(feature)
		print res
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def dirPicTest():

	clf = joblib.load('model/svm.pkl')
	print type(clf)
	print clf

	platePath = 'resource/general_test'
	for root, dirs, files in os.walk( platePath ):
		print len(files)	
		for file in files:
			path = os.path.join( root, file )
			print 'loading....',path
			imgPlate = cv2.imread(path,cv2.IMREAD_COLOR)
			PlateLocater.m_debug = False
			Result = PlateLocater.fuzzyLocate(imgPlate)

			for plate in Result:
				
				feature = PlateJudger.getHistogramFeatures(plate)
				res = clf.predict(feature)
				
				flag = (res==1)
				print flag[0]

				if flag[0]:
					cv2.imshow('test',plate)
					cv2.waitKey(0)
					cv2.destroyAllWindows()

				
				


	return None




if __name__ == '__main__':

    KFolderCrossValidation()






# http://scikit-learn.org/stable/modules/svm.html#svc

