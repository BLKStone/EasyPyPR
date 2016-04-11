# -*- coding: utf-8 -*-

import numpy as np
import cv2

import PlateJudger
import ANNtrain


global m_debug
m_debug = True

kPredictSize = 10


#
# 输入 inMat 截取的单个字符块的二值化矩阵
#     sizeData 低分辨率字符的尺寸
def features(inMat, sizeData):
    # 抠取中间区域
    center_rect = getCenterRect(inMat)
    x,y,w,h = center_rect
    roi = inMat[y:y+h,x:x+w]

    # low data feature
    low_data = cv2.resize(roi,(sizeData,sizeData))


    VERTICAL = 0
    HORIZONTAL = 1 
    # Histogram feature
    vhist = PlateJudger.ProjectedHistogram(low_data, VERTICAL)
    hhist = PlateJudger.ProjectedHistogram(low_data, HORIZONTAL)

    # print vhist.shape
    # print hhist.shape
    # print low_data

    numCols = vhist.shape[1] + hhist.shape[1] + low_data.shape[0] * low_data.shape[1]
    outMat = np.zeros((1, numCols))


    # feature,ANN的样本特征为水平、垂直直方图和低分辨率图像所组成的矢量
    index = 0
    for i in range(0,vhist.shape[1]):
        outMat[0,index] = vhist[0,i]
        index += 1

    # print outMat

    for i in range(0,hhist.shape[1]):
        outMat[0,index] = hhist[0,i]
        index += 1

    for i in range(0,low_data.shape[0]):
        for j in range(0,low_data.shape[0]):
            outMat[0,index] = low_data[i,j]
            index += 1

    return outMat


def getCenterRect(inMat):
    # 上下
    top = 0
    bottom = inMat.shape[0] - 1
    # 遍历统计这一行或一列中，非零元素的位置
    find = False
    for i in range(0,inMat.shape[0]):
        for j in range(0,inMat.shape[1]):
            if inMat[i,j]>20:
                top = i
                find = True
                break
        if find is True:
            break

    if m_debug: 
        print "top",top
        print "original bottom",bottom

    find = False
    for i in range(inMat.shape[0]-1,-1,-1):
        for j in range(inMat.shape[1]-1,-1,-1):
            if inMat[i,j]>20:
                bottom = i
                find = True
                break
        if find is True:
            break

    if m_debug:
        print "bottom",bottom


    # 左右
    left = 0
    right = inMat.shape[1] - 1

    find = False
    for i in range(0,inMat.shape[1]):
        for j in range(0,inMat.shape[0]):
            if inMat[j,i]>20:
                left = i
                find = True
                break
        if find is True:
            break

    if m_debug:
        print "left",left
        print "original right",right

    find = False
    for i in range(inMat.shape[1]-1,-1,-1):
        for j in range(inMat.shape[0]-1,-1,-1):
            if inMat[j,i]>20:
                right = i
                find = True
                break
        if find is True:
            break
    if m_debug:
        print "right",right

    x = left
    y = top
    width = right - left + 1
    height = bottom - top + 1

    return (x,y,width,height)



def initModel():
    global chinese_model 
    global digit_letter_model 

    i = 122
    data_path = '../goodmodel/ann_digit_letter_train_data_'+str(i)
    label_path = '../goodmodel/ann_digit_letter_train_label_'+str(i)

    chinese_model = ANNtrain.train_chinese_model()
    digit_letter_model = ANNtrain.train_digit_letter_model(data_path, label_path)

def identifyChinese(inMat):
    
    # the data type must be consistent
    feature = inMat.astype(np.float32)
    output = chinese_model.predict(feature)
    return ANNtrain.province_mapping.get(output[0],('x','x'))[1]
    

def identifyDigitLetter(inMat):

    # the data type must be consistent
    feature = inMat.astype(np.float32)
    output = digit_letter_model.predict(feature)
    print output
    return ANNtrain.digit_letter_mapping.get(output[0],('x','x'))[1]


