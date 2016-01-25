# -*- coding: utf-8 -*-

import numpy as np
import cv2

global m_debug
m_debug = True

colors = ['blue','yellow','white','unknown']

# 根据一幅图像与颜色模板获取对应的二值图
# 输入RGB图像, 颜色模板（蓝色、黄色）
# 返回灰度图（只有0和255两个值，255代表匹配，0代表不匹配） -> 匹配矩阵

def colorMatch(srcMat, color, adaptive_minsv):

    # S和V的最小值由adaptive_minsv这个bool值判断
    # 如果为true，则最小值取决于H值，按比例衰减
    # 如果为false，则不再自适应，使用固定的最小值minabs_sv
    # 默认为false
    pass


# 判断车牌颜色
def plateColorJudge(srcMat, color, adaptive_minsv):
    # 判断阈值
    thresh = 0.45
    match_mat = colorMatch(srcMat, color, adaptive_minsv);
    percent = cv2.countNonZero(match_mat) / float(srcMat.shape[0] * srcMat.shape[1]);

    if percent > thresh:
        return true;
    else:
        return false;


# 获取车牌类型
# inMat 车牌输入 
# adaptive_minsv 颜色匹配的方式
#                目前有两种方式
#                自适应方式 和 minsv方式
def getPlateType(inMat, adaptive_minsv):
    max_percent = 0
    max_color = "unknown"
    blue_percent = 0
    yellow_percent = 0
    white_percent = 0

    blue_flag, blue_percent = plateColorJudge(inMat,"blue",adaptive_minsv)
    yellow_flag, yellow_percent = plateColorJudge(inMat,"yellow",adaptive_minsv)
    white_flag, white_percent = plateColorJudge(inMat,"white",adaptive_minsv)


    if blue_flag is True:
        # print "blue" 
        return "blue"
    elif yellow_flag is True:
        # print "yellow"
        return "yellow"
    elif white_flag is True:
        # print "white"
        return "white"
    else:
        # print "other"
        # 如果任意一者都不大于阈值，则取百分比值最大者
        if blue_percent > yellow_percent:
            max_percent = blue_percent
            max_color = 'blue'
        else:
            max_percent = yellow_percent
            max_color = 'yellow'

        if white_percent > max_percent:
            max_percent = white_percent
            max_color = 'white'

        return max_color


# 字符尺寸验证
def verifyCharSizes(inMat):
    pass


# 字符预处理
def preprocessChar(inMat):
    pass


# 字符分割与排序
def charsSegment(inMat):
    w = inMat.shape[1]
    h = inMat.shape[0]
    print w,h
    chars = []
    return chars

# 根据特殊车牌来构造猜测中文字符的位置和大小

def getChineseRect():
    pass

