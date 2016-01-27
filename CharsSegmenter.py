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
    
    max_sv = 255
    minref_sv = 64
    minabs_sv = 95

    # blue的H范围
    min_blue = 100
    max_blue = 140

    # yellow的H范围
    min_yellow = 15
    max_yellow = 40

    # white的H范围
    min_white = 0
    max_white = 30

    # 转到HSV空间进行处理，颜色搜索主要使用的是H分量进行蓝色与黄色的匹配工作
    srcHSV = cv2.cvtColor(srcMat, cv2.COLOR_BGR2HSV)

    h,s,v = cv2.split(srcHSV)
    v = cv2.equalizeHist(v)
    srcHSV = cv2.merge((h,s,v))

    # 匹配模板基色,切换以查找想要的基色
    min_h = 0
    max_h = 0

    switch={
    'blue': (min_blue, max_blue),
    'yellow': (min_yellow, max_yellow),
    'white': (min_white, max_white)
    }
    min_h, max_h = switch.get(color,'unknown')

    diff_h = float((max_h - min_h) / 2)
    avg_h = min_h + diff_h

    channels = srcHSV.shape[2]
    rows = srcHSV.shape[0]
    cols = srcHSV.shape[1]

    s_all = 0
    v_all = 0
    count = 0
    # 遍历图像
    for i in range(0,rows):
        for j in range(0,cols):
            H = srcHSV[i,j,0]
            S = srcHSV[i,j,1]
            V = srcHSV[i,j,2]
            s_all += S
            v_all += V
            count += 1

            colorMatched = False

            if H > min_h and H < max_h: 
                # H与中心的绝对差距
                Hdiff = abs(H - avg_h)
                # H与中心的相对差距
                Hdiff_p = Hdiff / float(diff_h)

                # S和V的最小值由adaptive_minsv这个bool值判断
                # 如果为true，则最小值取决于H值，按比例衰减
                # 如果为false，则不再自适应，使用固定的最小值minabs_sv
                min_sv = 0
                if adaptive_minsv != 1:
                    min_sv = minabs_sv
                else:
                    min_sv = minref_sv - minref_sv / 2 * (1 - Hdiff_p)

                if ((S > min_sv and S < max_sv) and (V > min_sv and V < max_sv)):
                    colorMatched = True

            if colorMatched:
                srcHSV[i,j,0] = 255
            else:
                srcHSV[i,j,0] = 0
    
    if m_debug:
        print "avg_s:", s_all / count
        print "avg_v:", v_all / count



    return srcHSV[:,:,0]




# 判断车牌颜色
def plateColorJudge(srcMat, color, adaptive_minsv):
    # 判断阈值
    thresh = 0.45
    match_mat = colorMatch(srcMat, color, adaptive_minsv);
    percent = cv2.countNonZero(match_mat) / float(srcMat.shape[0] * srcMat.shape[1]);

    if percent > thresh:
        return (True, percent);
    else:
        return (False, percent);


# 获取车牌类型
# inMat 车牌输入 
# adaptive_minsv 颜色匹配的方式
#                目前有两种方式
#                自适应方式 和 minsv方式
# 当 adaptive_minsv 值为1或其他 使用自适应方式
# 当 adaptive_minsv 值为0 使用minsv方式
# 输出情况  'blue' | 'yellow' | 'white'
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

    # 取原车牌的一部分
    tmpMat = inMat[int(0.1*h):int(0.9*h),int(0.1*w):int(0.9*w),:]

    # 判断车牌颜色以此确认二值化的方法
    plateType = getPlateType(tmpMat, True);
    print plateType



    return chars

# 根据特殊车牌来构造猜测中文字符的位置和大小

def getChineseRect():
    pass

