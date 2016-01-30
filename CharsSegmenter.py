# -*- coding: utf-8 -*-

import numpy as np
import cv2

global m_debug
m_debug = True
m_defaut_plate_width = 136
colors = ['blue','yellow','white','unknown']



# 获取特殊字符的索引
# 特殊字符指的是 苏A7003X的"A"
# 输入是 车牌字符的坐标尺寸信息的list (x,y,width,height)
# 输出是 特殊字符的索引
def getSpecificRect(rects):
    max_height = 0
    max_width = 0
    
    for i in range(0,len(rects)):
        if rects[i][3] > max_height:
            max_height = rects[i][3]
        if rects[i][2] > max_width:
            max_width = rects[i][2]

    specific_index = 0
    for i in range(0,len(rects)):
        midx = rects[i][0] + rects[i][2]

        # 如果一个字符有一定的大小，并且在整个车牌的1/7到2/7之间，则是我们要找的特殊字符
        # 当前字符和下个字符的距离在一定的范围内
        if ((rects[i][2] > max_width * 0.8 or rects[i][3] > max_height * 0.8) and
            (midx <= int(m_defaut_plate_width / 7) * 2 and
             midx >= int(m_defaut_plate_width / 7) * 1)):
            specific_index = i

    return specific_index


# 根据特殊车牌来构造猜测中文字符的位置和大小
# 输入 特殊字符的坐标尺寸信息 (x,y,width,height)
# 输出 汉字字符的坐标尺寸信息 (x,y,width,height)
def getChineseRect(rect):
    height = rect[3]
    new_width = rect[2] * 1.15
    x = rect[0]
    y = rect[1]

    new_x = x - int(new_width * 1.15)

    if new_x < 0:
        new_x = 0

    return (new_x, y, int(new_width), height)




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
    
    # if m_debug:
    #     print "avg_s:", s_all / count
    #     print "avg_v:", v_all / count



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
def verifyCharSizes(mr):

    # mr ((5.5, 24.5), (1.0, 1.0), -90.0)
    # Char sizes 45x90
    aspect = 45.0 / 90.0
    charAspect = mr.shape[1] / float(mr.shape[0])
    error = 0.7
    minHeight = 10.0
    maxHeight = 35.0

    # We have a different aspect ratio for number 1, and it can be ~0.2
    minAspect = 0.05
    maxAspect = aspect + aspect * error

    # area of pixels
    area = cv2.countNonZero(mr)
    #  bb area
    bbArea = mr.shape[0] * mr.shape[1]

    percPixels = area / float(bbArea)

    if (percPixels <= 1 and 
        charAspect > minAspect and 
        charAspect < maxAspect and 
        mr.shape[0] >= minHeight and 
        mr.shape[1] < maxHeight):
        # 满足条件
        return True
    else:
        return False


# 字符预处理
def preprocessChar(inMat):
    pass

# 处理铆钉
# 这里会有一些和原项目的区别
# 我希望把 车牌判断 和 去除铆钉 的功能分离开
# 返回 (是否是车牌，去铆钉的结果)
def removeRivet(inMat):
    # 去除车牌上方的钮钉
    # 计算每行元素的阶跃数，如果小于X认为是柳丁，将此行全部填0（涂黑）
    # X的推荐值为 7，可根据实际调整
    x = 7
    whiteCount = 0
    jump = []

    # 遍历图像，计算每行阶跃数,存在jump中
    for i in range(0, inMat.shape[0]):
        jumpCount = 0
        for j in range(0, inMat.shape[1]-1):
            if inMat[i,j] != inMat[i,j+1]:
                jumpCount += 1
            if inMat[i,j] == 255:
                whiteCount += 1
        jump.append(jumpCount)

    iCount = 0
    for i in range(0, inMat.shape[0]):
        if jump[1] >= 16 and jump[i] <= 45:
            # 车牌字符应满足一定的跳变条件
            iCount += 1

    # # 不满足车牌的条件 每行跳变计数
    # if (iCount * 1.0 / inMat.shape[0]) <= 0.40:
    #     return (False, inMat)

    # # 不满足车牌的条件 白色点计数
    # if ( whiteCount * 1.0 / (inMat.shape[0] * inMat.shape[1]) < 0.15 or
    #    whiteCount * 1.0 / (inMat.shape[0] * inMat.shape[1]) > 0.50 ):
    #     return (False, inMat)

    # 去除铆钉 填黑色
    for i in range(0, inMat.shape[0]):
        if jump[i] < x:
            inMat[i,:] = 0

    return (True, inMat)



# 字符分割与排序
def charsSegment(inMat):
    w = inMat.shape[1]
    h = inMat.shape[0]

    # 输出的车牌字符
    chars = []

    # 取原车牌的一部分
    tmpMat = inMat[int(0.1*h):int(0.9*h),int(0.1*w):int(0.9*w),:]

    # 判断车牌颜色以此确认二值化的方法
    plateType = getPlateType(tmpMat, True);
    
    # if m_debug:
    #     print "车牌类型:",plateType

    plate_gray = cv2.cvtColor(inMat, cv2.COLOR_BGR2GRAY)

    # 二值化

    if plateType == 'blue':
        # 蓝色车牌
        plate_threshold = plate_gray.copy()
        w = plate_gray.shape[1]
        h = plate_gray.shape[0]
        roi = plate_gray[int(0.1*h):int(0.9*h),int(0.1*w):int(0.9*w)]
        threadhold_value, roi_threshold = cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        retval, plate_threshold = cv2.threshold(plate_gray,threadhold_value,255,cv2.THRESH_BINARY)
    elif plateType == 'yellow':
        # 黄色车牌 反二值化
        plate_threshold = plate_gray.copy()
        w = plate_gray.shape[1]
        h = plate_gray.shape[0]
        roi = plate_gray[int(0.1*h):int(0.9*h),int(0.1*w):int(0.9*w)]
        threadhold_value, roi_threshold = cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        retval, plate_threshold = cv2.threshold(plate_gray,threadhold_value,255,cv2.THRESH_BINARY_INV)
    elif plateType == 'white':
        # 白色车牌 反二值化
        plate_threshold = plate_gray.copy()
        w = plate_gray.shape[1]
        h = plate_gray.shape[0]
        roi = plate_gray[int(0.1*h):int(0.9*h),int(0.1*w):int(0.9*w)]
        threadhold_value, roi_threshold = cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        retval, plate_threshold = cv2.threshold(plate_gray,threadhold_value,255,cv2.THRESH_BINARY_INV)
    else: 
        # 当车牌类型未知时
        plate_threshold = plate_gray.copy()
        w = plate_gray.shape[1]
        h = plate_gray.shape[0]
        roi = plate_gray[int(0.1*h):int(0.9*h),int(0.1*w):int(0.9*w)]
        threadhold_value, roi_threshold = cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        retval, plate_threshold = cv2.threshold(plate_gray,threadhold_value,255,cv2.THRESH_BINARY)

    if m_debug:
        # print plate_threshold.shape
        cv2.imwrite("debug/plate_threshold.png",plate_threshold)

    # 去除车牌上方的柳钉以及下方的横线等干扰
    # 并且也判断了是否是车牌
    # 并且在此对字符的跳变次数以及字符颜色所占的比重做了是否是车牌的判别条件
    # 如果不是车牌，返回ErrorCode=0x02
    flag, plate_threshold = removeRivet(plate_threshold)
    # print 'removerivet_flag',flag

    if flag is not True:
        # 如果判断不是车牌
        # 直接返回空的字符组
        return chars

    plate_contours = plate_threshold.copy()
    plate_contours, contours, hierarchy = cv2.findContours(plate_contours,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    #print 'plate',plate_threshold[1:20,1:20]

    # 遍历获取的轮廓
    rects = []
    for i in range(0,len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        mr = plate_threshold[y:y+h,x:x+w]

        if not verifyCharSizes(mr):
            pass
        else:

            rects.append((x,y,w,h))

    # 对符合尺寸的图块按照从左到右进行排序
    rects = sorted(rects,key = lambda x:x[0]) 

    # for i in range(0,len(rects)):
    #     print rects[i][0], rects[i][1]

    # 没什么用的测试代码

    # print len(rects)
    # for rect in rects:
    #     x,y,w,h = rect
    #     mr = plate_threshold[y:y+h,x:x+w]
    #     char_rect.append(mr)
    #     if m_debug:
    #         cv2.rectangle(inMat,(x,y),(x+w,y+h),(255,255,0),1)

    # if m_debug:
    #     plate_contours_chosen = cv2.drawContours(inMat, box_rects, -1, (255,0,0), 1)
    #     cv2.imshow('contours chosen',plate_contours_chosen)
    #     cv2.imwrite('debug/plate_contours_chosen.png',plate_contours_chosen)

    # 汉字处理
    # 获得特殊字符对应的Rect,如苏A的"A"

    specific_index = getSpecificRect(rects)
    # print "特殊字符位置:", specific_index

    # 根据特定Rect向左反推出中文字符
    # 这样做的主要原因是根据findContours方法很难捕捉到中文字符的准确Rect，因此仅能
    # 退过特定算法来指定

    chinese_rect = getChineseRect(rects[specific_index])
    # print chinese_rect

    # 新建一个全新的排序Rect
    # 将中文字符Rect第一个加进来，因为它肯定是最左边的
    # 其余的Rect只按照顺序去6个，车牌只可能是7个字符！这样可以避免阴影导致的“1”字符
    new_sorted_rects = []
    new_sorted_rects.append(chinese_rect)
    for i in range(specific_index,specific_index+6):
        new_sorted_rects.append(rects[i])

    # print len(new_sorted_rects)

    for rect in new_sorted_rects:
        x,y,w,h = rect
        mr = plate_threshold[y:y+h,x:x+w]
        chars.append(mr)
        if m_debug:
            cv2.rectangle(inMat,(x,y),(x+w,y+h),(0,255,0),1)

    if m_debug:
        cv2.imwrite('debug/plate_contours_chosen.png',inMat)

    # print "chars:",len(chars)

    return chars


