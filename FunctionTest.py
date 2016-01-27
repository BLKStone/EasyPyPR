# -*- coding: utf-8 -*-

import numpy as np
import cv2
from sklearn import svm
import os
import PlateLocater
import PlateJudger
import CharsSegmenter

# 测试车牌定位 和 车牌判别
def testPlateLocater():
    PlateLocater.m_debug = False
    rootdir = "resource/easy_test"
    for parent,dirnames,filenames in os.walk(rootdir):
        
        index_file = 0
        for filename in filenames:
                cv2.destroyAllWindows()
                index_file += 1

                imgPlate = cv2.imread(os.path.join(parent,filename),cv2.IMREAD_COLOR)
                print '文件名:',filename.split('.')[0]

                cv2.imshow("原图",imgPlate)
                Result = PlateLocater.fuzzyLocate(imgPlate)
                print '候选车牌数量：',len(Result)

                index_loc = 0
                for img in Result:
                    index_loc += 1
                    cv2.imshow("候选车牌"+str(index_loc),img)
                cv2.waitKey(0)

                resultVec = PlateJudger.platesJudge(Result)
                print 'SVM筛选后的车牌数量：',len(resultVec)

                index_loc = 0
                for img in resultVec:
                    index_loc += 1
                    cv2.imshow("SVM-"+str(index_loc),img)
                cv2.waitKey(0)



                if index_file >20:
                     break

# 测试字符分割
# chars_segment.cpp 79L
def testCharsSegment():
    print "test_chars_segment"
    imgPlate = cv2.imread("resources/image/chars_segment.jpg",cv2.IMREAD_COLOR)
    cv2.imshow("test",imgPlate)

    # print cv2.countNonZero(imgPlate[:,:,0])
    # print imgPlate.shape[0],imgPlate.shape[1],imgPlate.shape[1]*imgPlate.shape[0]
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    segmented = CharsSegmenter.charsSegment(imgPlate)

    # index_char = 0
    # for char in segmented:
    #     index_char += 1
    #     cv2.imshow("chars_segment"+str(index_char), char)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    # testPlateLocater()
    testCharsSegment()