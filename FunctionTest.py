# -*- coding: utf-8 -*-

import numpy as np
import cv2
from sklearn import svm
import os
import PlateLocater
import PlateJudger
import CharsSegmenter
import CharsIndentifier

kPredictSize = 10

# 测试车牌定位 和 车牌判别
def test_plate_locate():
    PlateLocater.m_debug = False
    rootdir = "resources/easy_test"
    print 'test_plate_locate'

    for parent,dirnames,filenames in os.walk(rootdir):        
        index_file = 0
        print filenames        
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
def test_chars_segment():
    print "test_chars_segment"
    img_plate = cv2.imread("resources/image/chars_segment.jpg",cv2.IMREAD_COLOR)
    cv2.imshow("test",img_plate)

    # print cv2.countNonZero(img_plate[:,:,0])
    # print img_plate.shape[0],img_plate.shape[1],img_plate.shape[1]*img_plate.shape[0]
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    segmented = CharsSegmenter.charsSegment(img_plate)

    index_char = 0
    for char in segmented:
        index_char += 1
        cv2.imshow("chars_segment"+str(index_char), char)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 测试字符识别
# chars.hpp 30L
def test_chars_identify():
    print "test_chars_identify"
    img_plate = cv2.imread("resources/image/chars_identify.jpg",cv2.IMREAD_COLOR)

    # 车牌分割
    segmented = CharsSegmenter.charsSegment(img_plate)
    licence = ''
    for char in segmented:
        print 'char:'
        print char
        # licence += CharsIndentifier.identify(char)
        CharsIndentifier.features(char,kPredictSize)
    print "识别结果"
    print licence

def test_ann_train():
    print 'test_ann_train'
    import ANNtrain
    # ANNtrain.data_preprocess()
    ANNtrain.train_model()



if __name__ == '__main__':
    # test_plate_locate()
    # test_chars_segment()
    # test_chars_identify()
    test_ann_train()
