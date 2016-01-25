# -*- coding: utf-8 -*-

import numpy as np
import cv2
from sklearn import svm
import os
import PlateLocater
import PlateJudger


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





if __name__ == '__main__':
    testPlateLocater()