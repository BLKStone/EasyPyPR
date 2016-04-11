# -*- coding: utf-8 -*-

import numpy as np
import cv2
import PlateLocater
import PlateJudger
import CharsSegmenter
import CharsIndentifier

import DarkChannelRecover

global m_debug
m_debug = True

def plateRecognize(inMat):
    PlateLocater.m_debug = False
    CharsIndentifier.m_debug = False

    Result = PlateLocater.fuzzyLocate(inMat)

    if m_debug:
        print '候选车牌数量：',len(Result)
        index_loc = 0
        for img in Result:
            index_loc += 1
            cv2.imshow("候选车牌"+str(index_loc),img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    resultVec = PlateJudger.platesJudge(Result)
    

    if m_debug:
        print 'SVM筛选后的车牌数量：',len(resultVec)
        index_loc = 0
        for img in resultVec:
            index_loc += 1
            cv2.imshow("SVM-"+str(index_loc),img)
            cv2.imwrite("debug/SVM-"+str(index_loc)+".jpg", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    lisences = list()
    for img_plate in resultVec:
        segmented = CharsSegmenter.charsSegment(img_plate)

        if m_debug:
            index_char = 0
            for char in segmented:
                index_char += 1
                cv2.imshow("chars_segment"+str(index_char), char)
                cv2.imwrite("debug/segmented-"+str(index_char)+".jpg", char)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        lisence = CharsIndentifier.identifyPlate(segmented)
        print lisence
        lisences.append(lisence)


if __name__ == '__main__':
    file_path = 'resources/image/test_plate_foggy_1.jpg'
    imgPlate = cv2.imread(file_path, cv2.IMREAD_COLOR)
    imgPlateDefog = DarkChannelRecover.getRecoverScene(imgPlate)
    plateRecognize(imgPlateDefog)



