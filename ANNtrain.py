# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
import os
import json
import time

from sklearn.externals import joblib

import CharsIndentifier

reload(sys)
sys.setdefaultencoding('utf-8')

kPredictSize = 10

# 划分训练集和测试集

# 标签 与 文件夹 和 汉字的映射
province_mapping = {
    0: [ 'zh_cuan',    '川'],
    1: [ 'zh_gan1',    '甘'],
    2: [ 'zh_hei',     '黑'],
    3: [ 'zh_jin',     '津'],
    4: [ 'zh_liao',    '辽'],
    5: [ 'zh_min',     '闽'],
    6: [ 'zh_qiong',   '琼'],
    7: [ 'zh_sx',      '晋'],
    8: [ 'zh_xin',     '新'],
    9: [ 'zh_yue',     '粤'],
    10:[ 'zh_zhe',     '浙'],
    11:[ 'zh_e',       '鄂'],
    12:[ 'zh_gui',     '贵'],
    13:[ 'zh_hu',      '沪'],
    14:[ 'zh_jing',    '京'],
    15:[ 'zh_lu',      '鲁'],
    16:[ 'zh_ning',    '宁'],
    17:[ 'zh_shan',    '陕'],
    18:[ 'zh_wan',     '皖'],
    19:[ 'zh_yu',      '豫'],
    20:[ 'zh_yun',     '云'],
    21:[ 'zh_gan',     '赣'],
    22:[ 'zh_gui1',    '桂'],
    23:[ 'zh_ji',      '冀'],
    24:[ 'zh_jl',      '吉'],
    25:[ 'zh_meng',    '蒙'],
    26:[ 'zh_qing',    '青'],
    27:[ 'zh_su',      '苏'],
    28:[ 'zh_xiang',   '湘'],
    29:[ 'zh_yu1',     '渝'],
    30:[ 'zh_zang',    '藏'],
} 


# 持久化存储
def duration_store():
    data_string = json.dumps(province_mapping, sort_keys=True)
    data_string = data_string.decode('unicode_escape')
    # print data_string
    with open('etc/province_mapping.json','w') as f:
        f.write(data_string)
        f.close()


# 训练集 与 测试集的 分割 与 持久化存储
def data_preprocess():
    train_data = np.ones((1,120))
    train_label = np.array([0])
    test_data = np.ones((1,120))
    test_label = np.array([0])

    isfirst_whole = True
    percent = 0.7 # 训练集比例
    np.random.seed(int(time.time()))

    CharsIndentifier.m_debug = False
    basepath = 'resources/train/ann'
    for i in range(0, len(province_mapping)):
        print '处理',province_mapping[i][1],'中...'
        chinese_path = province_mapping[i][0]
        filepath = os.path.join(basepath,chinese_path)
        print "目录:",filepath
        for parent, dirs, files in os.walk( filepath ):

            isfirst_char = True
            for picture in files:
                char_path = os.path.join(parent,picture)
                print char_path 
                char_img = cv2.imread(char_path, cv2.IMREAD_GRAYSCALE)
                threadhold_value, char_threshold = cv2.threshold(char_img,20,255,cv2.THRESH_BINARY)
                # print char_img.shape
                # print char_threshold
                feature = CharsIndentifier.features(char_threshold,kPredictSize)
                print feature.shape
                if isfirst_char:
                    temp_data = feature
                    temp_label = np.array([i])
                    isfirst_char = False
                else:
                    temp_data = np.vstack([temp_data,feature])
                    temp_label = np.hstack([temp_label,i])

            # print temp_data.shape
            # print temp_label.shape

            # 乱序化数据
            # 随机数种子为当前时间 在之前确定
            indices = np.random.permutation(temp_data.shape[0])

            temp_data = temp_data[indices,:]
            temp_label = temp_label[indices]

            # 切分训练集和测试集
            slice_index = int(temp_data.shape[0] * percent)

            temp_train_data = temp_data[0:slice_index,:]
            temp_train_label = temp_label[0:slice_index]

            temp_test_data = temp_data[slice_index:,:]
            temp_test_label = temp_label[slice_index:]

            # print temp_train_data.shape
            # print temp_test_data.shape

            if isfirst_whole:
                train_data = temp_train_data
                train_label = temp_train_label
                test_data = temp_test_data
                test_label = temp_test_label
                isfirst_whole = False
            else:
                train_data = np.vstack([train_data,temp_train_data])
                train_label = np.hstack([train_label,temp_train_label])
                test_data = np.vstack([test_data,temp_test_data])
                test_label = np.hstack([test_label,temp_test_label])

    print train_data.shape
    print test_data.shape

    joblib.dump(train_data, 'data/ann_train_data.pkl')
    joblib.dump(train_label, 'data/ann_train_label.pkl')
    joblib.dump(test_data, 'data/ann_test_data.pkl')
    joblib.dump(test_label, 'data/ann_test_label.pkl')



def train_model():

    # input layer numbers
    feature_size = 120
    # hidden layer numbers
    neurons = 256
    # output layer numbers
    chars_numbers = 31



    # data pre process
    train_data = joblib.load('data/ann_train_data.pkl')
    raw_train_label = joblib.load('data/ann_train_label.pkl')

    print train_data.shape
    print raw_train_label.shape
    print train_data[:10,:10]
    print raw_train_label[:10]
    print raw_train_label[-10:]

    train_label = np.zeros([raw_train_label.shape[0],chars_numbers])
    for i in range(0,raw_train_label.shape[0]):
        train_label[i,raw_train_label[i]] = 1

    print train_label[:10,:]
    print '------------'
    print train_label[-10:,:]
    print train_label.shape
    print '------------'
    print train_label.dtype
    print train_label.astype(np.float32).dtype

    # solve the problem
    # cv2.error: /home/user/opencv-3.1.0/modules/ml/src/data.cpp:251: error: (-215) samples.type() == CV_32F || samples.type() == CV_32S in function setData
    train_data = train_data.astype(np.float32)
    train_label = train_label.astype(np.float32)

    cv_train_data = cv2.ml.TrainData_create(train_data, cv2.ml.ROW_SAMPLE, train_label)


    model = cv2.ml.ANN_MLP_create()

    layer_sizes = np.int32([feature_size, neurons, chars_numbers])
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001)
    criteria2 = (cv2.TERM_CRITERIA_COUNT, 100, 0.001)
    
    model.setLayerSizes(layer_sizes)
    model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
    model.setBackpropWeightScale(0.1)
    model.setBackpropMomentumScale(0.1)
    
    params = dict(term_crit = criteria,
                  train_method = cv2.ml.ANN_MLP_BACKPROP,
                  bp_dw_scale = 0.001,
                  bp_moment_scale = 0.0)
    
    print 'Training MLP ...'
    
    train_flag = model.train(cv_train_data)
    
    print type(model)
    model.save('ann_model')



