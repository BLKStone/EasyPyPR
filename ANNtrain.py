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
    1: [ 'zh_cuan',    '川'],
    2: [ 'zh_gan1',    '甘'],
    3: [ 'zh_hei',     '黑'],
    4: [ 'zh_jin',     '津'],
    5: [ 'zh_liao',    '辽'],
    6: [ 'zh_min',     '闽'],
    7: [ 'zh_qiong',   '琼'],
    8: [ 'zh_sx',      '晋'],
    9: [ 'zh_xin',     '新'],
    10:[ 'zh_yue',     '粤'],
    11:[ 'zh_zhe',     '浙'],
    12:[ 'zh_e',       '鄂'],
    13:[ 'zh_gui',     '贵'],
    14:[ 'zh_hu',      '沪'],
    15:[ 'zh_jing',    '京'],
    16:[ 'zh_lu',      '鲁'],
    17:[ 'zh_ning',    '宁'],
    18:[ 'zh_shan',    '陕'],
    19:[ 'zh_wan',     '皖'],
    20:[ 'zh_yu',      '豫'],
    21:[ 'zh_yun',     '云'],
    22:[ 'zh_gan',     '赣'],
    23:[ 'zh_gui1',    '桂'],
    24:[ 'zh_ji',      '冀'],
    25:[ 'zh_jl',      '吉'],
    26:[ 'zh_meng',    '蒙'],
    27:[ 'zh_qing',    '青'],
    28:[ 'zh_su',      '苏'],
    29:[ 'zh_xiang',   '湘'],
    30:[ 'zh_yu1',     '渝'],
    31:[ 'zh_zang',    '藏'],
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
        print '处理',province_mapping[i+1][1],'中...'
        chinese_path = province_mapping[i+1][0]
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
                    temp_label = np.array([(i+1)])
                    isfirst_char = False
                else:
                    temp_data = np.vstack([temp_data,feature])
                    temp_label = np.hstack([temp_label,(i+1)])

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
    pass



