# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
import os
import json
import time
import yaml
import re

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

digit_letter_mapping = {
    0: [ '0',    '0'],
    1: [ '1',    '1'],
    2: [ '2',    '2'],
    3: [ '3',    '3'],
    4: [ '4',    '4'],
    5: [ '5',    '5'],
    6: [ '6',    '6'],
    7: [ '7',    '7'],
    8: [ '8',    '8'],
    9: [ '9',    '9'],
    10:[ '10',   'A'],
    11:[ '11',   'B'],
    12:[ '12',   'C'],
    13:[ '13',   'D'],
    14:[ '14',   'E'],
    15:[ '15',   'F'],
    16:[ '16',   'G'],
    17:[ '17',   'H'],
  # 18:[ '18',   'I'], 根据现行中华人民共和国机动车号牌标准《GA36-2007》5.9.1的规定“序号的每一位可单独使用英文字母，26个字母中的O和I不能使用；
    18:[ '18',   'J'],
    19:[ '19',   'K'],
    20:[ '20',   'L'],
    21:[ '21',   'M'],
    22:[ '22',   'N'],
  # 24:[ '24',   'O'],
    23:[ '23',   'P'],
    24:[ '24',   'Q'],
    25:[ '25',   'R'],
    26:[ '26',   'S'],
    27:[ '27',   'T'],
    28:[ '28',   'U'],
    29:[ '29',   'V'],
    30:[ '30',   'W'],
    31:[ '31',   'X'],
    32:[ '32',   'Y'],
    33:[ '33',   'Z'],
} 


# 持久化存储
def duration_store():
    data_string = json.dumps(province_mapping, sort_keys=True)
    data_string = data_string.decode('unicode_escape')
    # print data_string
    with open('etc/province_mapping.json','w') as f:
        f.write(data_string)
        f.close()


# 用相同数据集训练的模型
# 多次训练后predict结果几乎一样
# 但在数据中，根据不同的随机选择0.7的数据
# 产生的模型预测的结果却有很大不同
#
def search_good_model():

    data_path = "../goodmodel/ann_digit_letter_train_data_"
    label_path = "../goodmodel/ann_digit_letter_train_label_"

    # 
    evaluation_data, evaluation_label = digit_letter_data_preprocess()
    evaluation_data = evaluation_data.astype(np.float32)
    evaluation_label = evaluation_label.astype(np.float32)

    # input layer numbers
    feature_size = 120
    # hidden layer numbers
    neurons = 256
    # output layer numbers
    chars_numbers = 34

    scores = np.array([])
    performance_rank = list()
    with open('model_performance.txt','w') as f:
        for i in range(1000):
            data_path = '../goodmodel/ann_digit_letter_train_data_'+str(i)
            label_path = '../goodmodel/ann_digit_letter_train_label_'+str(i)
            model = train_digit_letter_model(data_path, label_path)
            score = calculate_score(model, evaluation_data, evaluation_label)
            print i,score
            f.write(''+str(i)+": "+str(score)+"\n")
            scores = np.hstack([scores,score])
            performance_rank.append((i,score))

    
        print 'max performance:', np.max(scores)
        f.write('max performance:'+ str(np.max(scores))+'\n')
        print 'index:', np.where(scores == np.max(scores))[0][0]
        f.write('index:'+ str(np.where(scores == np.max(scores))[0][0]) + '\n')

        performance_rank = sorted(performance_rank, key = lambda x: x[1], reverse = True)
        f.write(str(performance_rank[:10]))
    f.close()
    


#
def calculate_score(model, evaluation_data, evaluation_label):
    
    output = model.predict(evaluation_data)[1]

    result = np.array([])
    for i in range(evaluation_data.shape[0]):
        res_tup = np.where(output[i,:] == np.max(output[i,:]))
        max_pos = res_tup[0][0]
        result = np.hstack([result, max_pos])

    result = result.astype(np.float32)
    precition = result == evaluation_label

    score =  np.count_nonzero(precition)/float(precition.shape[0])*100
    # print result.shape
    # print result.dtype
    # print evaluation_label.shape
    # print evaluation_label.dtype
    return score



def generate_data():

    # temp_data = np.array([[1,2],
    #                       [3,4],
    #                       [5,6],
    #                       [7,8],
    #                       [9,8]])
    # temp_label = np.array([1,2,3,4,5])

    # train_data, train_label, test_data, test_label = random_slice(temp_data, temp_label)

    # print train_data
    # print train_label
    # print test_data
    # print test_label
    # ---------------------------------------------------------------------------------------------

    temp_data, temp_label = digit_letter_data_preprocess()
    with open('../goodmodel/readme.txt','w') as f:
        for i in range(1000):
            seed = int(time.time()*1000%100000)
            np.random.seed(seed)
            train_data, train_label, test_data, test_label = random_slice(temp_data, temp_label)
            f.write(""+str(i)+":"+str(seed)+"\n")
            joblib.dump(train_data, '../goodmodel/ann_digit_letter_train_data_'+str(i))
            joblib.dump(train_label, '../goodmodel/ann_digit_letter_train_label_'+str(i))
            joblib.dump(test_data, '../goodmodel/ann_digit_letter_test_data_'+str(i))
            joblib.dump(test_label, '../goodmodel/ann_digit_letter_test_label_'+str(i))
    f.close()




    


# 训练
def digit_letter_data_preprocess(): 

    train_data = np.ones((1,120))
    train_label = np.array([0])
    test_data = np.ones((1,120))
    test_label = np.array([0])

    isfirst_whole = True
    percent = 0.7 # 训练集比例
    CharsIndentifier.m_debug = False
    basepath = 'resources/train/ann'
    for i in range(0, len(digit_letter_mapping)):

        remain_path = digit_letter_mapping.get(i,'badpath')

        if remain_path == 'badpath':
            print 'jump index %d' % i
            continue

        filepath = os.path.join(basepath,remain_path[1])
        for parent, dirs, files in os.walk( filepath ):
            print "目录:",filepath, "文件数:",len(files)
            isfirst_char = True
            for picture in files:
                char_path = os.path.join(parent,picture)
                # print char_path 
                char_img = cv2.imread(char_path, cv2.IMREAD_GRAYSCALE)
                threadhold_value, char_threshold = cv2.threshold(char_img,20,255,cv2.THRESH_BINARY)
                # print char_img.shape
                # print char_threshold
                feature = CharsIndentifier.features(char_threshold,kPredictSize)
                # print feature.shape
                if isfirst_char:
                    temp_train_data = feature
                    temp_train_label = np.array([i])
                    isfirst_char = False
                else:
                    temp_train_data = np.vstack([temp_train_data,feature])
                    temp_train_label = np.hstack([temp_train_label,i])

        if isfirst_whole:
            train_data = temp_train_data
            train_label = temp_train_label
            isfirst_whole = False
        else:
            train_data = np.vstack([train_data,temp_train_data])
            train_label = np.hstack([train_label,temp_train_label])

    print 'digit_letter_data_preprocess'
    print train_data.shape
    print train_label.shape

    return (train_data, train_label)
 

def random_slice(temp_data, temp_label):

    percent = 0.7 # 训练集比例

    # 乱序化数据
    # 随机数种子为当前时间 在之前确定
    indices = np.random.permutation(temp_data.shape[0])
    temp_data = temp_data[indices,:]
    temp_label = temp_label[indices]

    # 切分训练集和测试集
    slice_index = int(temp_data.shape[0] * percent)
    train_data = temp_data[0:slice_index,:]
    train_label = temp_label[0:slice_index]
    test_data = temp_data[slice_index:,:]
    test_label = temp_label[slice_index:]
    
    # print train_data.shape
    # print test_data.shape
    
    return (train_data, train_label, test_data, test_label)







# 训练集 与 测试集的 分割 与 持久化存储
def chinese_data_preprocess():
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

    joblib.dump(train_data, 'data/ann_chinese_train_data.pkl')
    joblib.dump(train_label, 'data/ann_chinese_train_label.pkl')
    joblib.dump(test_data, 'data/ann_chinese_test_data.pkl')
    joblib.dump(test_label, 'data/ann_chinese_test_label.pkl')




def train_digit_letter_model(data_path, label_path):

    # input layer numbers
    feature_size = 120
    # hidden layer numbers
    neurons = 256
    # output layer numbers
    chars_numbers = 34

    # data_path = 'data/ann_digit_letter_train_data.pkl'
    # label_path = 'data/ann_digit_letter_train_label.pkl'
    train_data = joblib.load(data_path)
    raw_train_label = joblib.load(label_path)

    train_label = np.zeros([raw_train_label.shape[0],chars_numbers])
    for i in range(0,raw_train_label.shape[0]):
        train_label[i,raw_train_label[i]] = 1

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

    print 'Training MLP ...'
    train_flag = model.train(cv_train_data)
    print 'Taining finishing...'

    # joblib.dump(model, 'model/ann_chinese.pkl')
    # print json.dumps(model)
    # model.save('model/ann_digit_letter.xml')

    return model






def train_chinese_model():

    # input layer numbers
    feature_size = 120
    # hidden layer numbers
    neurons = 256
    # output layer numbers
    chars_numbers = 31

    # data pre process
    train_data = joblib.load('data/ann_chinese_train_data.pkl')
    raw_train_label = joblib.load('data/ann_chinese_train_label.pkl')

    train_label = np.zeros([raw_train_label.shape[0],chars_numbers])
    for i in range(0,raw_train_label.shape[0]):
        train_label[i,raw_train_label[i]] = 1

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
    print 'Taining finishing...'

    # joblib.dump(model, 'model/ann_chinese.pkl')
    # print json.dumps(model)
    model.save('model/ann_chinese.xml')

    return model



# Since OpenCV yaml file is incompatible with python yaml
# Few reasons for incompatibility are:
# 1. Yaml created by OpenCV doesn't have a space after ":". Whereas Python requires it. [Ex: It should be a: 2, and not a:2 for Python]
# 2. First line of YAML file created by OpenCV is wrong. Either convert "%YAML:1.0" to "%YAML 1.0". Or skip the first line while reading.
def read_YAML_file(fileName):
    ret = {}
    skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
        yamlFileOut = myRe.sub(r': \1', yamlFileOut)
        ret = yaml.load(yamlFileOut)
    return ret

def load_chinese_model():
    model = cv2.FileStorage('model/ann_chinese.xml',cv2.FileStorage_READ)




if __name__ == '__main__':
    # generate_data()
    search_good_model()

