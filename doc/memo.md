1. 关于车牌字符训练集I和O的问题

链接：http://www.zhihu.com/question/19847950/answer/31046977

如果你指的是中国大陆的机动车号牌，根据现行中华人民共和国机动车号牌标准《GA36-2007》5.9.1的规定“序号的每一位可单独使用英文字母，26个字母中的O和I不能使用；序号中允许出现两位英文字母，26个字母中的O和I不能使用。所以不存在区分O和0、I和1的可能。如果是其他国家的车牌，根据不同国家的法律可能存在不同的情况，但基本分为两种，1，从字体上区分O和0、I和1。可参照德国车牌。2，不使用O、I如中国。当然在有些以英语国家，假设车牌为“201”，那读出来的可以是“two zero one” 也可以是“two O one”，这种读法可以被认为没有什么区别。


有一部分军车 车牌含有O，一般民用车牌中不会出现。

2. 汉字识别 训练集 为 31个省或自治区

未包括 香港，澳门特别行政区 和 台湾


3. 字符特征提取中的cv2.resize会改变二值化图像
原本图像中只有 0 和 255
经过resize之后会引入其他值
