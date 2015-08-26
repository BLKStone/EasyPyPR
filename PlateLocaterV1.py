# -*- coding: utf-8 -*-

import numpy as np
import cv2

global m_debug
m_debug = True

# 候选矩形尺寸与宽高比的检查
def verifySize(MinExteriRect):

	# 中国车牌尺寸 440 mm * 140 mm, 宽高比 3.142
	# China car plate size: 440mm * 140mm，aspect 3.142857
	# Real car plate size: 136 * 32, aspect 4
	
	error = 0.5
	aspect = 3.95
	m_verifyMin = 1
	m_verifyMax = 200

	# Set a min and max area. All other patchs are discarded
	# 设置(像素)面积范围
	area_min = 34 * 8 * m_verifyMin # minimum area
	area_max = 34 * 8 * m_verifyMax # maximum area

	# Get only patchs that match to a respect ratio.
	# 设置宽高比范围
	ratio_min = aspect - aspect * error
	ratio_max = aspect + aspect * error

	# 调试模式
	if not m_debug:
		print 'width',MinExteriRect[1][0]
		print 'height',MinExteriRect[1][1]
		print ratio_max
		print ratio_min

	# 高度为0的特殊情况，默认排除
	if MinExteriRect[1][1] == 0:
		return False

	# Calculate area and width-height raito
	# 计算面积与宽高比
	area = MinExteriRect[1][0] * MinExteriRect[1][1]
	#MinExteriRect 的结构为 (top-left corner(x,y), (width, height), angle of rotation )
	raito = MinExteriRect[1][0] / MinExteriRect[1][1]

	if ((area < area_min or area > area_max) or (raito < ratio_min or raito > ratio_max)):
		return False
	else:
		return True

# 获取resize后的选中矩形
#resultMat = showResultMat(img_rotated, rect_size, minRect.center, k++);
def showResultMat(imgRotated,rect_size,center,index):
	global m_debug

	imgCorp = cv2.getRectSubPix(imgRotated,rect_size,center)

	if m_debug:
		picname = 'debug/rotate_fragment_'+str(index)+'.png'
		cv2.imwrite(picname,imgCorp)

	imgResized = cv2.resize(imgCorp,(150,50))

	if m_debug:
		picname = 'debug/rotate_fragment_resize_'+str(index)+'.png'
		cv2.imwrite(picname,imgResized)

	return imgResized



blurBlock = 9

imgsrc = cv2.imread('plate_judge.jpg',1)

# 读取图片
img = cv2.imread('plate_judge.jpg',0)

if m_debug:
	cv2.imshow('image',img)
	cv2.imwrite('debug/image.png',img)

# 高斯模糊
blur = cv2.GaussianBlur(img,(9,9),0)

if m_debug:
	cv2.imshow('blur',blur)
	cv2.imwrite('debug/blur.png',blur)
	# print type(blur)

# Sobel算子检测垂直边缘
sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)

if m_debug:
	cv2.imshow('sobelx',sobelx)
	cv2.imwrite('debug/sobelx.png',sobelx)
	# print type(sobelx)


# 大津算法 二值化
# cv2.threshold(src, thresh, maxval, type[, dst]) → retval, dst
retval,thres = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

if m_debug:
	cv2.imshow('threshold',thres)
	cv2.imwrite('debug/threshold.png',thres)

# 图像中只有 0 与 255
# print type(thres)
# print thres[184:204,103:123]
# cv2.imwrite('threshold.png',thres)


# 形态学运算 闭操作
kernel = np.ones((12,16),np.uint8)
closing = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)

if m_debug:
	cv2.imshow('closing',closing)
	cv2.imwrite('debug/closing.png',closing)


# 求轮廓操作
closing, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# debug 模式，将轮廓绘制出来
imgContours = cv2.drawContours(imgsrc, contours, -1, (0,255,0), 2)

if m_debug:
	cv2.imshow('contours',imgContours)
	cv2.imwrite('debug/contours.png',imgContours)

# cv2.minAreaRect(points) → retval

# #  It returns a Box2D structure which contains following detals - ( top-left corner(x,y), (width, height), angle of rotation )
# mr = cv2.minAreaRect(contours[5])
# print type(mr)
# print mr
# print '---'
# print mr[1]
# print mr[1][0]
# print mr[1][1]
# print '---'

# box = cv2.boxPoints(mr)
# print box
# box = np.int0(box)
# print box
# print type(box)
# print box.dtype
# im = cv2.drawContours(imgsrc,[box],0,(0,0,255),2)

# cv2.imshow('im',im)

# print verifySize(mr)

rotate_rects = []
box_rects = []

for i in range(0,len(contours)):
	# print i
	mr = cv2.minAreaRect(contours[i])

	if not verifySize(mr):
		# del contours[i]
		pass
	else:
		box = cv2.boxPoints(mr)
		box = np.int0(box)

		# print 'box'
		# print box
		rotate_rects.append(mr)
		box_rects.append(box)


# 绘制选择后的
imgsrc = cv2.imread('plate_judge.jpg',1)
imgContoursChosen = cv2.drawContours(imgsrc, box_rects, -1, (255,0,0), 2)

if m_debug:
	cv2.imshow('contours chosen',imgContoursChosen)
	cv2.imwrite('debug/contoursChosen.png',imgContoursChosen)

# 旋转矩形

resultVec = []   # 存储结果的list

for i in range(0,len(rotate_rects)):
	mr = rotate_rects[i]

	if mr[1][1] == 0:
		continue

	ratio = mr[1][0] / mr[1][1]
	angle = mr[2]
	rect_size = [mr[1][0],mr[1][1]]

	if ( ratio < 1 ):
		angle = 90 + angle
		rect_size[0],rect_size[1] = rect_size[1],rect_size[0] # swap height and width



	center_x = (box_rects[i][0][0]+box_rects[i][1][0]+box_rects[i][2][0]+box_rects[i][3][0])/4
	center_y = (box_rects[i][0][1]+box_rects[i][1][1]+box_rects[i][2][1]+box_rects[i][3][1])/4
	center = (center_x,center_y)


	#cv2.getRotationMatrix2D(center, angle, scale) → retval
	# 获取旋转矩阵
	rotmat = cv2.getRotationMatrix2D(center,angle,1)

	print '---'
	print box_rects[i]
	print center_x,center_y
	print 'angle',angle
	print rotmat

	#cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst
	# 
	imgsrc = cv2.imread('plate_judge.jpg',1)
	rows,cols= img.shape
	rotated = cv2.warpAffine(imgsrc, rotmat,(cols,rows))
	picname = 'debug/rotate_'+str(i)+'.png'
	cv2.imwrite(picname,rotated)
	
	print picname

	#接下来的目标是获取到车牌碎块
	print center
	print (int(rect_size[0]),int(rect_size[1]))

	imgResized = showResultMat(rotated,(int(rect_size[0]),int(rect_size[1])),center,i)

	resultVec.append(imgResized)



cv2.waitKey(0)
cv2.destroyAllWindows()


# 参考
# opencv-python sobel http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html?highlight=sobel
# 