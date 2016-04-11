# -*- coding:utf-8 -*-
import os
from flask import Flask
from flask import render_template
from flask import request, redirect, url_for
from flask import send_from_directory
from flask import session
from werkzeug import secure_filename

import PlateRecognizer
import DarkChannelRecover
import cv2

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/error')
def error_page():
    error_message = session.get('error_message','')
    return render_template('error.html',error_message = error_message)


@app.route('/search', methods=['POST', 'GET'])
def search_page():
    if request.method == 'POST':
        print request.form
        print request.form['wd']
        # 获取请求关键词
        keyword = request.form['wd']
    # 如果请求访求是 GET 或验证未通过就会执行下面的代码

    if request.method == 'GET':

        # 获取图片路径 
        image_url = session.get('filepath', '')

        if session['filepath'] == '': 
            session['error_message'] = "未获取到图片"
            return redirect(url_for('error_page'))

        
        print "图片路径", image_url
        try: 
            # 初始化图像特征描述符
        
            imgPlate = cv2.imread(image_url, cv2.IMREAD_COLOR)
            # imgPlateDefog = DarkChannelRecover.getRecoverScene(imgPlate)
            PlateRecognizer.m_debug = False
            licenses = PlateRecognizer.plateRecognize(imgPlate)
            
            print licenses,'test'
            session['licenses'] = licenses
 
        except:
            session['error_message'] = "calculate error"
            redirect(url_for('error_page'))

    return redirect(url_for('result_page'))

@app.route('/result')
def result_page():

    # 获取最近上传的图像名称
    img_name = session.get('filepath','')
    if img_name == '':
        session['error_message'] = "未获取到计算结果"
        return redirect(url_for('error_page'))

    img_name = session['filepath'].split('/')[-1]

    # 获取检索结果
    licenses = session.get('licenses','')
    if licenses == '':
        session['error_message'] = "未获取到计算结果"
        return redirect(url_for('error_page'))

    return render_template('result.html', img_name = img_name, licenses = licenses)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print "获取到上传文件"
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['filepath'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            return render_template('up_success.html')
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.secret_key = 'A0Zr98j/3sadayX R~XHH!jmN]LWX/,?RTsak'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.debug = True
    app.run('0.0.0.0')
