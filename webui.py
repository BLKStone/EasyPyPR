# -*- coding:utf-8 -*-
import os
from flask import Flask
from flask import render_template
from flask import request, redirect, url_for
from flask import send_from_directory
from flask import session
from werkzeug import secure_filename


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/error')
def error_page():
    return render_template('error.html')

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