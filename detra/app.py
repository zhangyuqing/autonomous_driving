from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np
import copy
import json
import cv2
import os
import shlex
from subprocess import check_output
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, render_template, send_from_directory, jsonify

center_track_dir = '/home/yuqingz/autonomous_driving/exploration/img_cttrk/CenterTrack'
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
allowed_ext = image_ext + video_ext + ['txt']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

DATA_DIR = "/home/yuqingz/autonomous_driving/exploration/lidar_ptrcnn/data/argoverse-tracking/val/2c07fcda-6671-3ac0-ac23-4a232e0e031e/ring_front_center"
UPLOAD_DIR = '/home/yuqingz/autonomous_driving/user_data'
RESULT_DIR = f'{UPLOAD_DIR}/processed'
VIZ_DIR = f'{UPLOAD_DIR}/output_gif'

# call_args = [
#     'export', f'PYTHONPATH={center_track_dir}/src/lib:{center_track_dir}/src/lib/model/networks/DCNv2',
#     'python', '/home/yuqingz/autonomous_driving/detra/demo_track.py',
#     'tracking,ddd',
#     '--load_model', model_dir,
#     '--dataset', 'nuscenes',
#     '--pre_hm', '--track_thresh', '0.1', '--test_focal_length', '633',
#     '--demo', data_dir
# ]

app = Flask(
    __name__, template_folder='/home/yuqingz/autonomous_driving/detra/templates', static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'
for local webcam use cv2.VideoCapture(0)
'''


def setup():
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(VIZ_DIR):
        os.makedirs(VIZ_DIR)
    print('Finished setup')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext


@app.route('/', methods=['GET'])
def home():
    a = '''
    <h1>Autonomous Driving System</h1>
    <p>Computer vision has revolutionized the self-driving industry.
    Object detection and tracking are essential tasks that allow vehicles
    to identify obstacles in its course and take action. We hope to develop
    an AI system that is useful in self-driving vehicles.</p>
    '''
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect('/')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # call_args[-1] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            flash('File successfully uploaded')
            return redirect('/')
        else:
            flash(f'Allowed file types are {", ".join(allowed_ext)}')
            return redirect(request.url)


# @app.route('/download')
# def download_file():
#     filename = os.listdir(app.config['UPLOAD_FOLDER'])
#     path = os.path.join(app.config['UPLOAD_FOLDER'], filename[0])
#     return send_file(path, as_attachment=True)


@app.route('/get_files', methods=['GET', 'POST'])
def get_all_runable_files():
    all_files = os.listdir(app.config['UPLOAD_FOLDER'])
    all_files = [os.path.join(app.config['UPLOAD_FOLDER'], file) for file in all_files if (
        file != "output_gif" and file != "processed")]
    all_files = [DATA_DIR] + all_files
    jsn = {'all_files': all_files}
    return jsonify(jsn)


@app.route('/run', methods=['POST'])
def get_prediction():
    file_path = request.form['file_path']
    file_name = file_path.split('/')[-1]
    print(file_name)

    output_dir = f"{UPLOAD_DIR}/processed/{file_name}"
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(f'{VIZ_DIR}/{file_name}.gif'):
        print('Started making prediction')
        out = check_output(
            ['bash', '/home/yuqingz/autonomous_driving/detra/run_CenterTrack_3D_api.sh',
                file_path, output_dir]).decode('utf-8')
        out2 = check_output(
            ["python", "/home/yuqingz/autonomous_driving/detra/detimg2gif.py", output_dir])
    jsn = {'out_gif': f'/viz/{file_name}.gif'}
    return jsonify(jsn)


@app.route('/viz/<filename>')
def get_visualization(filename):
    return send_from_directory(VIZ_DIR, filename)


if __name__ == '__main__':
    setup()
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port=5000, debug=True)
