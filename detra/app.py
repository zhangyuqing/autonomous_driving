from flask import Flask, flash, request, redirect, send_file, render_template
from werkzeug.utils import secure_filename
from subprocess import check_output
import shlex

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import cv2
import json
import copy
import numpy as np
# from opts import opts
# from detector import Detector
# from demo_track import demo, save_and_exit, _to_list
import sys

# sys.path.append(f"{center_track_dir}/src/lib")
# sys.path.append(f"{center_track_dir}/src/lib/model/networks/DCNv2")

center_track_dir = '/home/yuqingz/autonomous_driving/exploration/img_cttrk/CenterTrack'
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
allowed_ext = image_ext + video_ext + ['txt']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

output_dir = "/home/yuqingz/autonomous_driving/examples/argo_3D_track_1log"
model_dir = f"{center_track_dir}/models/nuScenes_3Dtracking.pth"
data_dir = "/home/yuqingz/autonomous_driving/exploration/lidar_ptrcnn/data/argoverse-tracking/val/2c07fcda-6671-3ac0-ac23-4a232e0e031e/ring_front_center"

call_args = [
    'export', f'PYTHONPATH={center_track_dir}/src/lib:{center_track_dir}/src/lib/model/networks/DCNv2',
    'python', '/home/yuqingz/autonomous_driving/detra/demo_track.py',
    'tracking,ddd',
    '--load_model', model_dir,
    '--dataset', 'nuscenes',
    '--pre_hm', '--track_thresh', '0.1', '--test_focal_length', '633',
    '--demo', data_dir
]

app = Flask(
    __name__, template_folder='/home/yuqingz/autonomous_driving/detra/templates')
app.config['UPLOAD_FOLDER'] = '/home/yuqingz/autonomous_driving/user_data'

'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'
for local webcam use cv2.VideoCapture(0)
'''


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext


@app.route('/', methods=['GET'])
def home():
    return '''
    <h1>Autonomous Driving System</h1>
    <p>Computer vision has revolutionized the self-driving industry.
    Object detection and tracking are essential tasks that allow vehicles
    to identify obstacles in its course and take action. We hope to develop
    an AI system that is useful in self-driving vehicles.</p>
    '''


@app.route('/upload')
def upload():
    return render_template('upload.html')


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


@app.route('/download')
def download_file():
    filename = os.listdir(app.config['UPLOAD_FOLDER'])
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename[0])
    return send_file(path, as_attachment=True)


@app.route('/run')
def run():
    return render_template('run.html')


@app.route('/run', methods=['GET', 'POST'])
def get_prediction():
    if request.method == 'POST':
        print('Started making prediction')
        out = check_output(
            ['bash', '/home/yuqingz/autonomous_driving/detra/run_CenterTrack_3D_api.sh']).decode('utf-8')
        return "run successfully"
    return "not executed"


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port=5000, debug=True)
