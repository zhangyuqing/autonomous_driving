from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths

import os
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector
import sys
sys.path.append(
    "/home/yuqingz/autonomous_driving/exploration/img_cttrk/CenterTrack/src/lib")
sys.path.append(
    "/home/yuqingz/autonomous_driving/exploration/img_cttrk/CenterTrack/src/lib/model/networks/DCNv2")


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']
output_dir = "/home/yuqingz/autonomous_driving/examples/argo_3D_track_1log"


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = 4  # max(opt.debug, 1)
    opt.save_video = True
    opt.save_results = True
    opt.resize_video = True
    opt.video_w, opt.video_h = 800, 500
    # opt.show_track_color = False
    detector = Detector(opt)

    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        is_video = True
        # demo on video stream
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    else:
        is_video = False
        # Demo on images sequences
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

    # Initialize output video
    out = None
    out_name = opt.demo[opt.demo.rfind('/') + 1:]
    print('out_name', out_name)
    if opt.save_video:
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # fourcc = cv2.VideoWriter_fourcc(*'avi1')
        # print('{}/{}'.format(output_dir, opt.exp_id + '_' + out_name))
        out = cv2.VideoWriter('{}/{}'.format(output_dir,
                                             opt.exp_id + '_' + out_name), fourcc, opt.save_framerate, (
            opt.video_w, opt.video_h))

    if opt.debug < 5:
        detector.pause = False
    cnt = 0
    results = {}

    while True:
        if is_video:
            _, img = cam.read()
            if img is None:
                save_and_exit(opt, out, results, out_name=curr_log)
        else:
            if cnt < len(image_names):
                curr_name = image_names[cnt].split("/")
                curr_log, curr_ts = curr_name[9], curr_name[11]
                curr_ts = curr_ts.replace(
                    'ring_front_center_', '').replace('.jpg', '')
                img = cv2.imread(image_names[cnt])
            else:
                save_and_exit(opt, out, results, out_name=curr_log)
        cnt += 1

        # resize the original video for saving video results
        if opt.resize_video:
            img = cv2.resize(img, (opt.video_w, opt.video_h))

        # skip the first X frames of the video
        if cnt < opt.skip_first:
            continue

        # cv2.imshow('input', img)

        # track or detect the image.
        ret = detector.run(img)
        # print(ret.keys())

        # log run time
        time_str = 'frame {} |'.format(cnt)
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)

        # results[cnt] is a list of dicts:
        #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
        results[curr_ts] = ret['results']

        # save debug image to video
        if opt.save_video:
            # print(ret['generic'])
            out.write(ret['generic'])
            # if not is_video:
            cv2.imwrite('{}/{}_{}.jpg'.format(output_dir, curr_log, curr_ts),
                        ret['generic'])

        # esc to quit and finish saving video
        if cv2.waitKey(1) == 27:
            save_and_exit(opt, out, results, out_name=curr_log)
            return
    # save_and_exit(opt, out, results)


def save_and_exit(opt, out=None, results=None, out_name=''):
    if opt.save_results and (results is not None):
        save_dir = '{}/{}_results.json'.format(output_dir,
                                               opt.exp_id + '_' + out_name)
        print('saving results to', save_dir)
        json.dump(_to_list(copy.deepcopy(results)),
                  open(save_dir, 'w'))
    if opt.save_video and out is not None:
        out.release()
    sys.exit(0)


def _to_list(results):
    for img_id in results:
        for t in range(len(results[img_id])):
            for k in results[img_id][t]:
                if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                    results[img_id][t][k] = results[img_id][t][k].tolist()
    return results


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
