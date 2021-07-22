import torch
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import json
import argparse

# detect_dir = '/home/yuqingz/autonomous_driving/baseline/results/ptrcnn_tune2/overfit_trn1/detect'
# output_dir = '/home/yuqingz/autonomous_driving/baseline/results/ptrcnn_tune2/overfit_trn1/format'

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--detect",
                    default="/home/yuqingz/autonomous_driving/baseline/results/ptrcnn_detect",
                    help="Directory for detection result")
parser.add_argument("-o", "--output",
                    default="/home/yuqingz/autonomous_driving/baseline/results/ptrcnn_detect_format",
                    help="Output detection with cleaned argoverse format")
args = parser.parse_args()
if args.detect:
    detect_dir = args.detect
if args.output:
    output_dir = args.output

# print(detect_dir)
# print(output_dir)

# CATEGORIES = ['Car', 'Pedestrian', 'Cyclist'] # from pointrcnn.yaml
# use the same annotation as CBGS
# CATEGORIES = ['VEHICLE', 'PEDESTRIAN', 'BICYCLIST']
CATEGORIES = ['VEHICLE', 'PEDESTRIAN', 'BICYCLIST',
              'BUS', 'ON_ROAD_OBSTACLE', 'LARGE_VEHICLE']

detect_files = os.listdir(detect_dir)
label_out = {}
test = set()
for file in detect_files:
    fname = file.replace('.pkl', '').split('__')
    curr_log, lidar_timestamp = fname[0], fname[1]

    # if curr_log == '91326240-9132-9132-9132-591327440896':
    #     test.add(int(lidar_timestamp))

    res = pickle.load(open(detect_dir + '/' + file, 'rb'))
    res = res[0]

    # if int(lidar_timestamp) in {315969825919914000, 315969826219990000, 315969826019597000}:
    #     print(res)  # nothing was predicted

    n_object = len(res['pred_scores'])
    for obj in range(n_object):
        # convert Euler angle to quaternions
        curr_hd = res['pred_boxes'][obj, 6].detach().cpu().numpy().item()
        r = - curr_hd - (np.pi / 2)  # theta: rotation angle on Z
        rot = Rotation.from_euler('xyz', [0, 0, r], degrees=True)
        rot_quat = rot.as_quat()  # [x, y, z, w]

        curr_label = {
            'track_label_uuid': None,
            'tracked': True,
            'occlusion': 0,
            'timestamp': int(lidar_timestamp),
            'label_class': CATEGORIES[res['pred_labels'][obj]-1],
            'score': res['pred_scores'][obj].detach().cpu().numpy().item(),
            # transformation based on pcdet/datasets/waymo/waymo_eval.py,
            # boxes3d_kitti_fakelidar_to_lidar function
            'center': {
                'x': res['pred_boxes'][obj, 0].detach().cpu().numpy().item(),
                'y': res['pred_boxes'][obj, 1].detach().cpu().numpy().item(),
                # * 2,
                'z': res['pred_boxes'][obj, 2].detach().cpu().numpy().item(),
            },
            'rotation': {
                'w': rot_quat[3],
                'x': rot_quat[0],
                'y': rot_quat[1],
                'z': rot_quat[2]
            },
            'height': res['pred_boxes'][obj, 5].detach().cpu().numpy().item(),
            'width': res['pred_boxes'][obj, 4].detach().cpu().numpy().item(),
            'length': res['pred_boxes'][obj, 3].detach().cpu().numpy().item()
        }
        if curr_log not in label_out:
            label_out[curr_log] = []
        label_out[curr_log].append(curr_label)


for k in label_out.keys():
    dir_name = output_dir + '/' + k
    try:
        os.makedirs(dir_name)
        os.makedirs(dir_name + '/per_sweep_annotations_amodal')
        print("Directory ", dir_name,  " Created ")
    except FileExistsError:
        print("Directory ", dir_name,  " already exists")

    all_time = [item['timestamp'] for item in label_out[k]]
    all_time = list(set(all_time))
    print(k, len(all_time))

    for t in all_time:
        label_currtime = [item for item in label_out[k]
                          if item['timestamp'] == t]
        output_name = dir_name + \
            f'/per_sweep_annotations_amodal/tracked_object_labels_{str(t)}.json'
        with open(output_name, 'w') as outfile:
            json.dump(label_currtime, outfile)
            # print(f"saved {label_currtime}")
