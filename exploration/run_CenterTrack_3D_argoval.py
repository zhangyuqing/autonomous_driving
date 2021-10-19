from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import _init_paths
import os
import json
import numpy as np
from pyquaternion import Quaternion
import pycocotools.coco as coco
from utils import ddd_utils
from dataset.datasets.nuscenes import nuScenes
from nuscenes.utils.data_classes import Box
import argoverse

import sys
sys.path.append(
    "/home/yuqingz/autonomous_driving/exploration/img_cttrk/CenterTrack/src/lib")
sys.path.append(
    "/home/yuqingz/autonomous_driving/exploration/img_cttrk/CenterTrack/src/lib/model/networks/DCNv2")

CENTERTRACK_PATH = "/home/yuqingz/autonomous_driving/exploration/img_cttrk/CenterTrack"
IMAGE_PATH = "/home/yuqingz/autonomous_driving/exploration/lidar_ptrcnn/data/argoverse-tracking/val"
OUT_DIR = "/home/yuqingz/autonomous_driving/examples/argo_3D_track"
CLASS_NAMES = [
    'car', 'truck', 'bus', 'trailer',
    'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
    'traffic_cone', 'barrier']
CLASS_NAMES_MAP = {
    'car': 'VEHICLE', 'truck': 'LARGE_VEHICLE', 'bus': 'BUS',
    'trailer': 'LARGE_VEHICLE', 'construction_vehicle': 'LARGE_VEHICLE',
    'pedestrian': 'PEDESTRIAN', 'motorcycle': 'BICYCLIST', 'bicycle': 'BICYCLIST',
    'traffic_cone': 'ON_ROAD_OBSTACLE', 'barrier': 'ON_ROAD_OBSTACLE'
}
# from datasets/nuscenes.py

if __name__ == '__main__':
    os.system(
        f"export PYTHONPATH={CENTERTRACK_PATH}/src/lib:{CENTERTRACK_PATH}/src/lib/model/networks/DCNv2")

    log_ids = os.listdir(IMAGE_PATH)
    os.system(f"mkdir {OUT_DIR}/track_res")

    for lg in log_ids:
        curr_img_dir = f"{IMAGE_PATH}/{lg}/ring_front_center"
        os.system((f"python {CENTERTRACK_PATH}/src/demo_custom.py tracking,ddd "
                   f"--load_model {CENTERTRACK_PATH}/models/nuScenes_3Dtracking.pth "
                   f"--dataset nuscenes --pre_hm --track_thresh 0.1 "
                   f"--test_focal_length 633 "
                   f"--demo {curr_img_dir}"))

        os.system(f"mkdir {OUT_DIR}/track_res/{lg}")
        os.system(
            f"mkdir {OUT_DIR}/track_res/{lg}/per_sweep_annotations_amodal")

        curr_track_res = json.load(
            open(f"{OUT_DIR}/default_{lg}_results.json"))

        for curr_k in curr_track_res.keys():
            clean_res = []
            curr_ts_objs = curr_track_res[curr_k]
            for obj in curr_ts_objs:
                dim = obj['dim']
                loc = obj['loc']
                size = [float(obj['dim'][1]), float(
                    obj['dim'][2]), float(obj['dim'][0])]

                # self_coco = coco.COCO(ann_path)
                # image_info = self_coco.loadImgs(ids=[image_id])[0]

                rot_cam = Quaternion(axis=[0, 1, 0], angle=obj['rot_y'])
                # box = Box(loc, size, rot_cam, name='2', token='1')
                # box.translate(np.array([0, - box.wlh[2] / 2, 0]))
                # box.rotate(Quaternion(image_info['cs_record_rot']))
                # box.translate(np.array(image_info['cs_record_trans']))
                # box.rotate(Quaternion(image_info['pose_record_rot']))
                # box.translate(np.array(image_info['pose_record_trans']))
                # rotation = box.orientation

                # based on code in ddd_utils.py
                curr_track_label = {
                    "center": {"x": obj['loc'][0], "y": obj['loc'][1], "z": obj['loc'][2]},
                    "rotation": {"w": float(rot_cam.w), "x": float(rot_cam.x), "y": float(rot_cam.y), "z": float(rot_cam.z)},
                    "length": dim[2], "width": dim[1], "height": dim[0],
                    "track_label_uuid": str(obj['tracking_id']),
                    "label_class": CLASS_NAMES_MAP[CLASS_NAMES[obj['class']-1]],
                    "timestamp": curr_k
                }
                clean_res.append(curr_track_label)
            with open(f"{OUT_DIR}/track_res/{lg}/per_sweep_annotations_amodal/tracked_object_labels_{curr_k}.json", 'w') as outfile:
                json.dump(clean_res, outfile)

    os.system(("python /home/yuqingz/autonomous_driving/exploration/lidar_ptrcnn/argoverse-api/argoverse/evaluation/eval_tracking.py "
               f"--path_tracker_output {OUT_DIR}/track_res "
               f"--path_dataset {IMAGE_PATH} "
               "--d_max 100"))
