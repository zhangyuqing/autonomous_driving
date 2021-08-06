import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse

from collections import Counter, defaultdict
from PIL import Image
import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.data_loading.object_classes import OBJ_CLASS_MAPPING_DICT
from argoverse.data_loading.object_label_record import ObjectLabelRecord
from argoverse.utils.calibration import Calibration
from argoverse.utils.frustum_clipping import generate_frustum_planes
import argoverse.visualization.visualization_utils as viz_util
import argoverse.data_loading.object_label_record as lbl


def make_grid_ring_camera2(argoverse_data, idx, pred_labels):

    f, ax = plt.subplots(3, 3, figsize=(20, 15))

    camera = "ring_front_left"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes2(
        img, objects, pred_labels, calib))
    ax[0, 0].imshow(img_vis)
    ax[0, 0].set_title("Ring Front Left")
    ax[0, 0].axis("off")

    camera = "ring_front_center"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes2(
        img, objects, pred_labels, calib))
    ax[0, 1].imshow(img_vis)
    ax[0, 1].set_title("Right Front Center")
    ax[0, 1].axis("off")

    camera = "ring_front_right"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes2(
        img, objects, pred_labels, calib))
    ax[0, 2].imshow(img_vis)
    ax[0, 2].set_title("Ring Front Right")
    ax[0, 2].axis("off")

    camera = "ring_side_left"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes2(
        img, objects, pred_labels, calib))
    ax[1, 0].imshow(img_vis)
    ax[1, 0].set_title("Ring Side Left")
    ax[1, 0].axis("off")

    ax[1, 1].axis("off")

    camera = "ring_side_right"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes2(
        img, objects, pred_labels, calib))
    ax[1, 2].imshow(img_vis)
    ax[1, 2].set_title("Ring Side Right")
    ax[1, 2].axis("off")

    camera = "ring_rear_left"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes2(
        img, objects, pred_labels, calib))
    ax[2, 0].imshow(img_vis)
    ax[2, 0].set_title("Ring Rear Left")
    ax[2, 0].axis("off")

    ax[2, 1].axis("off")

    camera = "ring_rear_right"
    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(show_image_with_boxes2(
        img, objects, pred_labels, calib))
    ax[2, 2].imshow(img_vis)
    ax[2, 2].set_title("Ring Rear Right")
    ax[2, 2].axis("off")

    return f, ax


def show_image_with_boxes2(img, objects,
                           pred_labels, calib):
    """Show image with 2D bounding boxes."""
    img1 = np.copy(img)

    h, w = np.shape(img1)[0:2]
    planes = generate_frustum_planes(calib.K, calib.camera)
    assert planes is not None

    for obj in objects:
        if obj.occlusion == 100:
            continue
        box3d_pts_3d = obj.as_3d_bbox()
        uv_cam = calib.project_ego_to_cam(box3d_pts_3d)

        img1 = obj.render_clip_frustum_cv2(
            img1,
            uv_cam[:, :3],
            planes.copy(),
            copy.deepcopy(calib.camera_config),
            linewidth=3,
        )

    for plabel in pred_labels:
        plabel = plabel.as_3d_bbox()
        uv_cam_pred = calib.project_ego_to_cam(plabel)
        img1 = obj.render_clip_frustum_cv2(
            img1,
            uv_cam_pred[:, :3],
            planes.copy(),
            copy.deepcopy(calib.camera_config),
            linewidth=3,
            colors=((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        )

    return img1


def main():
    # data_dir = '/home/yuqingz/autonomous_driving/baseline/data/argoverse-tracking/train4'
    # detect_dir = '/home/yuqingz/autonomous_driving/baseline/results/ptrcnn_tune3/epoch50_3cls/format'
    # figure_dir = '/home/yuqingz/autonomous_driving/baseline/figures'

    parser = argparse.ArgumentParser()
    parser.add_argument("-data", "--data",
                        default="/home/yuqingz/autonomous_driving/baseline/data/argoverse-tracking/train4",
                        help="Directory for original data")
    parser.add_argument("-d", "--detect",
                        default="/home/yuqingz/autonomous_driving/baseline/results/ptrcnn_detect_format",
                        help="Output detection with cleaned argoverse format")
    parser.add_argument("-f", "--figure",
                        default="/home/yuqingz/autonomous_driving/baseline/figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    data_dir = args.data
    detect_dir = args.detect
    figure_dir = args.figure

    # print(data_dir)
    # print(detect_dir)
    # print(figure_dir)

    loader = ArgoverseTrackingLoader(data_dir)

    for curr_log in loader.log_list:
        count = 0
        data = loader.get(curr_log)

        # gtdir = data_dir + '/' + curr_log + '/per_sweep_annotations_amodal/'
        ddir = detect_dir + '/' + curr_log + '/per_sweep_annotations_amodal/'

        # label_files = os.listdir(gtdir)
        # label_files.sort()
        detect_files = os.listdir(ddir)
        detect_files.sort()

        # assert len(label_files) == len(detect_files)
        # assert all([label_files[i] == detect_files[i]
        #            for i in range(len(label_files))]) == True

        for idx in range(len(detect_files)):
            if idx % 5 == 0 and count < 6:
                # label_data = json.load(open(gtdir + label_files[idx], 'rb'))
                detect_res = json.load(open(ddir + detect_files[idx], 'rb'))
                ts = detect_files[idx].replace(
                    "tracked_object_labels_", "").replace(".json", "")

                f2 = plt.figure(figsize=(15, 8))
                ax2 = f2.add_subplot(111, projection='3d')
                viz_util.draw_point_cloud(ax2, 'Lidar scan', data, 0)
                for k in range(len(detect_res)):
                    pred_label = lbl.json_label_dict_to_obj_record(
                        detect_res[k])
                    viz_util.draw_box(
                        ax2, pred_label.as_3d_bbox().T, color='red')
                ax2.axis('off')
                f2.savefig(figure_dir + f'/lidar_{curr_log}_{ts}.png', dpi=300)

                count += 1

                # pred_labels = [lbl.json_label_dict_to_obj_record(
                #     detect_res[k]) for k in range(len(detect_res))]
                # f, _ = make_grid_ring_camera2(data, 0, pred_labels)
                # f.savefig(figure_dir + '/camera.png', dpi=300)


if __name__ == '__main__':
    main()
