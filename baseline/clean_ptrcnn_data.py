import numpy as np
import pickle
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from scipy.spatial.transform import Rotation
import argparse

data_dir = "/home/yuqingz/autonomous_driving/baseline/data/argoverse-tracking"

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--Output",
                    default="/home/yuqingz/autonomous_driving/baseline/data/ptrcnn_data",
                    help="Output directory for cleaned data")
args = parser.parse_args()
if args.Output:
    output_dir = args.Output
# output_dir = "/home/yuqingz/autonomous_driving/baseline/data/ptrcnn_data"

trn_loader = ArgoverseTrackingLoader(data_dir + '/train4')
val_loader = ArgoverseTrackingLoader(data_dir + '/val')

# process train data
for i, data in enumerate(trn_loader):
    for lidar_idx in range(len(data.lidar_list)):
        # get lidar point cloud
        lidar_pts = data.get_lidar(lidar_idx)

        lidar_pts = np.append(lidar_pts, np.zeros((lidar_pts.shape[0], 1)),
                              axis=1)
        assert lidar_pts.shape[1] == 4
        file_name = output_dir + '/train/' + data.current_log + \
            '__' + str(data.lidar_timestamp_list[lidar_idx]) + '.npy'
        np.save(file_name, lidar_pts)

        # get detected objects
        objs = data.get_label_object(lidar_idx)
        gt_boxes = []
        gt_names = []
        for obj in objs:
            curr_box = obj.translation  # box center
            w, l, h = obj.width, obj.length, obj.height
            # curr_box[2] += h / 2
            dx, dy, dz = l, w, h
            # from quoternion to Eular angle
            quat = [obj.quaternion[1], obj.quaternion[2],
                    obj.quaternion[3], obj.quaternion[0]]
            # object quaternion w x y z
            rot = Rotation.from_quat(quat)
            rot_euler = rot.as_euler('xyz', degrees=True)[2]
            heading = -(rot_euler + np.pi / 2)

            curr_box = list(curr_box)
            curr_box.extend([dx, dy, dz, heading])
            gt_boxes.append(curr_box)

            gt_names.append(obj.label_class)

        curr_label = {'gt_boxes': gt_boxes, 'gt_names': gt_names}
        label_file_name = file_name.replace('.npy', '.pkl')
        pickle.dump(curr_label, file=open(label_file_name, 'wb'))


# process val data
for i, data in enumerate(val_loader):
    for lidar_idx in range(len(data.lidar_list)):
        # get lidar point cloud
        lidar_pts = data.get_lidar(lidar_idx)

        lidar_pts = np.append(lidar_pts, np.zeros((lidar_pts.shape[0], 1)),
                              axis=1)
        assert lidar_pts.shape[1] == 4
        file_name = output_dir + '/val/' + data.current_log + \
            '__' + str(data.lidar_timestamp_list[lidar_idx]) + '.npy'
        np.save(file_name, lidar_pts)

        # get detected objects
        objs = data.get_label_object(lidar_idx)
        gt_boxes = []
        gt_names = []
        for obj in objs:
            curr_box = obj.translation  # box center
            w, l, h = obj.width, obj.length, obj.height
            # curr_box[2] += h / 2
            dx, dy, dz = l, w, h
            # from quoternion to Eular angle
            quat = [obj.quaternion[1], obj.quaternion[2],
                    obj.quaternion[3], obj.quaternion[0]]
            # object quaternion w x y z
            rot = Rotation.from_quat(quat)
            rot_euler = rot.as_euler('xyz', degrees=True)[2]
            heading = -(rot_euler + np.pi / 2)

            curr_box = list(curr_box)
            curr_box.extend([dx, dy, dz, heading])
            gt_boxes.append(curr_box)

            gt_names.append(obj.label_class)

        curr_label = {'gt_boxes': gt_boxes, 'gt_names': gt_names}
        label_file_name = file_name.replace('.npy', '.pkl')
        pickle.dump(curr_label, file=open(label_file_name, 'wb'))
