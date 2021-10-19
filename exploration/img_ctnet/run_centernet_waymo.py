import os
import cv2
from opts import opts
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

MODEL_PATH = '/home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/models/ddd_3dop.pth'
TASK = 'ddd'
EXP_ID = 'waymo2kitti'
DATASET = 'kitti'
CLASS_NAME = [
    '__background__', 'Pedestrian', 'Car', 'Cyclist']  # KITTI class
default_resolution = (1280, 384)
# from /home/yuqingz/autonomous_driving/2D/CenterNet/src/lib/datasets/dataset/kitti.py
save_dir = '/home/yuqingz/autonomous_driving/exploration/img_ctnet/waymo_pad'

INIT_OPT = [
    TASK,
    '--load_model', MODEL_PATH,
    '--exp_id', EXP_ID,
    '--dataset', DATASET
    #'--arch', 'resdcn_101'
]
opt = opts().init(INIT_OPT)
Dataset = dataset_factory[opt.dataset]
opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
detector = detector_factory[opt.task](opt)

img_dir = '/home/yuqingz/autonomous_driving/exploration/data/wod2kitti_pad/image_0'
imgs = os.listdir(img_dir)

for img in imgs:
    im = cv2.imread(img_dir + '/' + img)
    # im = cv2.resize(im, default_resolution)
    ret = detector.run(im)['results']

    results_dir = os.path.join(save_dir, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    out_path = os.path.join(results_dir, f'{img.replace(".png", "")}.txt')
    f = open(out_path, 'w')
    for cls_ind in ret:
        for j in range(len(ret[cls_ind])):
            class_name = CLASS_NAME[cls_ind]
            f.write('{} 0.0 0'.format(class_name))
            for i in range(len(ret[cls_ind][j])):
                f.write(' {:.2f}'.format(ret[cls_ind][j][i]))
            f.write('\n')
    f.close()
