import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
# import skimage.io as io
import pylab
import os
import json
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annFile = '/home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/data/kitti/annotations/kitti_waymo_val.json'
expDir = '/home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/exp/ctdet/kitti-finetune'
categories = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
              'Tram', 'Misc', 'DontCare']
# categories = ['__background__', "aeroplane", "bicycle", "bird", "boat",
#               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
#               "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
#               "train", "tvmonitor"]
cat_ids = {cat: i + 1 for i, cat in enumerate(categories)}
N = 100


def det2coco(exp_dir, GT):
    '''read .txt detection result from results/ directory, organize into COCO format'''
    res = []
    images = GT.dataset['images']
    for img in images:
        fn = img['file_name'].replace('.png', '.txt')
        with open(f'{exp_dir}/results/{fn}', 'r') as file:
            for line in file:
                line = line.strip().split()
                # print(line)
                # lab, _, _, _, x1, y1, x2, y2, _, _, _, _, _, _, _, score = line
                # x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                x, y, delta_x, delta_y, score = line[-5], line[-4], line[-3], line[-2], line[-1]
                idx = 0
                while line[idx] != '0.0':
                    idx += 1
                lab = " ".join(line[:idx])
                x, y, delta_x, delta_y = float(x), float(
                    y), float(delta_x), float(delta_y)
                res.append({
                    'image_id': img['id'],
                    'category_id': cat_ids[lab],
                    # [x1, y1, x2-x1, y2-y1],
                    'bbox': [x, y, delta_x, delta_y],
                    'score': float(score)
                })
    outfile = exp_dir + '/det_res.json'
    with open(outfile, 'w') as fout:
        json.dump(res, fout)
    return


def main():
    cocoGt = COCO(annFile)

    det2coco(expDir, cocoGt)
    resFile = expDir + '/det_res.json'
    cocoDt = cocoGt.loadRes(resFile)
    print(f"annFile, {annFile}")
    print(f"resFile, {resFile}")

    imgIds = sorted(cocoGt.getImgIds())
    # N = len(imgIds)
    # imgIds = imgIds[0:N]
    # imgIds = imgIds[np.random.randint(N)]
    # imgIds = [0, 1]
    # print("evaluation on imgIds:", imgIds)

    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    #cocoEval.params.maxDets = [100]
    # cocoEval.params.iouThrs = [0.5, .05, .95]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    main()
