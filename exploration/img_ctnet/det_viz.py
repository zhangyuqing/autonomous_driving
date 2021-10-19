# code source: https://github.com/zzzxxxttt/simple_kitti_visualization/blob/master/bbox_to_img.py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
# from skimage import io
from matplotlib.patches import Rectangle

colors = sns.color_palette('Paired', 9 * 2)
names = ['Car', 'Van', 'Truck', 'Pedestrian',
         'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
#file_id = '000001'
gt_dir = '/home/yuqingz/autonomous_driving/exploration/data/wod2kitti'
img_dir = '/home/yuqingz/autonomous_driving/exploration/data/wod2kitti_pad/image_0'
exp_dir = '/home/yuqingz/autonomous_driving/exploration/img_ctnet/waymo_pad'
default_resolution = (1280, 384)

if __name__ == '__main__':
    if not os.path.exists(f'{exp_dir}/bbox_figures'):
        os.makedirs(f'{exp_dir}/bbox_figures')

    all_files = os.listdir(img_dir)
    all_ids = [f.replace('.png', '') for f in all_files]

    for file_id in all_ids:
        # load image
        img = cv2.imread(f'{img_dir}/{file_id}.png')
        # img = cv2.resize(img, default_resolution)

        # load detection
        det = None
        if os.path.exists(f'{exp_dir}/results/{file_id}.txt'):
            with open(f'{exp_dir}/results/{file_id}.txt', 'r') as f:
                det = f.readlines()

        # load labels
        labels = None
        if os.path.exists(f'{gt_dir}/label_0/{file_id}.txt'):
            with open(f'{gt_dir}/label_0/{file_id}.txt', 'r') as f:
                labels = f.readlines()

        # load calibration file
        # with open(f'{gt_dir}/calib/{file_id}.txt', 'r') as f:
        #     lines = f.readlines()
        #     P2 = np.array(lines[2].strip().split(' ')[1:],
        #                   dtype=np.float32).reshape(3, 4)

        fig = plt.figure()
        # draw image
        plt.imshow(img)

        # for line in labels:
        #     line = line.split()
        #     lab, _, _, _, x1, y1, x2, y2, _, _, _, _, _, _, _ = line
        #     x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
        #     if lab != 'DontCare':
        #         plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
        #                                       linewidth=2,
        #                                       edgecolor=colors[names.index(lab)],
        #                                       facecolor='none'))
        #         plt.text(x1 + 3, y1 + 3, lab,
        #                  bbox=dict(facecolor=colors[names.index(lab)], alpha=0.5),
        #                  fontsize=7, color='k')
        if det and labels:
            for dt in det:
                dt = dt.split()
                det_lab, _, _, _, xhat1, yhat1, xhat2, yhat2, _, _, _, _, _, _, _, score = dt
                xhat1, yhat1, xhat2, yhat2 = map(
                    float, [xhat1, yhat1, xhat2, yhat2])
                # print(dt)

                # xhat, yhat, delta_xhat, delta_yhat, score = dt[-5], dt[-4], dt[-3], dt[-2], dt[-1]
                # idx = 0
                # while dt[idx] != '0.0':
                #     idx += 1
                # det_lab = " ".join(dt[:idx])
                # xhat, yhat, delta_xhat, delta_yhat = map(
                #     float, [xhat, yhat, delta_xhat, delta_yhat])
                # print(xhat, yhat, delta_xhat, delta_yhat, img.shape)

                if det_lab != 'DontCare':
                    plt.gca().add_patch(Rectangle((xhat1, yhat1), xhat2 - xhat1, yhat2 - yhat1,
                                                  linewidth=2,
                                                  edgecolor=colors[names.index(
                                                      det_lab)],
                                                  facecolor='none'))
                    plt.text(xhat1 + 3, yhat1 + 3, f"{det_lab}, {round(float(score), 2)}",
                             bbox=dict(
                                 facecolor=colors[names.index(det_lab)], alpha=0.5),
                             fontsize=7, color='k')
                    # plt.gca().add_patch(Rectangle((xhat, yhat), delta_xhat, delta_yhat,
                    #                               linewidth=2,
                    #                               edgecolor=colors[names.index(
                    #                                   det_lab) % len(colors)],
                    #                               facecolor='none'))
                    # plt.text(xhat + 3, yhat + 3, f"{det_lab}, {round(float(score), 2)}",
                    #          bbox=dict(
                    #              facecolor=colors[names.index(det_lab) % len(colors)], alpha=0.5),
                    #          fontsize=7, color='k')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{exp_dir}/bbox_figures/{file_id}.png',
                        bbox_inches='tight')
            # plt.show()
            plt.close()
