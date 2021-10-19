import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
from matplotlib.patches import Rectangle

colors = sns.color_palette('Paired', 9 * 2)
names = ['Car', 'Van', 'Truck', 'Pedestrian',
         'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
#file_id = '000001'
gt_dir = '/home/yuqingz/autonomous_driving/exploration/data/wod2kitti'
session_id = '0'
img_dir = f'{gt_dir}/image_{session_id}'
label_dir = f'{gt_dir}/label_{session_id}'
exp_dir = '/home/yuqingz/autonomous_driving/exploration/data/convert_check'
if not os.path.exists(f'{exp_dir}/image_{session_id}'):
    os.makedirs(f'{exp_dir}/image_{session_id}')
default_resolution = (1280, 384)

all_img_files = os.listdir(img_dir)
print(len(all_img_files))
all_img_ids = [f.replace('.png', '') for f in all_img_files]

all_label_files = os.listdir(label_dir)
print(len(all_label_files))
all_label_ids = [f.replace('.txt', '') for f in all_label_files]

all_ids = set(all_img_ids).intersection(set(all_label_ids))
all_ids = list(sorted(all_ids))
print(len(all_ids))

for file_id in all_ids:
    # load image
    img = cv2.imread(f'{img_dir}/{file_id}.png')

    labels = None
    if os.path.exists(f'{label_dir}/{file_id}.txt'):
        with open(f'{label_dir}/{file_id}.txt', 'r') as f:
            labels = f.readlines()


    fig = plt.figure()
    # draw image
    plt.imshow(img)

    for line in labels:
        line = line.split()
        lab, _, _, _, x1, y1, x2, y2, _, _, _, _, _, _, _ = line
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
        if lab != 'DontCare':
            plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            linewidth=2,
                                            edgecolor=colors[names.index(lab)],
                                            facecolor='none'))
            plt.text(x1 + 3, y1 + 3, lab,
                        bbox=dict(facecolor=colors[names.index(lab)], alpha=0.5),
                        fontsize=7, color='k')
        
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{exp_dir}/{file_id}.png',
                bbox_inches='tight')
    # plt.show()