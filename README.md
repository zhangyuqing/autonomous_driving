## AI System for Autonomous Driving Real-time Detection

For self-driving vehicles, object detection and tracking are essential tasks that allow vehicles to identify obstacles in its course. We hope to develop an AI system that performs object detection and tracking based on state-of-the-art research.

### Goals
+ Develop an AI system to recognize obstacles
+ Determine distances to the identified objects and give collision warnings when needed

### Deliverables

[Web-based API](http://gpu.ronghanghu.com:5000/)

<img src="https://github.com/zhangyuqing/autonomous_driving/blob/main/examples/webAPI_snapshot.png" alt="Web API Snapshot" width="600"/>

### Detection
Detection is performed with [CenterNet model](https://github.com/xingyizhou/CenterNet), pretrained on COCO dataset. CenterNet locolizes objects as their center points. The model was pretrained on MS COCO training images, and validated in ~20k hold-out testing images (test-dev), which achieved 45.1% mAP in multi-scale testing. See [CenterNet paper](https://arxiv.org/pdf/1904.08189.pdf) for details.

We applied the pre-trained CenterNet model ([ddd_3dop](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md)) on ~11k images from [Waymo perception data](https://waymo.com/open/download/) validation split. The model achieved 0.15 mAP (IoU 0.5) on this independent test data, 0.31 mAP on large objects.

<img src="https://github.com/zhangyuqing/autonomous_driving/blob/main/examples/det.gif" alt="Scene 1" width="600"/>

### Detection & tracking system
Detection and tracking pipeline is generated using [CenterTrack](https://github.com/xingyizhou/CenterTrack), pretrained on COCO for 80-category tracking. CenterTrack is a joint detection & tracking algorithm. It relies on CenterNet for detection, then associates the same objects from adjacent frames. The model was pretrained on nuScenes containing 700 image sequences, and validated on 150 sequences from nuScenes test data. The model achieved ~28% AMOTA@0.2 over 7 categories on nuScenes test set. See [CenterTrack paper](https://arxiv.org/pdf/2004.01177.pdf) for details.

We applied the pre-trained CenterTrack 3D model ([nuScenes_3Dtracking](https://github.com/xingyizhou/CenterTrack/blob/master/readme/MODEL_ZOO.md)) on train4 in [Argoverse 3D tracking data](https://www.argoverse.org/data.html#tracking-link). We leveraged the depth estimation from the algorithm output to adapt color of the bounding boxes, such that close enough objects are marked in red.

<img src="https://github.com/zhangyuqing/autonomous_driving/blob/main/examples/trk2.gif" alt="Scene 2" width="600"/>


### Deployment

+ Build API for model that takes video inputs, output detection, tracking, distance monitoring results as the illustration above.
+ Build a web-based UI for the API, takes video uploads, display detection results.


### (Additional) Exploration and learning on LiDAR point cloud data

In addition to using monocular camara images, we explored popular detection algorithms using 3D LiDAR point cloud data. [This directory](https://github.com/zhangyuqing/autonomous_driving/tree/main/exploration/lidar_ptrcnn) contains the experiments using Point R-CNN on point cloud data from the Argoverse 3D tracking dataset. 
