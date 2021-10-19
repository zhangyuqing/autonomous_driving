export PYTHONPATH='/home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/src:/home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/src/lib'

cd /home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/data
mkdir kitti
mkdir kitti/training

cp -r /home/yuqingz/autonomous_driving/exploration/img_ctnet/mmdetection3d/data/waymo/kitti_format/training/calib kitti/training

cd kitti
mkdir training/label_5
mkdir images
mkdir images/trainval
mkdir annotations
mkdir ImageSets_waymo

python /home/yuqingz/autonomous_driving/exploration/img_ctnet/data_setup_finetune.py

cd /home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/src/lib
python /home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/src/tools/convert_kitti_to_coco_finetune.py
