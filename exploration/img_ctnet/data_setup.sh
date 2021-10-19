export PYTHONPATH='/home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/src:/home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/src/lib'

cd /home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/data
mkdir kitti
mkdir kitti/training

cp -r /home/yuqingz/autonomous_driving/exploration/data/wod2kitti/calib kitti/training
cp -r /home/yuqingz/autonomous_driving/exploration/data/wod2kitti/label_0 kitti/training

cd kitti
mkdir images
mkdir images/trainval
mkdir annotations

cp -r /home/yuqingz/autonomous_driving/exploration/data/wod2kitti/image_0/*.png images/trainval

mkdir ImageSets_waymo
ls training/label_0 | tr -dc '[:digit:]|\n' > ImageSets_waymo/trainval.txt

cd /home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/src/lib
python /home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/src/tools/convert_kitti_to_coco_custom.py
