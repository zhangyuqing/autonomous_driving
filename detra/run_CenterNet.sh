CENTERNET_PATH=/home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet

export PYTHONPATH=$CENTERNET_PATH/src:$CENTERNET_PATH/src/lib

python $CENTERNET_PATH/src/demo.py ctdet \
    --demo /home/yuqingz/autonomous_driving/examples/data/image_scene2 \
    --load_model $CENTERNET_PATH/models/ctdet_coco_hg.pth \
    --arch hourglass



