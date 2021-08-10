CENTERTRACK_PATH=/home/yuqingz/autonomous_driving/exploration/img_cttrk/CenterTrack

export PYTHONPATH=$CENTERTRACK_PATH/src:$CENTERTRACK_PATH/src/lib:$CENTERTRACK_PATH/src/lib/model/networks/DCNv2

python $CENTERTRACK_PATH/src/demo_custom.py tracking \
    --load_model $CENTERTRACK_PATH/models/coco_tracking.pth \
    --demo /home/yuqingz/autonomous_driving/examples/nuscenes_mini.mp4
