#!/home/yuqingz/anaconda3/envs/CenterTrk/bin python3
#!/bin bash

CENTERTRACK_PATH=./exploration/img_cttrk/CenterTrack

export PYTHONPATH=$CENTERTRACK_PATH/src/lib:$CENTERTRACK_PATH/src/lib/model/networks/DCNv2

python ./detra/demo_track.py tracking,ddd \
    --load_model $CENTERTRACK_PATH/models/nuScenes_3Dtracking.pth \
    --dataset nuscenes --pre_hm --track_thresh 0.1 \
    --test_focal_length 633 \
    --demo $1 \
    --output_dir $2
#    --demo /home/yuqingz/autonomous_driving/exploration/lidar_ptrcnn/data/argoverse-tracking/val/2c07fcda-6671-3ac0-ac23-4a232e0e031e/ring_front_center

    
