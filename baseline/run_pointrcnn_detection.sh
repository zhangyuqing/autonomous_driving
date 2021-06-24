cd ./OpenPCDet/tools

python demo_v2.py \
    --cfg_file cfgs/kitti_models/pointrcnn.yaml \
    --ckpt /home/yuqingz/autonomous_driving/baseline/OpenPCDet/checkpoints/pointrcnn_7870.pth \
    --data_path /home/yuqingz/autonomous_driving/baseline/ptrcnn_data \
    --ext .npy \
    --output /home/yuqingz/autonomous_driving/baseline/ptrcnn_detect/
