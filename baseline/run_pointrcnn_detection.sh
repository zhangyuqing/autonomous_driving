cd /home/yuqingz/autonomous_driving/baseline/OpenPCDet/tools

python demo_v2.py \
    --cfg_file cfgs/kitti_models/pointrcnn.yaml \
    --data_path /home/yuqingz/autonomous_driving/baseline/data/ptrcnn_data \
    --ckpt /home/yuqingz/autonomous_driving/baseline/OpenPCDet/checkpoints/pointrcnn_7870.pth \
    --ext .npy \
    --output /home/yuqingz/autonomous_driving/baseline/results/ptrcnn_detect



