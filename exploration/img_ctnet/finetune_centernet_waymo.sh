cd /home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/src/lib

python /home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/src/main.py ctdet \
    --dataset kitti --kitti_split waymo --exp_id kitti-finetune \
    --batch_size 16 \
    --master_batch 7 \
    --num_epochs 10 \
    --lr_step 45,60 \
    --gpus 0,1

python /home/yuqingz/autonomous_driving/exploration/img_ctnet/CenterNet/src/test.py ctdet \
    --dataset kitti --kitti_split waymo --exp_id kitti-finetune \
    --keep_res --resume