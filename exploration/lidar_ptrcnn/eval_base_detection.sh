
cd /Users/yuqingz/GoogleDrive/2021_MLEbootcamp/2_capstone/autonomous_driving/baseline

python ./argoverse-api/argoverse/evaluation/detection/eval.py \
        --dt_fpath /Users/yuqingz/Documents/Argo/argoverse_detections_2020_train4_trn \
        --gt_fpath /Users/yuqingz/Documents/Argo/argoverse-tracking-v3/train4 \
        -f ./