
cd /Users/yuqingz/GoogleDrive/2021_MLEbootcamp/2_capstone/autonomous_driving/baseline

python ./argoverse-api/argoverse/evaluation/detection/eval.py \
        --dt_fpath /Users/yuqingz/Documents/Argo/ptrcnn_detect_format_tune1 \
        --gt_fpath /Users/yuqingz/Documents/Argo/argoverse-tracking/train4 \
        -f eval