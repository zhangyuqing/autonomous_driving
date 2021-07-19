cd /Users/yuqingz/GoogleDrive/2021_MLEbootcamp/2_capstone/autonomous_driving/baseline

python ./argoverse-api/argoverse/evaluation/eval_tracking.py \
        --path_tracker_output ./argoverse_cbgs_kf_tracker/ptrcnn_kf_tracking/train-split-track-preds-maxage15-minhits5-conf0.3 \
        --path_dataset /Users/yuqingz/Documents/Argo/argoverse-tracking-v2/train4 \
        --d_max 100

# f"{fn} {num_frames} {mota:.2f} {motp_c:.2f} {motp_o:.2f} {motp_i:.2f} {idf1:.2f} {most_track:.2f} "
#        f"{most_lost:.2f} {num_fp} {num_miss} {num_switch} {num_frag} \n"