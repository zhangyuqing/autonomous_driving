DETECTIONS_DATAROOT="/home/yuqingz/autonomous_driving/baseline/results/argoverse_detections_2020" # replace with your own path
RAW_DATA_DIR="/home/yuqingz/autonomous_driving/baseline/data/argoverse-tracking/train4" # should be either val or test set directory
SPLIT="train" # should be either 'val' or 'test'
OUTDIR="/home/yuqingz/autonomous_driving/baseline/results/cbgs_kf_tracker"
python /home/yuqingz/autonomous_driving/baseline/argoverse_cbgs_kf_tracker/run_ab3dmot.py \
    --dets_dataroot $DETECTIONS_DATAROOT \
    --raw_data_dir $RAW_DATA_DIR \
    --split $SPLIT \
    --tracks_dump_dir $OUTDIR
