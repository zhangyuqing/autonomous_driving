# set up conda environment, go to root directory
# source /home/yuqingz/anaconda3/bin/activate
# conda activate pcdet

cd /home/yuqingz/autonomous_driving/baseline

# parameters
RAW_DATA_DIR="/home/yuqingz/autonomous_driving/baseline/data/argoverse-tracking/train4"
CFG_FILE="/home/yuqingz/autonomous_driving/baseline/OpenPCDet/tools/cfgs/kitti_models/pointrcnn_custom.yaml"

CLEAN=false
CLEAN_DATA_DIR="/home/yuqingz/autonomous_driving/baseline/data/ptrcnn_data"

CKPT_DIR="/home/yuqingz/autonomous_driving/baseline/OpenPCDet/checkpoints/pointrcnn_7870_rmlast.pth"

TUNE=false
BATCH_SIZE=2
EPOCHS=80
TRAIN_EXP_NAME="exp"
TUNE_OUTPUT_DIR="/home/yuqingz/autonomous_driving/baseline/results/tune_ptrcnn"

DETECT_DIR="/home/yuqingz/autonomous_driving/baseline/results/ptrcnn_detect"
DETECT_FORMAT_DIR="/home/yuqingz/autonomous_driving/baseline/results/ptrcnn_detect_format"

TRACK_DIR="/home/yuqingz/autonomous_driving/baseline/results/cbgs_kf_tracker"
TRACK_OUTDIR="ptrcnn_kf_tracking/train-split-track-preds-maxage15-minhits5-conf0.3"

while [[ "$#" -gt 0 ]]
do
    case $1 in
        -prcssdir|--prcssdir)
        CLEAN_DATA_DIR=$2
        CLEAN=true
        ;;
        -ckptdir|--ckptdir)
        CKPT_DIR=$2
        ;;
        -batch_size|--batch_size)
        BATCH_SIZE=$2
        TUNE=true
        ;;
        -epochs|--epochs)
        EPOCHS=$2
        TUNE=true
        ;;
        -exp_name|--exp_name)
        TRAIN_EXP_NAME=$2
        TUNE=true
        ;;
        -tunedir|--tunedir)
        TUNE_OUTPUT_DIR=$2
        TUNE=true
        ;;
        -detect_dir|--detect_dir)
        DETECT_DIR=$2
        ;;
        -detect_format_dir|--detect_format_dir)
        DETECT_FORMAT_DIR=$2
        ;;
        -track_dir|--track_dir)
        TRACK_DIR=$2
        ;;
    esac
    shift
done

echo "CLEAN = $CLEAN"
if $CLEAN; then
    echo "CLEAN_DATA_DIR = $CLEAN_DATA_DIR"
fi
echo "CKPT_DIR = $CKPT_DIR"
echo "TUNE = $TUNE"
if $TUNE; then
    echo "BATCH_SIZE = $BATCH_SIZE"
    echo "EPOCHS = $EPOCHS"
    echo "TRAIN_EXP_NAME = $TRAIN_EXP_NAME"
    echo "TUNE_OUTPUT_DIR = $TUNE_OUTPUT_DIR"
fi
echo "DETECT_DIR = $DETECT_DIR"
echo "DETECT_FORMAT_DIR = $DETECT_FORMAT_DIR"


# process data into ptrcnn data format
if $CLEAN; then
    if [[ ! -e $CLEAN_DATA_DIR ]]; then
        mkdir -p $CLEAN_DATA_DIR
    fi
    python /home/yuqingz/autonomous_driving/baseline/clean_ptrcnn_data.py -o $CLEAN_DATA_DIR
fi


# tune detection PTRCNN model
if $TUNE; then
    cd /home/yuqingz/autonomous_driving/baseline/OpenPCDet/tools
    bash scripts/dist_tune.sh 3 \
        --cfg_file $CFG_FILE \
        --data_path $CLEAN_DATA_DIR \
        --ext ".npy" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --extra_tag $TRAIN_EXP_NAME \
        --ckpt $CKPT_DIR \
        --output $TUNE_OUTPUT_DIR    
    
    cd /home/yuqingz/autonomous_driving/baseline
fi


# run detection using model checkpoint
cd /home/yuqingz/autonomous_driving/baseline/OpenPCDet/tools

if [[ ! -e $DETECT_DIR ]]; then
    mkdir -p $DETECT_DIR
fi

python demo_v2.py \
    --cfg_file $CFG_FILE \
    --data_path $CLEAN_DATA_DIR \
    --ckpt $CKPT_DIR \
    --ext ".npy" \
    --output $DETECT_DIR


# clean detection output format
cd /home/yuqingz/autonomous_driving/baseline/

if [[ ! -e $DETECT_FORMAT_DIR ]]; then
    mkdir -p $DETECT_FORMAT_DIR
fi

python ./clean_ptrcnn_res.py \
    -d $DETECT_DIR \
    -o $DETECT_FORMAT_DIR
    

# evaluate detection
python /home/yuqingz/autonomous_driving/baseline/argoverse-api/argoverse/evaluation/detection/eval.py \
    -d $DETECT_FORMAT_DIR \
    -g "/home/yuqingz/autonomous_driving/baseline/data/argoverse-tracking-v2/train4" \
    -f eval


# run tracking
# python /home/yuqingz/autonomous_driving/baseline/argoverse_cbgs_kf_tracker/run_ab3dmot.py \
#     --dets_dataroot $DETECT_FORMAT_DIR \
#     --raw_data_dir $RAW_DATA_DIR \
#     --split "train" \
#     --tracks_dump_dir $TRACK_DIR


# evaluate tracking
# python /home/yuqingz/autonomous_driving/baseline/argoverse-api/argoverse/evaluation/eval_tracking.py \
#     --path_tracker_output  \
#     --path_dataset "/home/yuqingz/autonomous_driving/baseline/data/argoverse-tracking-v2/train4" \
#     --d_max 100
