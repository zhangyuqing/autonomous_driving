# set up conda environment, go to root directory
# source /home/yuqingz/anaconda3/bin/activate
# conda activate pcdet

cd /home/yuqingz/autonomous_driving/baseline

# parameters
RAW_DATA_DIR="/home/yuqingz/autonomous_driving/baseline/data/argoverse-tracking-v2/train4"  # directory for raw data
CFG_FILE="/home/yuqingz/autonomous_driving/baseline/OpenPCDet/tools/cfgs/kitti_models/pointrcnn_custom.yaml"  # model config file
CKPT_DIR="/home/yuqingz/autonomous_driving/baseline/OpenPCDet/checkpoints/pointrcnn_7870.pth"  # checkpoint to fine tune based on

CLEAN=false  # whether to clean data into required format
CLEAN_DATA_DIR="/home/yuqingz/autonomous_driving/baseline/data/ptrcnn_data"

TUNE=false  # whether to fine tune checkpoint
BATCH_SIZE=9  # batch_size (batch_size * GPU)
EPOCHS=80  # total epochs
TRAIN_EXP_NAME="exp"  # name of experiment / directory

DETECT_DIR="/home/yuqingz/autonomous_driving/baseline/results/ptrcnn_detect" # output directory of detection
DETECT_VIZ=false # whether to visualize the detection result

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
        ;;
        -detect_dir|--detect_dir)
        DETECT_DIR=$2
        ;;
        -viz|--viz)
        DETECT_VIZ=true
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
fi
echo "DETECT_DIR = $DETECT_DIR"


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
        --output $DETECT_DIR
    cd /home/yuqingz/autonomous_driving/baseline
fi


# run detection using model checkpoint
cd /home/yuqingz/autonomous_driving/baseline/OpenPCDet/tools

if [[ ! -e $DETECT_DIR/$TRAIN_EXP_NAME/detect ]]; then
    mkdir -p $DETECT_DIR/$TRAIN_EXP_NAME/detect
fi

if $TUNE; then
    CKPT_DIR=$DETECT_DIR/$TRAIN_EXP_NAME/ckpt/checkpoint_epoch_$EPOCHS.pth
    echo "Using checkpoint $CKPT_DIR"
fi

python demo_v2.py \
    --cfg_file $CFG_FILE \
    --data_path $CLEAN_DATA_DIR/train \
    --ckpt $CKPT_DIR \
    --ext ".npy" \
    --output $DETECT_DIR/$TRAIN_EXP_NAME/detect


# clean detection output format
cd /home/yuqingz/autonomous_driving/baseline/

if [[ ! -e $DETECT_DIR/$TRAIN_EXP_NAME/format ]]; then
    mkdir -p $DETECT_DIR/$TRAIN_EXP_NAME/format
fi

python ./clean_ptrcnn_res.py \
    -d $DETECT_DIR/$TRAIN_EXP_NAME/detect \
    -o $DETECT_DIR/$TRAIN_EXP_NAME/format \
    -data $RAW_DATA_DIR
    

# evaluate detection
python /home/yuqingz/autonomous_driving/baseline/argoverse-api/argoverse/evaluation/detection/eval_custom.py \
    -d $DETECT_DIR/$TRAIN_EXP_NAME/format \
    -g $RAW_DATA_DIR \
    -f /home/yuqingz/autonomous_driving/baseline/baseline/figures


# visualize detection
if $DETECT_VIZ; then
    if [[ ! -e $DETECT_DIR/$TRAIN_EXP_NAME/figures ]]; then
        mkdir -p $DETECT_DIR/$TRAIN_EXP_NAME/figures
    fi

    python /home/yuqingz/autonomous_driving/baseline/detect_visualize.py \
        -data $RAW_DATA_DIR \
        -d $DETECT_DIR/$TRAIN_EXP_NAME/format \
        -f $DETECT_DIR/$TRAIN_EXP_NAME/figures
fi



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
