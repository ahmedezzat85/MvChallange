PYTHON=python3
IMG_SZ=128
NUM_EPOCH=1
BATCH_SZ=32
FP16=0
FMT=NHWC
model=mobilenet
MODEL_LOG_DIR=test-shuffle
LR=0.001
L2_REG=0.0001
DATA_AUG=1
OPT=adam
EPOCH=0

$PYTHON train.py --model-name $model --data-format $FMT --num-epoch $NUM_EPOCH \
		--batch-size $BATCH_SZ	--fp16 $FP16 --optimizer $OPT --lr $LR --wd $L2_REG \
		--log-subdir $MODEL_LOG_DIR --data-aug $DATA_AUG --begin-epoch $EPOCH --input-size $IMG_SZ