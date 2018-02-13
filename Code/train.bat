@echo off

set PYTHON=python3
set IMG_SZ=224
set NUM_EPOCH=1
set BATCH_SZ=8
set FP16=0
set FMT=NHWC
set model=mobilenet
set MODEL_LOG_DIR=mobilenet
set MODEL_PARAM="{\"model\": \"FREEZE_ALL_1.0\"}"
set LR=0.001
set L2_REG=0.0
set DATA_AUG=1
set OPT=adam
set EPOCH=0
set DECAY=0.0
set DECAY_STEP=5

%PYTHON% deep_nn.py train --model-name %model%  --model-param %MODEL_PARAM% --data-format %FMT% ^
		--num-epoch %NUM_EPOCH% --batch-size %BATCH_SZ%	--fp16 %FP16% --optimizer %OPT% --lr %LR% ^
		--wd %L2_REG% --log-subdir %MODEL_LOG_DIR% --data-aug %DATA_AUG% --begin-epoch %EPOCH% ^
		--input-size %IMG_SZ% --lr-decay %DECAY% --lr-step %DECAY_STEP%