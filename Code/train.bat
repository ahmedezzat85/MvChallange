@echo off

set PYTHON=python3
set IMG_SZ=192
set NUM_EPOCH=1
set BATCH_SZ=16
set FP16=0
set FMT=NHWC
set model=mobilenet
set MODEL_LOG_DIR=test
set LR=0.001
set L2_REG=0.0
set DATA_AUG=1
set OPT=adam
set EPOCH=0

%PYTHON% deep_nn.py train --model-name %model% --data-format %FMT% --num-epoch %NUM_EPOCH% ^
		--batch-size %BATCH_SZ%	--fp16 %FP16% --optimizer %OPT% --lr %LR% --wd %L2_REG% ^
		--log-subdir %MODEL_LOG_DIR% --data-aug %DATA_AUG% --begin-epoch %EPOCH% --input-size %IMG_SZ%