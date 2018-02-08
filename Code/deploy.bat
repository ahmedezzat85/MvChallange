@echo off

set PYTHON=python3
set IMG_SZ=%2
set FMT=NHWC
set model=%1
set MODEL_LOG_DIR=mobilenet
set CHKPT=%3

%PYTHON% deep_nn.py deploy --model-name %model% --data-format %FMT% --log-subdir %MODEL_LOG_DIR% ^
		--input-size %IMG_SZ% --checkpoint %CHKPT%