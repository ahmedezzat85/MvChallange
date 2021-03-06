#!/bin/sh

DATASET_DIR=$PWD/../dataset
EXEC_DIR=$PWD/../Code

cd $DATASET_DIR
wget "https://www.topcoder.com/contest/problem/IntelMovidius/training.tar"
wget "https://drive.google.com/uc?export=download&id=1He5BDXktQ2IXvqttmdy4I-pSzolYVPeG" -O train_patch.zip
tar -xf training.tar
unzip train_patch.zip
cp train_patch/* training/

cd $EXEC_DIR
python mv_dataset.py -w

cd $PWD