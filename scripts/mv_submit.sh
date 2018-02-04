#!/bin/sh

CWD=$PWD
MODEL=$1
INPUT_SIZE=$2
DATASET_KEY=$3
OUT_DIR=$PWD/../submission
# TODO: Check inputs
# TODO: password automation

# Create directory for submission
sudo rm $OUT_DIR -R
mkdir $OUT_DIR
sudo chmod 777 $OUT_DIR -R

# Generate NCS Compiled Model
./ncs_graph_gen.sh $MODEL $INPUT_SIZE
mv compiled.graph $OUT_DIR

cd $OUT_DIR
cp ../Code/mvncs_inference.py .
python3 mvncs_inference.py -d $DATASET_KEY -s

cd $PWD
echo 'DONE...'