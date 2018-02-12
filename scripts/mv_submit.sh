#!/bin/sh

MODEL=$1
IN_SZ=$2
DATASET_KEY=$3
MODEL_DIR=../bin/$MODEL
MODEL_FILE=$MODEL_DIR/network.meta
OUT_DIR=$PWD/../submission
COMPILED_GRAPH=compiled.graph

# TODO: Check inputs
# TODO: password automation

# Create directory for submission
sudo rm $COMPILED_GRAPH
sudo rm $OUT_DIR -R
mkdir $OUT_DIR
sudo chmod 777 $OUT_DIR -R
cd $OUT_DIR

# Copy executable scripts
cp ../Code/mvncs_inference.py .
cp ../Code/base_inference.py .
cp ../Code/image_preprocessing.py .

# Generate NCS Compiled Model
echo 'Generating Graph For Model for "'$MODEL  $IN_SZ'"'
cp $MODEL_DIR/network.* .
mvNCCompile network.meta -w network -s 12 -in input -on output -o $COMPILED_GRAPH -is $IN_SZ $IN_SZ

# Run inference
python3 mvncs_inference.py -d $DATASET_KEY $4

mkdir supporting
sudo chmod 777 supporting -R
mv mvncs_inference.py supporting/
mv base_inference.py supporting/
mv image_preprocessing.py supporting/
cp ../scripts/README supporting/
cp ../dataset/eval_set.csv supporting/

cd $PWD
echo 'DONE...'