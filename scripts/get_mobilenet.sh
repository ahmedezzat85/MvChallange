#!/bin/sh

Download_DIR=$PWD/../Code/model/mobilenet_v1

mkdir $Download_DIR
sudo chmod 777 $Download_DIR
cd $Download_DIR

wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz
wget http://download.tensorflow.org/models/mobilenet_v1_0.75_224_2017_06_14.tar.gz
wget http://download.tensorflow.org/models/mobilenet_v1_0.50_224_2017_06_14.tar.gz
wget http://download.tensorflow.org/models/mobilenet_v1_0.25_224_2017_06_14.tar.gz
wget http://download.tensorflow.org/models/mobilenet_v1_1.0_192_2017_06_14.tar.gz
wget http://download.tensorflow.org/models/mobilenet_v1_0.75_192_2017_06_14.tar.gz
wget http://download.tensorflow.org/models/mobilenet_v1_0.50_192_2017_06_14.tar.gz
wget http://download.tensorflow.org/models/mobilenet_v1_0.25_192_2017_06_14.tar.gz

tar -xzf mobilenet_v1_1.0_224_2017_06_14.tar.gz
tar -xzf mobilenet_v1_0.75_224_2017_06_14.tar.gz
tar -xzf mobilenet_v1_0.50_224_2017_06_14.tar.gz
tar -xzf mobilenet_v1_0.25_224_2017_06_14.tar.gz
tar -xzf mobilenet_v1_1.0_192_2017_06_14.tar.gz
tar -xzf mobilenet_v1_0.75_192_2017_06_14.tar.gz
tar -xzf mobilenet_v1_0.50_192_2017_06_14.tar.gz
tar -xzf mobilenet_v1_0.25_192_2017_06_14.tar.gz

cd $PWD