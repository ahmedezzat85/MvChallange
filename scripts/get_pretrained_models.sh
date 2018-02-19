#!/bin/sh

Download_DIR=$PWD/../pretrained/

mkdir $Download_DIR
sudo chmod 777 $Download_DIR
cd $Download_DIR

wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz
wget http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz

mkdir mobilenet_v1
mv mobilenet_v1_1.0_224_2017_06_14.tar.gz mobilenet_v1
cd mobilenet_v1
tar -xzf mobilenet_v1_1.0_224_2017_06_14.tar.gz

mkdir inception_v2
mv inception_v2_2016_08_28.tar.gz inception_v2
cd ../inception_v2
tar -xzf inception_v2_2016_08_28.tar.gz

# wget http://download.tensorflow.org/models/mobilenet_v1_0.75_224_2017_06_14.tar.gz
# wget http://download.tensorflow.org/models/mobilenet_v1_0.50_224_2017_06_14.tar.gz
# wget http://download.tensorflow.org/models/mobilenet_v1_0.25_224_2017_06_14.tar.gz
# wget http://download.tensorflow.org/models/mobilenet_v1_1.0_192_2017_06_14.tar.gz
# wget http://download.tensorflow.org/models/mobilenet_v1_0.75_192_2017_06_14.tar.gz
# wget http://download.tensorflow.org/models/mobilenet_v1_0.50_192_2017_06_14.tar.gz
# wget http://download.tensorflow.org/models/mobilenet_v1_0.25_192_2017_06_14.tar.gz
# tar -xzf mobilenet_v1_0.75_224_2017_06_14.tar.gz
# tar -xzf mobilenet_v1_0.50_224_2017_06_14.tar.gz
# tar -xzf mobilenet_v1_0.25_224_2017_06_14.tar.gz
# tar -xzf mobilenet_v1_1.0_192_2017_06_14.tar.gz
# tar -xzf mobilenet_v1_0.75_192_2017_06_14.tar.gz
# tar -xzf mobilenet_v1_0.50_192_2017_06_14.tar.gz
# tar -xzf mobilenet_v1_0.25_192_2017_06_14.tar.gz

cd $PWD