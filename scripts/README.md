## REPOSITORY

Complete Code is under https://github.com/ahmedezzat85/MvChallange.git

## PREREQUISITES

* python 3.5 or later.
* 'pandas' package (for easy handling of CSV).
* tensorflow 1.4 or higher (only for training).

## SUPPORTING FILES

-- supporting/base_inference.py
        Inference engine.
-- supporting/mvncs_inference.py
        Run inference on the Movidius NCS.
-- supporting/eval_set.csv
        Validation dataset (taken from the training set)
-- supporting/image_preprocessing.py
        Image preprocessing functions.

## TRAINING METHODOLOGY

### Model

MobileNetV1_1.0_224 pretrained on imagenet and finetuned on the IntelMovidius-200 Dataset.

### Training 

The dataset is splitted to 76000/4000 train/val split. The jpeg images are packed into tfrecords files.
To generate the tfrecord files, run `python mv_dataset.py -w`
The global variable `DATASET_DIR` needs to be configured for dataset root directory. It includes all the
csv files in addition to the training directory containing the 80000 training images.

### Training Hyperparameters

* Using Dropout 0.5
* Weight Decay 0.00001
* ADAM Optimizer
* Lerning rate = 1e-4
* Learning rate decay 0.98 every 1 Epoch

### Preprocessing

`preprocess_image` in file `vgg_preprocessing.py` implements the image preprocessing for training , which are:
* Scaling of the short side of the input image randomly in {256, 640}.
* Normalize the pixel values to be between {0,1}
* Resize the image preserving the aspect ratio.
* Random crop of size 224x224.
* Random flip the image horizontally.
* Random brightness.
* RGB Mean subtraction.

## NCS GRAPH GENERATION

After training, the tensorflow checkpoint is loaded and re-saved after removing the training specific nodes (following NCSDK recommendation, see `tf_dnn.py --> TFClassifier.deploy`)

`mvNCCompile network.meta -w network -s 12 -in input -on output -o compiled.graph -is 224 224`

## INFERENCE

### Preprocessing

* Convert the image from BGR to RGB.
* Normalize image (divide by 255).
* Image is resized such that the short side is '256'. The resize operation preserves the aspect ratio.
* CENTER CROP the resized image to (224, 224)
* Subtract RGB Mean
see supporting/image_preprocessing.py --> preprocess_image for implementation details.

### Run Inference

Main script <supporting/mvncs_inference.py>

python3 mvncs_inference.py [-d DATASET] [-s]

  -d DATASET, --dataset DATASET  
                    Dataset Key for defining target set for evaluation. 
                    Possible values now ar {'example', 'prov', 'eval'}.
                    - 'eval': A subset division of the training set for validation.
                    - 'prov': The provisional dataset.
                    Complete set is defined in base_inference.py
  -s, --score 
            If passed, the scoring function will run to record the score (not meaningful for 'prov')

  Typical Usage:
        python3 mvncs_inference.py -d eval -s
        python3 mvncs_inference.py -d prov

## NOTES

For configuring the paths of the dataset images and csv files, see Both '_DATA_ROOT_DIR' and '_DATASETS' global
variables at the start of the <base_inference.py> file.