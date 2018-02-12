SUPPORTING FILES
----------------
-- supporting/base_inference.py
        Inference engine.
-- supporting/mvncs_inference.py
        Run inference on the Movidius NCS.
-- supporting/eval_set.csv
        Validation dataset (taken from the training set)
-- supporting/image_preprocessing.py
        Image preprocessing functions.


IMAGE PREPROCESSING
-------------------
* Convert the image from BGR to RGB.
* Normalize image (divide by 255).
* Image is resized such that the short side is '256'. The resize operation preserves the aspect ratio.
* CENTER CROP the resized image to (224, 224)
* Subtract RGB Mean
see supporting/image_preprocessing.py --> preprocess_image for implementation details.

DEPENDENCIES
------------
* python 3.5 or later.
* 'pandas' package (for easy handling of CSV).

RUN INFERENCE TEST
------------------
Main script <supporting/mvncs_inference.py>

python3 mvncs_inference.py [-d DATASET] [-s]

  -d DATASET, --dataset DATASET  
                    Dataset Key for defining target set for evaluation. 
                    Possible values now are {'eval', 'prov'}.
                    - 'eval': A subset division of the training set for validation.
                    - 'prov': The provisional dataset.
  -s, --score 
            If passed, the scoring function will run to record the score (not meaningful for 'prov')

  Typical Usage:
        python3 mvncs_inference.py -d eval -s
        python3 mvncs_inference.py -d prov

NOTES
-----
For configuring the paths of the dataset images and csv files, see Both '_DATA_ROOT_DIR' and '_DATASETS' global
variables at the start of the <base_inference.py> file.