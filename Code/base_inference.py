import os
import cv2
import numpy as np
import pandas as pd
from image_preprocessing import preprocess_image

##=======##=======##=======##=======##
# GLOBAL VARIABLES & CONSTANTS
##=======##=======##=======##=======##
_DATA_ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')
_DATASETS = {
    'example': {
        'data_csv': 'training_ground_truth.csv',
        'data_dir': 'training'
    },
    'eval': {
        'data_csv': 'eval_set.csv', 
        'data_dir': 'training'
    },
    'train': {
        'data_csv': 'train_set.csv',
        'data_dir': 'training'
    },
    'test': {
        'data_csv': 'test_set.csv', 
        'data_dir': 'training'
    },
    'prov': {
        'data_csv': 'provisional.csv', 
        'data_dir': 'provisional'
    }
}

class BaseInference(object):
    """
    """
    def __init__(self, 
                 dataset_key,
                 inference_file,
                 score_inference,
                 input_size,
                 preserve_aspect):
        
        if dataset_key not in _DATASETS:
            raise ValueError('Unknown dataset %s', dataset_type)
        
        dataset = _DATASETS[dataset_key]

        self.summary         = []
        self.img_size        = input_size
        self.images_dir      = os.path.join(_DATA_ROOT_DIR, dataset['data_dir'])
        self.score_flag      = score_inference
        self.resize_side     = 256
        self.out_csv_file    = inference_file
        self.csv_data_file   = os.path.join(_DATA_ROOT_DIR, dataset['data_csv'])
        self.preserve_aspect = preserve_aspect

    def _preprocess(self, image):
        """ """
        return preprocess_image(image, self.resize_side, self.img_size, self.preserve_aspect)
    
    def forward(self, image):
        raise NotImplementedError()

    def __call__(self):
        df = pd.read_csv(self.csv_data_file, sep=',')
        images = df['IMAGE_NAME']
        if self.score_flag is True: labels = df['LABEL_INDEX']
        score = InferenceScore()

        for i, image in enumerate(images):
            # Read and decode image
            image_path = os.path.join(self.images_dir, image)
            im = cv2.imread(image_path)

            # Perform Preprocessing
            im = self._preprocess(im)

            # Run the classification model
            probs, infer_time = self.forward(im)

            # Get top-5 predicted classes
            top5  = np.argsort(-probs)[:5]
            top5_classes = top5 + 1 # Labels are stored starting from 1
            top5_probs   = probs[top5]
            
            # Format Message
            row = (image, )
            for cls_id, prob in zip(top5_classes, top5_probs): row += (cls_id, prob, )
            row += (infer_time,)
            self.summary.append(row)

            if self.score_flag is True:
                cls_id = labels[i]
                score.update(cls_id, probs[cls_id - 1], top5_classes, top5_probs, infer_time)

        # Write Inferences to CSV file
        header = ['IMAGE_NAME']
        for i in range(1,6): header += ['LABEL_INDEX #{:d}'.format(i), 'PROBABILITY #{:d}'.format(i)]
        header += ['INFERENCE_TIME']
        df = pd.DataFrame(self.summary, columns=header)
        df.to_csv(self.out_csv_file, index=False, header=False)
        
        # Dump Score
        if self.score_flag is True:
            score.finish()

    def close(self):
        pass


class InferenceScore(object):
    """
    """
    def __init__(self):
        self.probs        = []
        self.count        = 0
        self.min_prob     = np.float(1e-15)
        self.max_prob     = 1.0 - self.min_prob
        self.top1_acc     = 0
        self.top5_acc     = 0
        self.image_time   = 0

    def update(self, cls_id, cls_prob, top5_classes, top5_probs, infer_time):
        """ """
        prob = 0
        self.image_time += infer_time
        if cls_id == top5_classes[0]: self.top1_acc += 1
        if cls_id in top5_classes: 
            self.top5_acc += 1
            prob = cls_prob / np.sum(top5_probs)
        prob = max(self.min_prob, min(self.max_prob, prob))
        self.probs.append(prob)

        self.count += 1
        if self.count % 100 == 0:
            print('(',self.count,')', (self.top1_acc * 100.0) / self.count, 
                                      (self.top5_acc * 100.0) / self.count)

    def finish(self, mult=100, n_classes=200, log_loss_max=15.0, time_limit=1000.0):
        """ """
        probs = np.array(self.probs)
        log_loss = np.mean(-np.log(probs))
        self.image_time /= self.count
        t = mult * self.image_time
        if self.image_time > time_limit or log_loss > log_loss_max:
            score = 0.0
        else:
            t_max = mult * time_limit
            score = 1e6 * (1.0 - log_loss * np.log(t) / (log_loss_max *  np.log(t_max)))

        print ('TOP-1 Accuracy  = ', self.top1_acc, self.top1_acc * 100 / self.count)
        print ('TOP-5 Accuracy  = ', self.top5_acc, self.top5_acc * 100 / self.count)
        print ('Inference Time  = ', self.image_time)
        print ('Log Loss        = %.9f' % log_loss)
        print ('Inference Score = %.2f' % score)
        return score