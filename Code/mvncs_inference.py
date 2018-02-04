import os
import cv2
import numpy as np
import pandas as pd

from mvnc import mvncapi as mvnc

##=======##=======##=======##=======##
# GLOBAL VARIABLES & CONSTANTS
##=======##=======##=======##=======##
_DATA_ROOT_DIR = 'data'
_DATA_FILES = {
    'eval': 'eval_set.csv', 
    'prov': 'provisional.csv'}

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_RGB_MEAN = np.array([_R_MEAN, _G_MEAN, _B_MEAN]) / 255


##=======##=======##=======##=======##
# MvNCS DEVICE ABSTRACTION
##=======##=======##=======##=======##
class MvNCS(object):
    """ A simple wrapper for the Movidius NCS device.
    """
    def __init__(self, dev_idx=0):
        # configure the NCS
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 1)

        # Get a list of ALL the sticks that are plugged in
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0: raise ValueError('No devices found')

        # Pick the first stick to run the network
        self.dev = mvnc.Device(devices[dev_idx])

        # Open the NCS
        try:
            self.dev.OpenDevice()
        except:
            raise ValueError('Cannot Open NCS Device')

    def load_model(self, graph_file):
        with open(graph_file, mode='rb') as f:
            blob = f.read()

        self.ncs_graph = self.dev.AllocateGraph(blob)
        self.ncs_graph.SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)

    def forward(self, input_tensor):
        self.ncs_graph.LoadTensor(input_tensor, None)
        net_out, _ = self.ncs_graph.GetResult()
        ncs_time = np.sum(self.ncs_graph.GetGraphOption(mvnc.GraphOption.TIME_TAKEN)) 
        return net_out, ncs_time

    def unload_model(self):
        self.ncs_graph.DeallocateGraph()

    def close(self):
        self.dev.CloseDevice()
     
##=======##=======##=======##=======##
# INFERENCE APIs
##=======##=======##=======##=======##
class MvNCSInference(object):
    """
    """
    def __init__(self, 
                 dataset_type='prov',
                 score_inference=False,
                 model='compiled.graph',
                 input_size=224,
                 preserve_aspect=True):
        
        if dataset_type not in _DATA_FILES:
            raise ValueError('Unknown dataset %s', dataset_type)

        self.summary         = []
        self.img_size        = input_size
        self.images_dir      = os.path.join(_DATA_ROOT_DIR, dataset_type)
        self.score_flag      = score_inference
        self.resize_side     = 256
        self.csv_data_file   = os.path.join(_DATA_ROOT_DIR, _DATA_FILES[dataset_type])
        self.preserve_aspect = preserve_aspect

        self.mvncs = MvNCS()
        self.mvncs.load_model(model)

    def _resize_keep_aspect(self, image):
        """ Perform Resize for the input image while preserving 
        the aspect ratio. The short edge of the image is set to 
        be equal to r'self.rescale_size' and the other dimension
        is scaled accordingly to keep the aspect ratio as is.  
        """
        # Resize
        h, w, _ = image.shape
        scale = self.resize_side / min(h, w)
        h = int(h * scale)
        w = int(w * scale)
        image = cv2.resize(image, (w, h))
        
        # Center Crop to the target size 
        hs = (h - self.img_size) // 2
        ws = (w - self.img_size) // 2
        image = image[hs:hs+self.img_size, ws:ws+self.img_size]
        return image

    def _preprocess(self, image):
        """ """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = image / 255

        if self.preserve_aspect is True:
            image = self._resize_keep_aspect(image)
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))

        image = image - _RGB_MEAN
        return np.expand_dims(image, 0)

    def __call__(self):
        df = pd.read_csv(self.csv_data_file, sep=',')
        images = df['IMAGE_NAME']
        if self.score_flag is True: labels = df['CLASS_INDEX']
        score = InferenceScore(len(images))

        for i, image in enumerate(images):
            # Read and decode image
            image_path = os.path.join(self.images_dir, image)
            im = cv2.imread(image_path)

            # Perform Preprocessing
            im = self._preprocess(im)

            # Run the classification model
            probs, infer_time = self.mvncs.forward(im)

            # Get top-5 predicted classes
            top5  = np.argsort(-probs)[:5]
            top5_classes = top5 + 1 # Labels are stored starting from 1
            top5_probs   = probs[top5]
            
            # Format Message
            row = (image, )
            for cls_id, prob in zip(top5_classes, top5_probs): row += (cls_id, prob, )
            row += (infer_time)
            self.summary.append(row)

            if self.score_flag is True:
                cls_id = labels[i]
                score.update(cls_id, probs[cls_id - 1], top5_classes, top5_probs, infer_time)
        
        header = ['IMAGE_NAME']
        for i in range(1,6): header += ['LABEL_INDEX #{:d}'.format(i), 'PROBABILITY #{:d}'.format(i)]
        header += ['INFERENCE_TIME']
        df = pd.DataFrame(self.summary, columns=header)
        if self.score_flag is True:
            score.finish()

    def close():
        self.mvncs.close()

class InferenceScore(object):
    """
    """
    def __init__(self, n_images):
        self.probs        = np.zeros(n_images)
        self.count        = 0
        self.min_prob     = np.float(1e-15)
        self.max_prob     = 1.0 - self.min_proba
        self.top1_acc     = 0
        self.top5_acc     = 0
        self.image_time   = 0

    def update(self, cls_id, cls_prob, top5_classes, top5_probs, infer_time):
        """ """
        self.probs[self.count] = 0
        self.image_time += infer_time
        if cls_id == top5_classes[0]: self.top1_acc += 1
        if cls_id in top5_classes: 
            self.top5_acc += 1
            cls_prob /= np.sum(top5_probs)
            cls_prob = max(self.min_prob, min(self.max_prob, cls_prob))
            self.probs[self.count] = cls_prob
        self.count += 1
        if self.count % 100 == 0:
            print('(',self.count,')', (self.top1_acc * 100.0) / self.count, 
                                      (self.top5_acc * 100.0) / self.count)

    def finish(self, mult=100, n_classes=200, log_loss_max=7.0, time_limit=1000.0):
        """ """
        log_loss = np.mean(-np.log(self.probs))
        self.top1_acc /= self.count
        self.top5_acc /= self.count
        self.image_time /= self.count
        t = mult * self.image_time
        if self.image_time > time_limit or log_loss > log_loss_max:
            score = 0.0
        else:
            t_max = mult * time_limit
            score = 1e6 * (1.0 - log_loss * np.log(t) / (log_loss_max *  np.log(t_max)))

        print ('TOP-1 Accuracy  = ', self.top1_acc, self.top1_acc * 100 / self.count)
        print ('TOP-5 Accuracy  = ', self.top5_acc, self.top5_acc * 100 / self.count)
        print ('Inference Time  = ', self.image_time / self.count)
        print ('Log Loss        = %.9f' % log_loss)
        print ('Image Time      = %.9f' % image_time)
        print ('Inference Score = %.2f' % score)
        return score

if __name__ == '__main__':
    infer = TensorflowInference('mobilenet')
    infer.run()
    infer.close()
