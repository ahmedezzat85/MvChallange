import os
import cv2
import argparse
import numpy as np
from mvnc import mvncapi as mvnc

from base_inference import BaseInference


class MvNCSInference(BaseInference):
    """
    """
    def __init__(self, 
                 dataset_key='prov',
                 inference_file='inferences.csv',
                 score_inference=False,
                 model='compiled.graph',
                 input_size=224,
                 preserve_aspect=True):
        super(MvNCSInference, self).__init__(dataset_key, inference_file, score_inference, 
                                                input_size, preserve_aspect)
        self._open()
        self._load_model(model)

    def _open(self, dev_idx=0):
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

    def _load_model(self, graph_file):
        with open(graph_file, mode='rb') as f:
            blob = f.read()

        self.ncs_graph = self.dev.AllocateGraph(blob)
        self.ncs_graph.SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)

    def forward(self, input_tensor):
        self.ncs_graph.LoadTensor(input_tensor, None)
        net_out, _ = self.ncs_graph.GetResult()
        ncs_time = np.sum(self.ncs_graph.GetGraphOption(mvnc.GraphOption.TIME_TAKEN)) 
        return net_out, ncs_time

    def close(self):
        self.ncs_graph.DeallocateGraph()
        self.dev.CloseDevice()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MvNCS Inference Script')
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-s', '--score'  , action='store_true')
    args = parser.parse_args()

    infer = MvNCSInference(dataset_key=args.dataset, score_inference=args.score)
    infer()
    infer.close()
