"""
"""
import os
from datetime import datetime

import utils
from tf_dnn import TFClassifier
from mv_dataset import TFDatasetReader
from arg_parser import parse_cmd_line_options

_CUR_DIR       = os.path.dirname(__file__)
_LOGS_ROOT_DIR = os.path.join(_CUR_DIR, '..', 'logs')
_MODEL_BIN_DIR = os.path.join(_CUR_DIR, '..', 'bin')

class DeepNN(object):
    """
    """
    def __init__(self, args):
        self.flags    = args
        self.logs_dir = os.path.join(_LOGS_ROOT_DIR, args.model_name, args.log_subdir)
        self.bin_dir  = os.path.join(_MODEL_BIN_DIR, args.model_name, args.log_subdir)
        self.dataset  = TFDatasetReader(image_size=args.input_size)
        self.module   = TFClassifier(args.model_name, args.data_format, self.logs_dir)

    def train(self):
        begin_epoch = self.flags.begin_epoch
        log_file    = os.path.join(self.logs_dir, self.flags.model_name+'_'+str(begin_epoch)+'.log')
        self.logger = utils.create_logger(self.flags.model_name, log_file)

        # Pack the hyper-parameters
        hp_dict = {'batch_size': self.flags.batch_size, 'optimizer': self.flags.optimizer,
                    'lr': self.flags.lr, 'wd': self.flags.wd, 'data_aug': self.flags.data_aug}
        hp = utils.DictToAttrs(hp_dict)

        t_start = datetime.now()
        self.logger.info("Training Started at  : " + t_start.strftime("%Y-%m-%d %H:%M:%S"))

        if self.flags.lr_decay:
            num_iter = self.flags.num_epoch // self.flags.lr_step 
            for i in range(num_iter):
                self.module.train(self.dataset, hp, self.flags.lr_step, begin_epoch, logger=self.logger)
                begin_epoch += self.flags.lr_step
                hp.lr *= self.flags.lr_decay

                log_file    = os.path.join(self.logs_dir, self.flags.model_name+'_'+str(begin_epoch)+'.log')
                for h in self.logger.handlers[:]: self.logger.removeHandler(h)
                self.logger = utils.create_logger(self.flags.model_name, log_file)
        else:
            self.module.train(self.dataset, hp, self.flags.num_epoch, begin_epoch, logger=self.logger)

        self.logger.info("Training Finished at : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("Total Training Time  : " + str(datetime.now() - t_start))

    def deploy(self):
        """ """
        self.module.deploy(self.bin_dir, self.flags.input_size, self.dataset.num_classes)

def main():
    """ """
    args = parse_cmd_line_options()
    cmd = args.subcmd.lower()

    dnn = DeepNN(args)
    if cmd == 'train':
        dnn.train()
    elif cmd == 'deploy':
        dnn.deploy()
    else:
        raise ValueError('Unknown sub-command %s', cmd)

if __name__ == '__main__':
    main()