"""
"""
import os
from mv_dataset import TFDatasetReader
from tf_classifier import TFClassifier
from arg_parser import parse_cmd_line_options
import utils

_CUR_DIR       = os.path.dirname(__file__)
_LOGS_ROOT_DIR = os.path.join(_CUR_DIR, '..', 'logs')
_MODEL_BIN_DIR = os.path.join(_CUR_DIR, '..', 'bin')

def train():
    args = parse_cmd_line_options()
    logs_dir = os.path.join(_LOGS_ROOT_DIR, args.model_name, args.log_subdir)
    bin_dir  = os.path.join(_MODEL_BIN_DIR, args.model_name, args.log_subdir)

    # Load Dataset
    dataset = TFDatasetReader(image_size=args.input_size)

    # Invoke trainer
    tf_module = TFClassifier(args.model_name, dataset, args.data_format, args.data_aug, logs_dir, bin_dir)
    # Pack the hyper-parameters
    hp_dict = {'batch_size': args.batch_size, 'optimizer': args.optimizer, 'lr': args.lr, 'wd': args.wd,
                'lr_decay': args.lr_decay, 'lr_step': args.lr_step}
    hp = utils.DictToAttrs(hp_dict)
    tf_module.train(args.num_epoch, args.begin_epoch, hp)


if __name__ == '__main__':
    train()