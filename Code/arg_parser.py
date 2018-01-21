""" Synaplexus Trainer Script 
"""
import os
import argparse

def parse_cmd_line_options():
    """
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model-name'  , type=str      , help='Neural Network Model name. e.g. VGG.')
    arg_parser.add_argument('--lr'          , type=float    , help='Learning Rate.')
    arg_parser.add_argument('--wd'          , type=float    , help='L2 Regularization Parameter or Weight Decay.')
    arg_parser.add_argument('--optimizer'   , type=str      , help='Optimization technique for parameter update.')
    arg_parser.add_argument('--fp16'        , type=int      , help='Whether to use fp16 for the entire model parameters. Default is fp32.')
    arg_parser.add_argument('--data-aug'    , type=int      , help='Use data-augmentation to extend the dataset.')
    arg_parser.add_argument('--batch-size'  , type=int      , help='Training mini-batch size.')
    arg_parser.add_argument('--lr-step'     , type=int      , help='Learning rate decay step in epochs.')
    arg_parser.add_argument('--lr-decay'    , type=int      , help='Learning rate decay rate.')
    arg_parser.add_argument('--data-format' , type=str      , help='Data Format')
    arg_parser.add_argument('--num-epoch'   , type=int      , help='Number of epochs for the training process.')
    arg_parser.add_argument('--begin-epoch' , type=int      , help='Epoch ID of from which the training process will start. Useful for training resume.')
    arg_parser.add_argument('--log-subdir'  , type=str      , help='Logs Directory')
    arg_parser.add_argument('--input-size'  , type=int      , help='Input Image Size.')

    arg_parser.set_defaults(
        model_name  = 'mlp',
        data_format = 'NCHW',
        lr          = 1e-3,
        wd          = 0,
        optimizer   = 'Adam',
        fp16        = 0,
        data_aug    = 0,
        batch_size  = 128,
        lr_step     = 10000,
        lr_decay    = 0,
        begin_epoch = 0,
        num_epoch   = 1
    )

    args = arg_parser.parse_args()
    return args