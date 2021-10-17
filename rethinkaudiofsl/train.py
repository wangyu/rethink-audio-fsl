from __future__ import print_function
import argparse
import os
import imp
from sklearn.metrics import average_precision_score
import algorithms as alg
from dataloader import FSD_MIX_CLIPS, FewShotDataloader

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, default='', help='config file with parameters of the experiment. '
                                                                          'It is assumed that the config file is placed in ./config/.')
parser.add_argument('--checkpoint', type=int, default=0, help='checkpoint (epoch id) that will be loaded. '
                                                              'If a negative value is given then the latest existing checkpoint is loaded.')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--disp_step', type=int, default=200, help='display step during training')
parser.add_argument('--inlen', type=int, default=10, help='input length (sec)')
parser.add_argument('--openl3', action='store_true', help='use reduced classes')
parser.add_argument('--poly', type=str, default='1', help='polyphony of support examples')

args = parser.parse_args()

exp_config_file = os.path.join('.','config', args.config + '.py')
exp_directory = os.path.join('.','experiments', args.config)

# Load the configuration params of the experiment
print('Launching experiment: %s' % exp_config_file)
config = imp.load_source("",exp_config_file).config
config['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored
print("Loading experiment %s from file: %s" % (args.config, exp_config_file))
print("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))

# Set train and test datasets and the corresponding data loaders
data_train_opt = config['data_train_opt']
data_test_opt = config['data_test_opt']

train_split, test_split = 'train', 'val'
dataset_train = FSD_MIX_CLIPS(phase=train_split, inlen=args.inlen, openl3=args.openl3)
dataset_test = FSD_MIX_CLIPS(phase=test_split, inlen=args.inlen, openl3=args.openl3)

dloader_train = FewShotDataloader(
    dataset=dataset_train,
    nKnovel=data_train_opt['nKnovel'],
    nKbase=data_train_opt['nKbase'],
    nExemplars=data_train_opt['nExemplars'], # num training examples per novel category
    nTestNovel=data_train_opt['nTestNovel'], # num test examples for all the novel categories
    nTestBase=data_train_opt['nTestBase'], # num test examples for all the base categories
    batch_size=data_train_opt['batch_size'],
    num_workers=args.num_workers,
    epoch_size=data_train_opt['epoch_size'], # num of batches per epoch
    poly=args.poly
)

dloader_test = FewShotDataloader(
    dataset=dataset_test,
    nKnovel=data_test_opt['nKnovel'],
    nKbase=data_test_opt['nKbase'],
    nExemplars=data_test_opt['nExemplars'], # num training examples per novel category
    nTestNovel=data_test_opt['nTestNovel'], # num test examples for all the novel categories
    nTestBase=data_test_opt['nTestBase'], # num test examples for all the base categories
    batch_size=data_test_opt['batch_size'],
    num_workers=0,
    epoch_size=data_test_opt['epoch_size'], # num of batches per epoch
    poly=args.poly
)

config['disp_step'] = args.disp_step
algorithm = alg.FewShot(config)
if args.cuda: # enable cuda
    algorithm.load_to_gpu()

if args.checkpoint != 0: # load checkpoint
    algorithm.load_checkpoint(
        epoch=args.checkpoint if (args.checkpoint > 0) else '*',
        train=True)

# train the algorithm
algorithm.solve(dloader_train, dloader_test)