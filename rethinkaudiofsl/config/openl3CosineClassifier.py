config = {}
# set the parameters related to the training and testing set

nKbase = 59

data_train_opt = {}
data_train_opt['nKnovel'] = 0
data_train_opt['nKbase'] = nKbase
data_train_opt['nExemplars'] = 0
data_train_opt['nTestNovel'] = 0
data_train_opt['nTestBase'] = 32
data_train_opt['batch_size'] = 2
data_train_opt['epoch_size'] = data_train_opt['batch_size'] * 1000

data_test_opt = {}
data_test_opt['nKnovel'] = 5
data_test_opt['nKbase'] = nKbase
data_test_opt['nExemplars'] = 1
data_test_opt['nTestNovel'] = 8 * data_test_opt['nKnovel']  # min n_example in novel classes = 10
data_test_opt['nTestBase'] = 8 * data_test_opt['nKnovel']
data_test_opt['batch_size'] = 1
data_test_opt['epoch_size'] = 500

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt

config['max_num_epochs'] = 60

networks = {}
net_optionsF = {'userelu': True, 'usebn':True}
net_optim_paramsF = {'optim_type': 'adam', 'lr': 0.001, 'beta':(0.9, 0.999), 'amsgrad':True}
networks['feat_model'] = {'def_file': 'architectures/dense.py', 'pretrained': None, 'opt': net_optionsF, 'optim_params': net_optim_paramsF}

net_optim_paramsC = {'optim_type': 'adam', 'lr': 0.001, 'beta':(0.9, 0.999), 'amsgrad':True}
net_optionsC = {'classifier_type': 'cosine', 'weight_generator_type': 'none', 'nKall': nKbase, 'nFeat':2048, 'scale_cls': 10}
networks['classifier'] = {'def_file': 'architectures/ClassifierWithFewShotGenerationModule.py', 'pretrained': None, 'opt': net_optionsC, 'optim_params': net_optim_paramsC}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'BCEWithLogitsLoss', 'opt':None}  # BCEWithLogitsLoss = sigmoid + BCELoss, use this for multilabel classification case
config['criterions'] = criterions

config['openl3'] = True

config['algorithm_type'] = 'FewShot'
