config = {}
# set the parameters related to the training and testing set

nKbase = 59
nKnovel = 5
nExemplars = 5

data_train_opt = {}
data_train_opt['nKnovel'] = nKnovel
data_train_opt['nKbase'] = -1
data_train_opt['nExemplars'] = nExemplars
data_train_opt['nTestNovel'] = nKnovel * 5
data_train_opt['nTestBase'] = nKnovel * 5
data_train_opt['batch_size'] = 2
data_train_opt['epoch_size'] = data_train_opt['batch_size'] * 1000

data_test_opt = {}
data_test_opt['nKnovel'] = nKnovel
data_test_opt['nKbase'] = nKbase
data_test_opt['nExemplars'] = nExemplars
data_test_opt['nTestNovel'] = 15 * data_test_opt['nKnovel']
data_test_opt['nTestBase'] = 15 * data_test_opt['nKnovel']
data_test_opt['batch_size'] = 1
data_test_opt['epoch_size'] = 500

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt

config['max_num_epochs'] = 60

networks = {}
net_optionsF = {'userelu': True, 'usebn':True}
pretrainedF = './experiments/fsd_openl3CosineClassifier/feat_model_net_epoch*.best'
networks['feat_model'] = {'def_file': 'architectures/dense.py', 'pretrained': pretrainedF, 'opt': net_optionsF, 'optim_params': None}

net_optim_paramsC = {'optim_type': 'adam', 'lr': 0.001, 'beta':(0.9, 0.999), 'amsgrad':True}
pretrainedC = './experiments/fsd_openl3CosineClassifier/classifier_net_epoch*.best'
net_optionsC = {'classifier_type': 'cosine', 'weight_generator_type': 'attention_based', 'nKall': nKbase, 'nFeat':2048, 'scale_cls': 10, 'scale_att': 10.0}
networks['classifier'] = {'def_file': 'architectures/ClassifierWithFewShotGenerationModule.py', 'pretrained': pretrainedC, 'opt': net_optionsC, 'optim_params': net_optim_paramsC}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'BCEWithLogitsLoss', 'opt':None}
config['criterions'] = criterions

config['openl3'] = True

config['algorithm_type'] = 'FewShot'