import os
import h5py
import argparse
import imp
from sklearn.metrics import average_precision_score  # import before torch to avoid error

from dataloader import SimpleDataset, SimpleDataManager
import algorithms as alg

def save_features(model, data_loader, outfile):
    # initialize h5 file
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = None
    all_feats = None
    count = 0

    # dataloader returns batches of x:logmel, y:multihot label vector with shape (140,)
    # compute and save features for each batch
    for i, (x, y) in enumerate(data_loader):
        if i % 10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        feats = model(x)

        # initialize feature and label datasets in h5 file,
        # with shape (max_count, feat_size) and (max_count, label_vector_size)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
            all_labels = f.create_dataset('all_labels', [max_count] + list(y.size()[1:]), dtype='f')

        # write batch features and corresponding labels to h5 file
        all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count + feats.size(0)] = y.data.cpu().numpy()
        count = count + feats.size(0)

    # save total number of batches
    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()


if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default='',
                        help='config file with parameters of the experiment. '
                             'It is assumed that all the config file is placed on ./config/')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--openl3', action='store_true', help='use reduced classes')
    parser.add_argument('--poly', type=str, default='1', help='polyphony of support examples')
    args = parser.parse_args()

    # Load config file, set exp folder
    exp_config_file = os.path.join('.', 'config', args.config + '.py')
    exp_directory = os.path.join('.', 'experiments', args.config)

    config = imp.load_source("", exp_config_file).config
    config['exp_dir'] = exp_directory  # the place where logs, models, and other stuff will be stored

    # initialize an object of class Fewshot (in algorithm/fewshot.py)
    # this also initialize the class Algorithm, which init networks, criterions, and optimizers
    algorithm = alg.FewShot(config)

    # load best model and set it to eval mode
    if args.cuda:
        algorithm.load_to_gpu()
    algorithm.load_checkpoint(epoch='*', train=False, suffix='.best')

    for key, network in algorithm.networks.items():
        network.eval()

    # get eval data file path
    # base_eval_file has tracks with base labels only; novel_eval_file has tracks with novel(test) labels only
    base_filename = 'base_test'
    novel_filename = 'test'

    # if args.poly == 'mix':
    #     novel_filename += '_mixed'

    base_eval_file = 'base_test_filelist.pkl'
    novel_eval_file = 'test_query_filelist.pkl'

    # set dataloader
    batch_size = 32
    datamgr = SimpleDataManager(batch_size=batch_size)
    base_data_loader = datamgr.get_data_loader(base_eval_file, args.openl3)
    novel_data_loader = datamgr.get_data_loader(novel_eval_file, args.openl3)

    # set paths for output feature files
    base_outfilename, novel_outfilename = 'eval_base', 'eval_novel'

    base_outfile = os.path.join(exp_directory, 'features', base_outfilename + '.hdf5')
    novel_outfile = os.path.join(exp_directory, 'features', novel_outfilename + '.hdf5')

    dirname = os.path.dirname(base_outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    # compute and save features for base and novel eval data
    save_features(algorithm.networks['feat_model'], base_data_loader, base_outfile)
    save_features(algorithm.networks['feat_model'], novel_data_loader, novel_outfile)
