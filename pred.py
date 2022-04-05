import os
from os.path import join
import numpy as np
import argparse
import pickle as pkl
import random
from tqdm import tqdm
import imp
import h5py
from sklearn.metrics import average_precision_score  # import before torch to avoid error

import torch
import torch.nn as nn
import algorithms as alg


def feat_loader(filename):
    with h5py.File(filename, 'r') as f:
        feats = f['all_feats'][...]
        labels = f['all_labels'][...]
    while np.sum(feats[-1]) == 0:  # remove zeros feats in the last batch
        feats = np.delete(feats, -1, axis=0)
        labels = np.delete(labels, -1, axis=0)
    return feats, labels


def get_score(classifier, features_test, Kbase_ids, features_train, labels_train):
    scores, att_coeff, novel_weight = classifier(features_test=features_test, Kbase_ids=Kbase_ids,
                                                 features_train=features_train, labels_train=labels_train)
    scores = nn.Sigmoid()(scores)
    scores = scores.data.cpu().numpy().squeeze()
    return scores, att_coeff, novel_weight


if __name__ == '__main__':
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default='',
                        help='config file with parameters of the experiment. '
                             'It is assumed that all the config file is placed on ./config/')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--n_base', type=int, default=59, help='number of base tags')
    parser.add_argument('--n_novel', type=int, default=15, help='number of novel tags')
    parser.add_argument('--n_pos', type=int, default=5, help='number of positive support examples')
    parser.add_argument('--niter', type=int, default=100, help='number of evaluation iteration')
    parser.add_argument('--poly', type=str, default='1', help='1/mix, polyphony of support examples')
    parser.add_argument('--snr', type=str, default='mix', help='mix/low/high, snr of support examples')
    args = parser.parse_args()

    # load config file, set exp folder
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

    # set Kbase, a sequence of base labels, which is shuffled in each training iteration.
    # At eval, Kbase always = range(n_base)
    n_base = args.n_base
    Kbase = torch.LongTensor(range(n_base)).unsqueeze(0).cuda().contiguous()

    # load pre-computed features of base and novel eval tracks
    base_feat_filename, novel_feat_filename = 'eval_base', 'eval_novel'
    base_featfile = join(exp_directory, 'features', base_feat_filename+ 'poly.hdf5')
    novel_featfile = join(exp_directory, 'features', novel_feat_filename + 'poly.hdf5')

    base_feats, base_labels = feat_loader(base_featfile)
    novel_feats, novel_labels = feat_loader(novel_featfile)

    # allocate features to cuda
    base_feats = torch.from_numpy(base_feats).cuda()
    novel_feats = torch.from_numpy(novel_feats).cuda()

    # expand or reshape dimension as needed before applying classifier
    base_feats = base_feats.unsqueeze(0)
    novel_feats = novel_feats.unsqueeze(0)

    # combine base and novel feats
    feats_all = torch.cat((base_feats, novel_feats), dim=1)
    n_all = args.n_base + args.n_novel
    n_all_examples = feats_all.shape[1]

    # load support example filelist
    test_support_filelist = pkl.load(open('test_support_filelist.pkl','rb'))
    # support_snr_idx = pkl.load(open(join(datadir, 'fsd_test_support_idx_openl3.pickle'), 'rb'))
    # if args.poly == 'mix':
    #     unlabeled_train_tracks_with_novel_tags = pkl.load(open(join(datadir, 'fsd_test_support_openl3_mixed.pickle'), 'rb'))
    #     support_snr_idx = pkl.load(open(join(datadir, 'fsd_test_support_idx_openl3_mixed.pickle'), 'rb'))

    # create a lookup dictionary, key:value = tag:tracks labeled with tag
    novel_tag_to_file = dict()
    for i in range(len(test_support_filelist['data'])):
        for tag in test_support_filelist['labels'][i]:
            if tag not in novel_tag_to_file:
                novel_tag_to_file[tag] = []
            novel_tag_to_file[tag].append(test_support_filelist['data'][i])

    # Pred
    # pred with base classes only
    scores_base, att_coeff_base, novel_weight = get_score(algorithm.networks['classifier'], features_test=feats_all,
                                                          Kbase_ids=Kbase, features_train=None, labels_train=None)

    # pred with base+novel classes
    # initialize pred, novel_tags, att_coefficient arrays
    pred_all = np.zeros((args.niter, n_all_examples, n_all))
    novel_tags_all = np.zeros((args.niter, args.n_novel))
    att_coeff_all = np.zeros((args.niter, args.n_novel * args.n_pos, args.n_base))  # 5 support examples per novel class

    novel_weight_all = np.zeros((args.niter, args.n_novel, 2048))

    # for each prediction iteration, sample novel tags, sample novel support examples, predict
    novel_range = range(args.n_base + args.n_novel, args.n_base + 2 * args.n_novel)

    for iteration in tqdm(range(args.niter)):
        novel_tags = list(novel_range)

        # 5 support examples per novel class
        novel_train_feats = np.zeros((args.n_novel * args.n_pos, 512))

        # initialize labels array for novel support examples, batch_size = 1
        novel_train_labels = np.zeros((1, args.n_novel * args.n_pos, n_all))

        # sample 5 support examples per selected novel class
        i = 0
        for t in range(args.n_novel):
            tag = novel_tags[t]
            # if args.snr == 'low':
            #     idx_list = support_snr_idx[tag]['low']
            #     if len(idx_list) < args.n_pos:
            #         idx_list += support_snr_idx[tag][10]
            #     if len(idx_list) < args.n_pos:
            #         idx_list += support_snr_idx[tag][15]
            #     if len(idx_list) < args.n_pos:
            #         idx_list += support_snr_idx[tag][20]
            #     rand_idx = np.asarray(random.sample(idx_list, args.n_pos))
            # elif args.snr == 'high':
            #     idx_list = support_snr_idx[tag]['high']
            #     if len(idx_list) < args.n_pos:
            #         idx_list += support_snr_idx[tag][5]
            #     if len(idx_list) < args.n_pos:
            #         idx_list += support_snr_idx[tag][0]
            #     if len(idx_list) < args.n_pos:
            #         idx_list += support_snr_idx[tag][-5]
            #     rand_idx = np.asarray(random.sample(idx_list, args.n_pos))
            # else:
            rand_idx = np.random.permutation(len(novel_tag_to_file[tag]))[:args.n_pos]
            while len(rand_idx) < args.n_pos:
                pad_len = args.n_pos - len(rand_idx)
                rand_idx = np.concatenate((rand_idx, rand_idx[:pad_len]), axis=0)

            # load emb and label of the selected example and save them in the corresponding arrays
            for idx in rand_idx:
                path = novel_tag_to_file[tag][idx]
                feat = pkl.load(open(path, 'rb'))
                novel_train_feats[i] = feat
                novel_train_labels[0][i][args.n_base + t] = 1.  # relabel as (n_base + index in sampled novel_tags)
                i += 1

        novel_train_feats = torch.from_numpy(novel_train_feats).float().cuda()
        novel_train_feats = algorithm.networks['feat_model'](novel_train_feats).unsqueeze(0)

        # also move label array of novel support examples to cuda
        novel_train_labels = torch.from_numpy(novel_train_labels).float().cuda()

        # get predicted scores with novel support examples provided
        scores_both, att_coeff, novel_weight = get_score(algorithm.networks['classifier'], features_test=feats_all,
                                                         Kbase_ids=Kbase, features_train=novel_train_feats,
                                                         labels_train=novel_train_labels)

        # save scores, sampled novel tags, and att coefficients to corresponding arrays
        pred_all[iteration] = scores_both
        novel_tags_all[iteration] = novel_tags
        if att_coeff is not None:
            att_coeff_all[iteration] = att_coeff.squeeze()
        if novel_weight is not None:
            novel_weight_all[iteration] = novel_weight

    # save prediction results
    outfilename = 'base'+str(args.n_base)+'_novel'+str(args.n_novel)+'_pos'+str(args.n_pos)

    outfile_base = join(exp_directory, 'preds', outfilename+'_base.pkl')
    outfile_both = join(exp_directory, 'preds', outfilename +'_both.pkl')

    dirname = os.path.dirname(outfile_base)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    with open(outfile_base, 'wb') as f:
        pkl.dump(scores_base, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(outfile_both, 'wb') as f:
        pkl.dump(dict(preds=pred_all, novel_tags=novel_tags_all, att_coeff_all=att_coeff_all,
                      novel_weight_all=novel_weight_all), f, protocol=pkl.HIGHEST_PROTOCOL)
