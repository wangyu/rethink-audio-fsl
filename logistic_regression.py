import os
from os import makedirs
from os.path import join, isdir
import pickle as pkl
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import imp
import algorithms as alg
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import argparse

def feat_loader(filename):
    with h5py.File(filename, 'r') as f:
        feats = f['all_feats'][...]
        labels = f['all_labels'][...]
    while np.sum(feats[-1]) == 0:  # remove zeros feats in the last batch
        feats = np.delete(feats, -1, axis=0)
        labels = np.delete(labels, -1, axis=0)
    return feats, labels


def train_clf_and_compute_metrics(niter, novel_tag_to_support_track, train_tracks, cl,
                                  feat_model, n_pos, n_neg, emb_dim, test_embs, test_labels):
    maps = []
    fs = []
    ps = []
    rs = []
    preds_all = np.zeros((10, test_embs.shape[0]))
    clfs = []

    for n in range(niter):
        # sample positive examples and compute positive embeddings
        rand_pos_idx = np.random.permutation(len(novel_tag_to_support_track[cl]))[:n_pos]
        pos_paths = [novel_tag_to_support_track[cl][idx] for idx in rand_pos_idx]
        pos_embs = np.zeros((n_pos, emb_dim))
        for i in range(n_pos):
            pos_embs[i] = pkl.load(open(pos_paths[i], 'rb'))

        pos_embs = feat_model(torch.from_numpy(pos_embs).float().cuda())
        pos_embs = pos_embs.data.cpu().numpy()

        # sample negative examples and compute negative embeddings
        rand_neg_idx = np.random.permutation(len(train_tracks['data']))[:n_neg]
        neg_paths = [train_tracks['data'][idx] for idx in rand_neg_idx]
        neg_embs = np.zeros((n_neg, emb_dim))
        for i in range(n_neg):
            neg_embs[i] = pkl.load(open(neg_paths[i], 'rb'))

        neg_embs = feat_model(torch.from_numpy(neg_embs).float().cuda()).data.cpu().numpy()

        # combine positive and negative examples
        X_embs = np.concatenate((pos_embs, neg_embs), axis=0)
        y = np.asarray([1] * n_pos + [0] * n_neg)

        shuffler = np.random.permutation(len(X_embs))
        X_embs = X_embs[shuffler]
        y = y[shuffler]

        # train LR clf
        clf = LogisticRegression(random_state=0, max_iter=1000, class_weight='balanced').fit(X_embs, y)

        # predict test data
        preds = clf.predict(test_embs)
        preds_all[n] = preds

        # compute metrics
        targets = []
        for labels in test_labels:
            if cl in labels:
                targets.append(1)
            else:
                targets.append(0)

        maps.append(average_precision_score(targets, preds))
        fs.append(f1_score(targets, preds))
        ps.append(precision_score(targets, preds))
        rs.append(recall_score(targets, preds))
        clfs.append(clf)

    return np.mean(maps), np.mean(fs), np.mean(ps), np.mean(rs), preds_all, clfs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default='',
                        help='config file with parameters of the experiment. '
                             'It is assumed that all the config file is placed on ./config/')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--n_base', type=int, default=59, help='number of base tags')
    parser.add_argument('--n_novel', type=int, default=15, help='number of novel tags')
    parser.add_argument('--n_pos', type=int, default=5, help='number of positive support examples')
    parser.add_argument('--n_neg', type=int, default=100, help='number of negative examples')
    parser.add_argument('--niter', type=int, default=10, help='number of evaluation iteration')
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

    feat_model = algorithm.networks['feat_model']

    # load filelists
    train_tracks = pkl.load(open('base_train_filelist.pkl', 'rb'))
    test_supports = pkl.load(open('test_support_filelist.pkl', 'rb'))
    test_tracks_base = pkl.load(open('base_test_filelist.pkl', 'rb'))
    test_tracks_novel = pkl.load(open('test_query_filelist.pkl', 'rb'))

    test_paths = test_tracks_base['data'] + test_tracks_novel['data']
    test_labels = test_tracks_base['labels'] + test_tracks_novel['labels']

    # load computed features
    base_feats, base_labels = feat_loader(config['exp_dir']+'features/eval_base.hdf5')
    novel_feats, novel_labels = feat_loader(config['exp_dir']+'features/eval_novel.hdf5')

    test_embs = np.concatenate((base_feats, novel_feats), axis=0)

    # get lookup dict
    novel_tag_to_support_track = dict()
    for i in range(len(test_supports['data'])):
        for tag in test_supports['labels'][i]:
            if tag not in novel_tag_to_support_track:
                novel_tag_to_support_track[tag] = []
            novel_tag_to_support_track[tag].append(test_supports['data'][i])

    # pred
    novel_tags = list(range(74, 89))
    preds = np.zeros((10, test_embs.shape[0], 15))
    clfs = []
    metrics = dict()

    for cl in tqdm(novel_tags):
        map, f, p, r, pred_cl, clf = train_clf_and_compute_metrics(args.niter, novel_tag_to_support_track,
                                                                    train_tracks, cl, feat_model,
                                                                    n_pos=args.n_pos, n_neg=args.n_neg, emb_dim=512,
                                                                    test_embs=test_embs, test_labels=test_labels)
        preds[:, :, novel_tags.index(cl)] = pred_cl
        clfs.append(clf)
        metrics[cl] = dict()
        metrics[cl]['map'], metrics[cl]['f'], metrics[cl]['p'], metrics[cl]['r'] = map, f, p, r

    # save trained LR models, predictions, and metrics
    outfolder = './experiments/LR'
    clf_folder = join(outfolder, 'clfs')
    pred_folder = join(outfolder, 'preds')
    metric_folder = join(outfolder, 'metrics')

    if not isdir(clf_folder):
        makedirs(clf_folder)
        makedirs(pred_folder)
        makedirs(metric_folder)

    outfile = 'pos'+str(args.n_pos) + '_neg'+str(args.n_neg)

    with open(join(clf_folder, outfile+'.pkl'), 'wb') as f:
        pkl.dump(clfs, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(join(pred_folder, outfile+'.pkl'), 'wb') as f:
        pkl.dump(preds, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(join(metric_folder, outfile+'.pkl'), 'wb') as f:
        pkl.dump(metrics, f, protocol=pkl.HIGHEST_PROTOCOL)
