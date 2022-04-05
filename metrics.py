import os
from os.path import join
import numpy as np
import argparse
import pickle as pkl
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve


def load_pred_dict(exppath, n_base=100, n_novel=20, n_pos=5, poly='1', snr='mix'):
    file = 'base' + str(n_base) + '_novel' + str(n_novel) + '_pos' + str(n_pos)

    pred_base = pkl.load(open(join(exppath, 'preds', file + '_base.pkl'), 'rb'))
    pred_dict_both = pkl.load(open(join(exppath, 'preds', file + '_both.pkl'), 'rb'))

    return pred_base, pred_dict_both


def compute_metrics(pred, target):
    niter = pred.shape[0]
    n_cl = pred.shape[-1]

    # initialize metrics dictionary
    metrics = dict()

    for metric in ['AP', 'P', 'R', 'F', 'tp', 'tn', 'fp', 'fn']:
        metrics[metric] = np.zeros((niter, n_cl))

    for n in tqdm(range(niter)):
        for cl in range(n_cl):
            cl_activation = target[n, :, cl]  # ground truth class activation
            cl_activation = [int(x) for x in cl_activation]
            cl_pred = pred[n, :, cl]  # prediction in cl
            cl_pred_binary = np.where(cl_pred > 0.5, 1, 0)  # convert likelihood to binary with 0.5 threshold

            if np.any(cl_activation):
                metrics['AP'][n][cl] = average_precision_score(cl_activation, cl_pred)
                tn, fp, fn, tp = confusion_matrix(cl_activation, cl_pred_binary).ravel()
                metrics['tn'][n][cl], metrics['fp'][n][cl], metrics['fn'][n][cl], metrics['tp'][n][cl] = tn, fp, fn, tp
                metrics['P'][n][cl] = precision_score(cl_activation, cl_pred_binary)
                metrics['R'][n][cl] = recall_score(cl_activation, cl_pred_binary)
                metrics['F'][n][cl] = f1_score(cl_activation, cl_pred_binary)
            else:
                print('No ground truth for class: ', cl)

    return metrics


if __name__ == '__main__':
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default='',
                        help='config file with parameters of the experiment. '
                             'It is assumed that all the config file is placed on ./config/')
    parser.add_argument('--n_base', type=int, default=59, help='number of base tags')
    parser.add_argument('--n_novel', type=int, default=15, help='number of novel tags')
    parser.add_argument('--n_pos', type=int, default=5, help='number of positive support examples')
    parser.add_argument('--openl3', action='store_true', help='use reduced classes')
    parser.add_argument('--poly', type=str, default='1', help='polyphony of support examples')
    parser.add_argument('--snr', type=str, default='mix', help='mix/low/high, snr of support examples')
    args = parser.parse_args()

    # set exp path
    exppath = os.path.join('.', 'experiments', args.config)

    # load prediction
    pred_base, pred_dict_both = load_pred_dict(exppath, n_base=args.n_base, n_novel=args.n_novel, n_pos=args.n_pos,
                                               poly=args.poly, snr=args.snr)

    # get number of eval examples, number of prediction iteration, and sampled novel tags
    n_eval = pred_base.shape[0]
    niter = pred_dict_both['preds'].shape[0]
    novel_tags = pred_dict_both['novel_tags']

    # load labels file
    eval_tracks_with_base_tags = pkl.load(open('base_test_filelist.pkl', 'rb'))
    eval_tracks_with_novel_tags = pkl.load(open('test_query_filelist.pkl', 'rb'))
    # if args.poly == 'mix':
    #     eval_tracks_with_novel_tags = pkl.load(open(join(pklpath, 'fsd_test_openl3_mixed.pickle'), 'rb'))

    # write target for evaluating with base labels only
    base_label = eval_tracks_with_base_tags['labels']
    n_eval_base = len(base_label)  # number of examples with base labels

    base_target = np.zeros((n_eval, args.n_base))
    for i in range(n_eval_base):  # examples with novel labels will not be labeled
        for label in base_label[i]:
            base_target[i][label] = 1

    # write target for evaluating with base and novel labels
    novel_label = eval_tracks_with_novel_tags['labels']
    n_eval_novel = len(novel_label) # number of examples with novel labels

    # initialize with base_target and zero activation in novel labels, for all prediction iteration
    both_target = np.concatenate((base_target, np.zeros((n_eval, args.n_novel))), axis=1)  # (n_eval, arg.n_base+args.n_novel)
    both_targets = np.repeat(both_target[np.newaxis, :, :], niter, axis=0)  # (niter, n_eval, arg.n_base+args.n_novel)

    # relabel examples with novel labels in each prediction iteration
    # in each iteration, get sampled novel_tag,
    # go over each novel example, check if it has label in novel_tags, if yes, label it
    for n in range(niter):
        novel_tag = novel_tags[n]
        for i in range(n_eval_novel):
            for label in novel_label[i]:
                if label in novel_tag:
                    both_targets[n][n_eval_base + i][args.n_base + list(novel_tag).index(label)] = 1

    # expand base pred and target dimension with niter=1
    pred_base = np.expand_dims(pred_base, axis=0)
    base_target = np.expand_dims(base_target, axis=0)

    # compute metrics
    metrics_base = compute_metrics(pred_base, base_target)
    metrics_both = compute_metrics(pred_dict_both['preds'], both_targets)

    # save files
    outfilename = 'base' + str(args.n_base) + '_novel' + str(args.n_novel) + '_pos' + str(args.n_pos)

    outfile_base = join(exppath, 'metrics', outfilename + '_base.pkl')
    outfile_both = join(exppath, 'metrics', outfilename + '_both.pkl')

    dirname = os.path.dirname(outfile_base)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    with open(outfile_base, 'wb') as f:
        pkl.dump(metrics_base, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open(outfile_both, 'wb') as f:
        pkl.dump(metrics_both, f, protocol=pkl.HIGHEST_PROTOCOL)
