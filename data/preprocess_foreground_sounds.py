import os
from os import listdir
from os import makedirs
from os.path import join, isdir
import json
import random
import shutil
import argparse
import numpy as np
import pandas as pd
import librosa
import sox


def filter_single_labeled(ann, inter_nodes):
    # get single-labeled filenames and classes
    class_to_file = dict()
    file_to_class = dict()

    for idx in range(len(ann)):
        fname = str(ann['fname'][idx])
        labels = ann['labels'][idx].split(',')
        leaf_labels = list(set(labels) - set(inter_nodes))

        if len(leaf_labels) == 1:
            leaf_label = leaf_labels[0]
            if leaf_label not in class_to_file:
                class_to_file[leaf_label] = []
            class_to_file[leaf_label].append(fname)
            file_to_class[fname] = leaf_label

    return class_to_file, file_to_class


def filter_pp_rating(ratings, vocab, inter_nodes, files):
    # get all files that have a single leaf label with pp rating from all annotators
    singlePP_files = []
    for file in files:
        mids = list(ratings[file].keys())
        # convert mids to labels
        labels = [vocab[1][np.where(vocab[2] == mid)[0][0]] for mid in mids if mid in vocab[2].values]
        leaf_labels = list(set(labels) - set(inter_nodes))
        # covert leaf_labels to mids        
        leaf_mids = [vocab[2][np.where(vocab[1] == label)[0][0]] for label in leaf_labels if label in vocab[1].values]
        
        if len(leaf_labels) == 1 and all(x == 1.0 for x in ratings[file][leaf_mids[0]]):
            singlePP_files.append(file)

    return singlePP_files


def filter_duration(max_duration, audiopath, files):
    # only keep files with duration shorter than max_duration
    return [
        f for f in files if librosa.get_duration(filename=join(audiopath, f+'.wav')) < max_duration
    ]


def filter_class_occrrences(min_occur, class_to_file):
    return {cl:class_to_file[cl] for cl in class_to_file if len(class_to_file[cl]) >= min_occur}


def trim_edge_silence(audiofile, outfile, silence_threshold, min_silence_duration):
    tfm = sox.Transformer()
    tfm.silence(location=1, silence_threshold=silence_threshold, min_silence_duration=min_silence_duration)
    tfm.silence(location=-1, silence_threshold=silence_threshold, min_silence_duration=min_silence_duration)
    tfm.build(audiofile, outfile)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fsdpath', type=str, required=True, help='path to FSD50K data folder.')
    parser.add_argument('--outpath', type=str, default='./', help='path to save reorganized FSD50K audio files for the'
                                                                  'following Scaper generation.')
    parser.add_argument('--max_clip_duration', type=int, default=4, help='max duration for each clip in sec)')
    parser.add_argument('--min_class_occurrence', type=int, default=10, help='min number of examples for a class to be included')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Load annotations
    ann_dev = pd.read_csv(join(args.fsdpath, 'FSD50K.ground_truth', 'dev.csv'))
    ann_eval = pd.read_csv(join(args.fsdpath, 'FSD50K.ground_truth', 'eval.csv'))

    # Load annotation ratings
    with open(join(args.fsdpath, 'FSD50K.metadata/pp_pnp_ratings_FSD50K.json')) as f:
        ratings = json.load(f)

    # Load vocab
    vocab = pd.read_csv(join(args.fsdpath, '/FSD50K.ground_truth/vocabulary.csv'), header=None)

    # Load intermediate nodes
    with open("inter_nodes.json") as f:
        inter_nodes = json.load(f)

    # Get dictionaries with single-labeled files
    class_to_file_dev, file_to_class_dev = filter_single_labeled(ann_dev, inter_nodes)
    class_to_file_eval, file_to_class_eval = filter_single_labeled(ann_eval, inter_nodes)

    # For dev set, only keep files with PP rating
    # Didn't do this for the eval set since it has been more carefully and thoroughly curated
    # according to the original FSD50k paper
    PP_files_dev = filter_pp_rating(ratings, vocab, inter_nodes, files=list(file_to_class_dev.keys()))

    # Get dev and eval files with duration shorter than max_duration
    audiopath_dev = join(args.fsdpath, 'FSD50K.dev_audio')
    audiopath_eval = join(args.fsdpath, 'FSD50K.eval_audio')

    short_files_dev = filter_duration(max_duration=args.max_clip_duration, audiopath=audiopath_dev, files=PP_files_dev)
    short_files_eval = filter_duration(max_duration=args.max_clip_duration, audiopath=audiopath_eval, files=list(file_to_class_eval.keys()))

    # Get dictionaries with filtered files: single-labeled, PP-rating, shorter than max duration
    class_to_shortPP_file_dev = {cl: list(set(class_to_file_dev[cl]) & set(short_files_dev)) for cl in class_to_file_dev}
    class_to_short_file_eval = {cl: list(set(class_to_file_eval[cl]) & set(short_files_eval)) for cl in class_to_file_eval}

    # Filter out rare classes
    common_class_to_shortPP_file_dev = filter_class_occrrences(min_occur=args.min_class_occurrence,
                                                               class_to_file=class_to_shortPP_file_dev)
    common_class_to_short_file_eval = filter_class_occrrences(min_occur=args.min_class_occurrence,
                                                               class_to_file=class_to_short_file_eval)

    # Load class lists for each split
    with open("all_tag.json") as f:
        all_tag = json.load(f)
    with open("train_tag.json") as f:
        train_tag = json.load(f)
    with open("val_tag.json") as f:
        val_tag = json.load(f)
    with open("test_tag.json") as f:
        test_tag = json.load(f)

    
    # Trim files and save to new folders where each folder is named by a class label
    for cl in common_class_to_shortPP_file_dev:
        if cl in train_tag:
            outpath = join(args.outpath, 'foreground', 'base', 'train', str(all_tag.index(cl)))
        elif cl in val_tag:
            outpath = join(args.outpath, 'foreground', 'val', str(all_tag.index(cl)))
        else:
            outpath = join(args.outpath, 'foreground', 'test', str(all_tag.index(cl)))

        if not isdir(outpath):
            makedirs(outpath)

        for file in common_class_to_shortPP_file_dev[cl]:
            trim_edge_silence(
                audiofile=join(audiopath_dev, file+ '.wav'), outfile=join(outpath, file+ '.wav'), silence_threshold=0.1, min_silence_duration=0.05
            )

    for cl in common_class_to_short_file_eval:
        if cl in train_tag:
            outpath = join(args.outpath, 'foreground', 'base', 'test', str(all_tag.index(cl)))
        elif cl in val_tag:
            outpath = join(args.outpath, 'foreground', 'val', str(all_tag.index(cl)))
        else:
            outpath = join(args.outpath, 'foreground', 'test', str(all_tag.index(cl)))

        if not isdir(outpath):
            makedirs(outpath)

        for file in common_class_to_shortPP_file_dev[cl]:
            trim_edge_silence(
                audiofile=join(audiopath_eval, file+'.wav'), outfile=join(outpath, file+'.wav'), silence_threshold=0.1, min_silence_duration=0.05
            )

    # Split train/val examples in base classes
    for cl in train_tag:
        train_path = join(args.outpath, 'foreground', 'base', 'train', str(cl))
        val_path =  join(args.outpath, 'foreground', 'base', 'val', str(cl))
        if not isdir(val_path):
            makedirs(val_path)

        # shuffle all files in the folder
        fnames = [f for f in listdir(train_path) if '.wav' in f]
        n_val = int(np.ceil(len(fnames) * 0.15))
        random.shuffle(fnames)

        # move a portion of files to the val folder
        f_val = fnames[:n_val]
        for f in f_val:
            shutil.move(join(train_path, f), join(val_path, f))
