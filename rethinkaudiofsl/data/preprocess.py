import os
from os import listdir
from os.path import join
import json
import pandas as pd
import matplotlib.pyplot as plt
import random
import IPython.display as ipd
import librosa
import numpy as np
import shutil
from shutil import copyfile
import pandas as pd
import argparse


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


def filter_pp_rating(ratings, vocab, inter_nodes, files): #file_to_class, class_to_file
    # get all files that have a single leaf label with pp rating from all annotators
    singlePP_files = []
    for file in files:
        mids = list(ratings[file].keys())
        # convert mids to labels
        labels = [vocab[1][np.where(vocab[2] == mid)[0][0]] for mid in mids if mid in vocab[2].values]
        leaf_labels = list(set(labels) - set(inter_nodes))

        if len(leaf_labels) == 1 and all(x == 1.0 for x in ratings[file][leaf_labels[0]]):
            singlePP_files.append(file)

    return singlePP_files


def filter_duration(max_duration, audiopath, files):
    # only keep files with duration shorter than max_duration
    return [
        f for f in files if librosa.get_duration(filename=join(audiopath, f+'.wav')) < max_duration
    ]


def filter_class_occrrences(min_occur, class_to_file):
    return {cl:class_to_file[cl] for cl in class_to_file if len(class_to_file[cl]) >= min_occur}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fsdpath', type=str, default='./fsd50k', help='path to FSD50K data folder.')
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
    short_files_dev = filter_duration(
        max_duration=args.max_duration, audiopath=join(args.fsdpath, 'FSD50K.dev_audio'), files=PP_files_dev
    )
    short_files_eval = filter_duration(
        max_duration=args.max_duration, audiopath=join(args.fsdpath, 'FSD50K.eval_audio'), files=list(file_to_class_eval.keys())
    )

    # Get dictionaries with filtered files: single-labeled, PP-rating, shorter than max duration
    class_to_shortPP_file_dev = {cl: list(set(class_to_file_dev[cl]) & set(short_files_dev)) for cl in class_to_file_dev}
    class_to_short_file_eval = {cl: list(set(class_to_file_eval[cl]) & set(short_files_eval)) for cl in class_to_file_eval}

    # Filter out rare classes
    common_class_to_shortPP_file_dev = filter_class_occrrences(min_occur=args.min_class_occurrence,
                                                               class_to_file=class_to_shortPP_file_dev)
    common_class_to_short_file_eval = filter_class_occrrences(min_occur=args.min_class_occurrence,
                                                               class_to_file=class_to_short_file_eval)

    # Load class lists for each split
    with open("train_tag.json") as f:
        train_tag = json.load(f)
    with open("val_tag.json") as f:
        val_tag = json.load(f)
    with open("test_tag.json") as f:
        test_tag = json.load(f)

    
    # Copy files to new folders where each folder is named by a class label
    audiopath = join(datapath, 'FSD50K.eval_audio')

    for cl in class_short_file_eval_filtered:
        folderpath = join(datapath, 'scaper', 'foreground', str(all_tag.index(cl)))
        for fname in class_short_file_eval_filtered[cl]:
            copyfile(join(audiopath, fname + '.wav'), join(folderpath, fname + '.wav'))


    # inter_nodes = ['Aircraft', 'Alarm', 'Animal', 'Bell', 'Bicycle', 'Bird',
    #                'Bird_vocalization_and_bird_call_and_bird_song', 'Brass_instrument', 'Breathing', 'Car',
    #                'Cat', 'Chime', 'Clock', 'Cymbal', 'Dog', 'Domestic_animals_and_pets', 'Domestic_sounds_and_home_sounds',
    #                'Door', 'Drum', 'Engine', 'Explosion', 'Fire', 'Fowl', 'Glass', 'Guitar', 'Hands',
    #                'Human_group_actions', 'Human_voice', 'Insect', 'Keyboard_(musical)', 'Laughter',
    #                'Liquid', 'Mallet_percussion', 'Mechanisms',' Motor_vehicle_(road)', 'Music', 'Musical_instrument',
    #                'Ocean', 'Percussion', 'Plucked_string_instrument', 'Pour', 'Power_tool', 'Rail_transport',
    #                'Rain', 'Respiratory_sounds', 'Shout', 'Singing', 'Speech', 'Telephone', 'Thunderstorm',
    #                'Tools', 'Typing', 'Vehicle', 'Water', 'Wild_animals', 'Wood']



    # singlePP_file_to_class = {file:file_to_class[file] for file in singlePP_files}
    # class_to_singlePP_file = {cl:list(set(class_to_file[cl] & set(singlePP_files))) for cl in class_to_file}
    # # remove classes that do not have any single-PP files
    # for cl in class_to_singlePP_file:
    #     if len(class_to_singlePP_file[cl]) == 0:
    #         del class_to_singlePP_file[cl]
    #
    # return class_to_singlePP_file, singlePP_file_to_class