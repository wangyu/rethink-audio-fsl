"""
-------------------------------File info-------------------------
% - File name: refine_clips_annotations.py
% - Description: Remove duplicate annotations from FSD_MIX_CLIPS_annotations 
% - and generate updated annotation files
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2022-05-24
%  Copyright (C) ASVP, South China university of technology; 2022
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, ASVP, SCUT
% - Contact: ee_w.xie@mail.scut.edu.cn
------------------------------------------------------------------
"""
import os
from os.path import isfile, join, isdir
import pandas as pd
from collections import Counter
import argparse
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clips_ann_path', type=str, required=True, help='path to FSD_MIX_CLIPS_annotations folder')
    parser.add_argument('--savepath', type=str, required=True, help='path to save revised annotations)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    meta_list = ['base_train.csv', 'base_val.csv', 'base_test.csv', 'novel_val.csv', 'novel_test.csv']  # -

    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    for _, meta_name in enumerate(meta_list):
        filename_list = []

        annfile = join(args.clips_ann_path, meta_name)  # -
        ann = pd.read_csv(annfile)
        for idx in range(len(ann)):
            filename = ann['filename'][idx]
            start_time = ann['start_time'][idx]
            start_sample = int(start_time * 44100)

            fname = filename.replace('.wav', '_' + str(start_sample))
            filename_list.append(fname)
        unique_filenames = set(filename_list)

        print(f'num. of files in {meta_name} :{len(filename_list)}, unique num: {len(set(filename_list))}')
        unique_index_list = []
        for idx in range(len(ann)):
            filename = ann['filename'][idx]
            start_time = ann['start_time'][idx]
            start_sample = int(start_time * 44100)
            fname = filename.replace('.wav', '_' + str(start_sample))

            if fname in unique_filenames:
                unique_index_list.append(idx)
                unique_filenames.remove(fname)
        print(f'check num. of unique index: {len(unique_index_list)}')

        ann_revised = pd.DataFrame(ann, index=unique_index_list)
        revised_ann_dir = os.path.join(args.savepath, meta_name)
        ann_revised.to_csv(revised_ann_dir, index=False, encoding="utf-8")


