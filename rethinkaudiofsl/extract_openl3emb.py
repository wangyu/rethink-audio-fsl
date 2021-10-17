import os
from os.path import isfile, join
import pandas as pd
import pickle as pkl
import argparse

import openl3
import soundfile as sf


def get_openl3(annfile, audiofolder, savefolder, overwrite=False):
    model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music", embedding_size=512)
    ann = pd.read_csv(annfile)

    for idx in range(len(ann)):
        fname = ann['filename'][idx]
        audiofile = audiofolder + fname
        start_time = ann['start_time'][idx]
        start_sample =  int(start_time * 44100)
        outfile = savefolder + fname + '_' + str(start_sample)+ '.pkl'

        if not isfile(outfile) or overwrite:
            audio, sr = sf.read(audiofile)
            audio = audio[start_sample:start_sample + 44100]
            emb, ts = openl3.get_audio_embedding(audio, sr, model=model, center=False)
            emb = emb.squeeze()

            with open(outfile, 'wb') as f:
                pkl.dump(emb, f, protocol=pkl.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annpath', type=str, required=True, help='path to FSD_MIX_CLIPS.annotations folder')
    parser.add_argument('--audiopath', type=str, required=True, help='path to generated FSD_MIX_SED audio folder')
    parser.add_argument('--savepath', type=str, required=True, help='path to save openl3 embs)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    for class_split in ['base', 'val', 'test']:
        if class_split == 'base':
            for data_split in ['train', 'val', 'test']:
                annfile = join(args.annpath, class_split + '_' + data_split + '.csv')
                audiofolder = join(args.audiopath, class_split, data_split)
                savefolder = join(args.savepath, class_split, data_split)
        else:
            annfile = join(args.annpath, 'novel_' + class_split + '.csv')
            audiofolder = join(args.audiopath, class_split)
            savefolder = join(args.savepath, class_split)

        get_openl3(annfile, audiofolder=audiofolder, savefolder=savefolder, overwrite=False)
