import os
from os.path import isfile, join
import pandas as pd
import pickle as pkl
import argparse

import openl3
import soundfile as sf


def get_openl3_and_filelists(annfile, audiofolder, savefolder, overwrite=False):
    # load pre-trained openl3 audio embedding model
    model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music", embedding_size=512)

    filelist = {'data': [], 'labels': []}
    ann = pd.read_csv(annfile)

    for idx in range(len(ann)):
        fname = ann['filename'][idx]
        start_time = ann['start_time'][idx]
        labels = [int(x) for x in ann['labels'][1][1:-1].split(',')]  # convert string to list of int

        start_sample =  int(start_time * 44100)
        outfile = savefolder + fname + '_' + str(start_sample)+ '.pkl'

        if not isfile(outfile) or overwrite:
            audio, sr = sf.read(join(audiofolder, fname))
            audio = audio[start_sample:start_sample + 44100]
            emb, ts = openl3.get_audio_embedding(audio, sr, model=model, center=False)
            emb = emb.squeeze()

            with open(outfile, 'wb') as f:
                pkl.dump(emb, f, protocol=pkl.HIGHEST_PROTOCOL)

            filelist['data'].append(outfile)
            filelist['labels'].append(labels)

    return filelist


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

                filelist = get_openl3_and_filelists(annfile, audiofolder=audiofolder, savefolder=savefolder, overwrite=False)

                with open(class_split+'_'+data_split+'.pkl', 'wb') as f:
                    pkl.dump(filelist, f, protocol=pkl.HIGHEST_PROTOCOL)
        else:
            annfile = join(args.annpath, 'novel_' + class_split + '.csv')
            audiofolder = join(args.audiopath, class_split)
            savefolder = join(args.savepath, class_split)

            filelist = get_openl3_and_filelists(annfile, audiofolder=audiofolder, savefolder=savefolder, overwrite=False)

            with open(class_split + '.pkl', 'wb') as f:
                pkl.dump(filelist, f, protocol=pkl.HIGHEST_PROTOCOL)
