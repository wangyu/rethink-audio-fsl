import os
from os import makedirs, listdir
from os.path import join, isdir
import scaper
import argparse

def generate(jamspath, sourcepath, savepath, class_split, data_split):
    # set paths
    jamsfolder = join(jamspath, class_split)
    savefolder = join(savepath, class_split)
    fg_path = join(sourcepath, 'foreground', class_split)
    bg_path = join(sourcepath, 'background')

    # For base classes, data are further split into train/val/test folders
    if data_split:
        jamsfolder = join(jamsfolder, data_split)
        savefolder = join(savefolder, data_split)
        fg_path = join(fg_path, data_split)

    if not isdir(savefolder):
        makedirs(savefolder)

    # generate audio files given jams files
    for fname in listdir(jamsfolder)[:1]:
        jamsfile = join(jamsfolder, fname)
        savefile = join(savefolder, fname.replace('.jams', '.wav'))

        scaper.generate_from_jams(jams_infile = jamsfile,
                                  audio_outfile=savefile,
                                  fg_path=fg_path,
                                  bg_path=bg_path,
                                  jams_outfile=None,
                                  save_isolated_events=False,
                                  isolated_events_path=None,
                                  disable_sox_warnings=True,
                                  txt_path=None,
                                  txt_sep='\t')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jamspath', type=str, required=True, help='path to FSD_MIX_SED.annotations folder')
    parser.add_argument('--sourcepath', type=str, required=True, help='path to FSD_MIX_SED.source folder')
    parser.add_argument('--savepath', type=str, required=True, help='path to save generated soundscapes)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    for class_split in ['base', 'val', 'test']:
        if class_split == 'base':
            for data_split in ['train', 'val', 'test']:
                generate(args.jamspath, args.sourcepath, args.savepath, class_split=class_split, data_split=data_split)
        else:
            generate(args.jamspath, args.sourcepath, args.savepath, class_split=class_split, data_split=None)

