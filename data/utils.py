import os
from os.path import join
import jams
import tqdm
import librosa


def get_1sec_dict(annfolder):
    dict_1sec = {'data': [], 'start_frame': [], 'labels': []}

    for path, dirs, files in os.walk(annfolder):
        for idx in tqdm.tqdm(range(len(files))):
            file = files[idx]
            if 'jams' in file:
                jam = jams.load(join(path, file))
                for ann in jam.annotations:
                    # define 1sec window around each event
                    for obs in ann.data:
                        if obs.value['role'] == 'foreground':
                            label = [obs.value['label']]
                            start_time = obs.time
                            end_time = start_time + obs.duration
                            start_frame = librosa.time_to_frames(start_time, sr=16000, hop_length=160)
                            end_frame = librosa.time_to_frames(end_time, sr=16000, hop_length=160)

                            # if the event is shorter than 1sec, center-align the window
                            # otherwise, the window is left-aligned
                            if end_frame - start_frame < 100:
                                start_frame = (start_frame + end_frame) // 2 - 50

                                # shift the window accordingly if it's close to the clip edges
                            start_frame = max(0, start_frame)
                            start_frame = min(900, start_frame)

                            end_frame = start_frame + 100

                            # decide the level of polyphony and class labels for current window
                            for other_obs in ann.data:
                                if other_obs.value['role'] == 'foreground' and other_obs.value['label'] != label:
                                    other_start_time = other_obs.time
                                    other_end_time = other_start_time + other_obs.duration
                                    other_start_frame = librosa.time_to_frames(other_start_time, sr=16000,
                                                                               hop_length=160)
                                    other_end_frame = librosa.time_to_frames(other_end_time, sr=16000, hop_length=160)

                                    # Case 1: another event is present throughout the whole window
                                    if other_start_frame < start_frame and other_end_frame > end_frame:
                                        label.append(other_obs.value['label'])

                                        # Case 2: another event is present within the window
                                    elif other_start_frame > start_frame and other_end_frame < end_frame:
                                        label.append(other_obs.value['label'])

                                        # Case 3&4: the overlap between 2 events is longer than
                                    # half of the length of another event
                                    elif other_end_frame < end_frame:
                                        overlap = other_end_frame - start_frame
                                        if overlap >= (other_end_frame - other_start_frame) / 2:
                                            label.append(other_obs.value['label'])
                                    elif other_start_frame > start_frame:
                                        overlap = end_frame - other_start_frame
                                        if overlap >= (other_end_frame - other_start_frame) / 2:
                                            label.append(other_obs.value['label'])

                            dict_1sec['data'].append(
                                join(path, file).replace('annotations', 'melspec').replace('jams', 'npy'))
                            dict_1sec['start_frame'].append(start_frame)
                            dict_1sec['labels'].append(label)

    return dict_1sec

