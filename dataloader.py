from __future__ import print_function

import os
from os.path import join
import numpy as np
import random
import pickle

import torch
import torch.utils.data as data
import torchnet as tnt

from abc import abstractmethod


# Set the appropriate paths of the datasets here.
FILELIST_DIR = './'


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label_list in enumerate(labels):
        for label in label_list:
            if label not in label2inds:
                label2inds[label] = []
            label2inds[label].append(idx)

    return label2inds


def load_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo)
    return data


class SimpleDataset:
    def __init__(self, eval_file, openl3=False):
        self.eval = pickle.load(open(eval_file, 'rb'))
        self.data = self.eval['data']
        self.labels = self.eval['labels']
        self.openl3 = openl3
        if not self.openl3:
            self.start_frame = self.eval['start_frame']

    def __getitem__(self, index):
        path, label = self.data[index], self.labels[index]

        multihot_label = torch.zeros((87,))
        for t in label:
            multihot_label[t] = 1

        if self.openl3:
            emb = np.load(path, mmap_mode='r')
            emb = torch.from_numpy(emb)
            return emb, multihot_label
        else:
            mel = np.load(path, mmap_mode='r')
            if mel.shape[1] < 1000:  # pad if too short
                mel = np.pad(mel, ((0, 0), (0, 1000 - mel.shape[1])), 'constant')
            start = self.start_frame[index]
            mel = mel[:, start:start + 100]

            mel = np.expand_dims(mel, -1)  # (nmel, nframe, 1)
            mel = torch.from_numpy(mel)

            return mel, multihot_label

    def __len__(self):
        return len(self.data)


class DataManager:
    @abstractmethod
    def get_data_loader(self, base_eval_file, novel_eval_file):
        pass

class SimpleDataManager(DataManager):
    def __init__(self, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size

    def get_data_loader(self, eval_file, openl3=False):  # parameters that would change on train/val set
        """
        Build a DataLoader based on filelist
        Parameters
        ----------
        data_file: A filelist with datapaths and class labels
        Returns
        -------
        data_loader: DataLoader object
        """
        dataset = SimpleDataset(eval_file, openl3)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=False, num_workers=12, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


class FSD_MIX_CLIPS(data.Dataset):
    def __init__(self, phase='train', openl3=False):
        self.base_folder = 'fsd_mix_clips'
        assert(phase=='train' or phase=='val' or phase=='test')
        self.phase = phase
        self.name = 'fsd_mix_clips_' + phase
        self.openl3 = openl3
        if not self.openl3:
            self.start_frame = self.eval['start_frame']

        print('Loading fsdSED dataset - phase {0}'.format(phase))

        file_train_categories_train_phase = 'base_train_filelist.pkl'
        file_train_categories_val_phase = 'base_val_filelist.pkl'
        file_train_categories_test_phase = 'base_test_filelist.pkl'
        file_val_categories_val_phase = 'val_filelist.pkl'
        file_test_categories_test_phase = 'test_filelist.pkl'

        if self.phase=='train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data_train = load_data(file_train_categories_train_phase)
            self.data = data_train['data']
            self.labels = data_train['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)

        elif self.phase=='val' or self.phase=='test':
            if self.phase=='test':
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = load_data(file_train_categories_test_phase)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = load_data(file_test_categories_test_phase)
            else: # phase=='val'
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = load_data(file_train_categories_val_phase)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = load_data(file_val_categories_val_phase)

            self.data = np.concatenate([data_base['data'], data_novel['data']], axis=0)
            self.labels = data_base['labels'] + data_novel['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

            self.labelIds_base = buildLabelIndex(data_base['labels']).keys()
            self.labelIds_novel = buildLabelIndex(data_novel['labels']).keys()
            self.num_cats_base = len(self.labelIds_base)
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert(len(intersection) == 0)

        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))


    def __getitem__(self, index):
        path, label = self.data[index], self.labels[index]

        if self.openl3:
            emb = np.load(path, mmap_mode='r')
            return emb, label
        else:
            # # use this for pann backbone
            mel = np.load(path, mmap_mode='r', allow_pickle=True)
            if mel.shape[1] < 1000:  # pad if too short
                mel = np.pad(mel, ((0, 0), (0, 1000 - mel.shape[1])), 'constant')

            start = self.start_frame[index]
            mel = mel[:, start:start+100]

            mel = np.expand_dims(mel, -1)  # (nmel, nframe, 1)
            return mel, label

    def __len__(self):
        return len(self.data)


class FewShotDataloader():
    def __init__(self,
                 dataset,
                 nKnovel=5, # number of novel categories.
                 nKbase=-1, # number of base categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=15*5, # number of test examples for all the novel categories.
                 nTestBase=15*5, # number of test examples for all the base categories.
                 batch_size=1, # number of training episodes per batch.
                 num_workers=4,
                 epoch_size=2000, # number of batches per epoch.
                 poly='1'
                 ):

        self.dataset = dataset
        self.phase = self.dataset.phase
        max_possible_nKnovel = (self.dataset.num_cats_base if self.phase=='train'
                                else self.dataset.num_cats_novel)

        assert(nKnovel >= 0 and nKnovel < max_possible_nKnovel)
        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase

        if self.phase=='train' and nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel

        assert(nKbase >= 0 and nKbase <= max_possible_nKbase)
        self.nKbase = nKbase

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.is_eval_mode = (self.phase=='test') or (self.phase=='val')
        self.poly = poly

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).
        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.
        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert(cat_id in self.dataset.label2ind)
        # assert(len(self.dataset.label2ind[cat_id]) >= sample_size)
        # if a category does not have enough examples, sample duplicates
        if len(self.dataset.label2ind[cat_id]) < sample_size:
            print('sample duplicates')
            return random.sample(self.dataset.label2ind[cat_id]*(sample_size//len(self.dataset.label2ind[cat_id])+1), sample_size)

        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.
        Args:
            cat_set: string that specifies the set of categories from which
                categories will be sampled.
            sample_size: number of categories that will be sampled.
        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.
        """
        if cat_set=='base':
            labelIds = self.dataset.labelIds_base
        elif cat_set=='novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))
        assert(len(labelIds) >= sample_size)
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        """
        Samples `nKbase` number of base categories and `nKnovel` number of novel
        categories.
        Args:
            nKbase: number of base categories
            nKnovel: number of novel categories
        Returns:
            Kbase: a list of length 'nKbase' with the ids of the sampled base
                categories.
            Knovel: a list of lenght 'nKnovel' with the ids of the sampled novel
                categories.
        """
        if self.is_eval_mode:
            assert(nKnovel <= self.dataset.num_cats_novel)
            # sample from the set of base categories 'nKbase' number of base categories.
            Kbase = sorted(self.sampleCategories('base', nKbase))
            # sample from the set of novel categories 'nKnovel' number of novel categories.
            Knovel = sorted(self.sampleCategories('novel', nKnovel))
        else:
            # sample from the set of base categories 'nKnovel' + 'nKbase' number of categories.
            cats_ids = self.sampleCategories('base', nKnovel+nKbase)
            assert(len(cats_ids) == (nKnovel+nKbase))
            # Randomly pick 'nKnovel' number of fake novel categories and keep the rest as base categories.
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_test_examples_for_base_categories(self, Kbase, nTestBase, nKnovel):
        """
        Sample `nTestBase` number of images from the `Kbase` categories.
        Args:
            Kbase: a list of length `nKbase` with the ids of the categories from
                where the images will be sampled.
            nTestBase: the total number of images that will be sampled.
        Returns:
            Tbase: a list of length `nTestBase` with 2-element tuples. The 1st
                element of each tuple is the image id that was sampled and the
                2nd elemend is its category label (which is in the range
                [0, len(Kbase)-1]).
        """
        Tbase = []
        if len(Kbase) > 0:
            # Sample for each base category a number images such that the total
            # number sampled images of all categories to be equal to `nTestBase`.
            KbaseIndices = np.random.choice(np.arange(len(Kbase)), size=nTestBase, replace=True)
            KbaseIndices, NumImagesPerCategory = np.unique(KbaseIndices, return_counts=True)

            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                imd_ids = self.sampleImageIdsFrom(Kbase[Kbase_idx], sample_size=NumImages)

                for img_id in imd_ids:
                    labels = self.dataset[img_id][1]

                    label_multihot = torch.zeros((len(Kbase)+nKnovel,))  # add nKnovel slot for eval
                    for label in labels:
                        if label in Kbase:
                           label_multihot[Kbase.index(label)] = 1  # use indecies as labels, which change with sampled KBase in each episode

                    Tbase.append((img_id, label_multihot))

        assert(len(Tbase) == nTestBase)
        return Tbase

    def sample_train_and_test_examples_for_novel_categories(
            self, Knovel, nTestNovel, nExemplars, nKbase):
        """Samples train and test examples of the novel categories.
        Args:
    	    Knovel: a list with the ids of the novel categories.
            nTestNovel: the total number of test images that will be sampled
                from all the novel categories.
            nExemplars: the number of training examples per novel category that
                will be sampled.
            nKbase: the number of base categories. It is used as offset of the
                category index of each sampled image.
        Returns:
            Tnovel: a list of length `nTestNovel` with 2-element tuples. The
                1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [nKbase, nKbase + len(Knovel) - 1]).
            Exemplars: a list of length len(Knovel) * nExemplars of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(Knovel) - 1]).
        """

        if len(Knovel) == 0:
            return [], []

        nKnovel = len(Knovel)
        Tnovel = []
        Exemplars = []
        assert((nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(nTestNovel / nKnovel)

        for K in Knovel:
            if self.poly == '1':
                imd_ids = self.sampleImageIdsFrom(K, sample_size=(nEvalExamplesPerClass + nExemplars*5))  # oversample support examples if need poly=1
            else:
                imd_ids = self.sampleImageIdsFrom(K, sample_size=(nEvalExamplesPerClass + nExemplars))

            imds_tnovel = imd_ids[:nEvalExamplesPerClass]
            imds_exemplars = imd_ids[nEvalExamplesPerClass:]

            # encode multihot label for multilabel, include all novel labels
            for img_id in imds_tnovel:
                labels = self.dataset[img_id][1]
                label_multihot = torch.zeros((nKbase + len(Knovel),))

                for label in labels:
                    if label in Knovel:
                        label_multihot[nKbase + Knovel.index(label)] = 1
                Tnovel.append((img_id, label_multihot))

            if self.poly == '1':
                count = 0
                while count < nExemplars:
                    for img_id in imds_exemplars:
                        labels = self.dataset[img_id][1]
                        if len(labels) == 1:
                            label_multihot = torch.zeros((nKbase + len(Knovel),))
                            label_multihot[nKbase + Knovel.index(labels[0])] = 1
                            Exemplars.append((img_id, label_multihot))
                            count += 1
                            if count == nExemplars:
                                break
                    if count < nExemplars:
                        imds_exemplars_next = self.sampleImageIdsFrom(K, sample_size=(nExemplars)*5) # sample another batch of support candidates
                        imds_exemplars_next = [x for x in imds_exemplars_next if x not in imds_tnovel+imds_exemplars]
                        imds_exemplars = imds_exemplars_next
            else:
                for img_id in imds_exemplars:
                    labels = self.dataset[img_id][1]
                    label_multihot = torch.zeros((nKbase + len(Knovel),))

                    for label in labels:
                        if label in Knovel:
                            label_multihot[nKbase + Knovel.index(label)] = 1
                    Exemplars.append((img_id, label_multihot))

        assert(len(Tnovel) == nTestNovel)
        assert(len(Exemplars) == len(Knovel) * nExemplars)
        random.shuffle(Exemplars)

        return Tnovel, Exemplars

    def sample_episode(self):
        """Samples a training episode."""
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        Tbase = self.sample_test_examples_for_base_categories(Kbase, nTestBase, nKnovel)
        Tnovel, Exemplars = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, nTestNovel, nExemplars, nKbase)

        # concatenate the base and novel category examples.
        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel

        return Exemplars, Test, Kall, nKbase

    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.
        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).
        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = torch.stack([torch.from_numpy(self.dataset[img_idx][0]) for img_idx, _ in examples], dim=0)
        labels = torch.stack([label for _, label in examples])
        return images, labels

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        def load_function(iter_idx):
            Exemplars, Test, Kall, nKbase = self.sample_episode()
            Xt, Yt = self.createExamplesTensorData(Test)
            Kall = torch.LongTensor(Kall)
            if len(Exemplars) > 0:
                Xe, Ye = self.createExamplesTensorData(Exemplars)
                return Xe, Ye, Xt, Yt, Kall, nKbase
            else:
                return Xt, Yt, Kall, nKbase

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return (self.epoch_size / self.batch_size)
