from __future__ import print_function

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import utils

from . import Algorithm


def top1accuracy(output, target):
    _, pred = output.max(dim=1)
    pred = pred.view(-1)
    target = target.view(-1)
    accuracy = 100 * pred.eq(target).float().mean().item()
    return accuracy


def mAP(output, target, nKnovel):
    APs, Ps, Rs, Fs = [], [], [], []
    n_cl = target.shape[1]
    for cl in range(n_cl):
        cl_activation = target[:, cl]
        cl_activation = [int(x) for x in cl_activation]
        cl_pred = output[:, cl]  # prediction in cl
        cl_pred_binary = np.where(cl_pred > 0.5, 1, 0)
        if np.any(cl_activation):
            AP = average_precision_score(cl_activation, cl_pred)
            P = precision_score(cl_activation, cl_pred_binary)
            R = recall_score(cl_activation, cl_pred_binary)
            F = f1_score(cl_activation, cl_pred_binary)
            APs.append(AP)
            Ps.append(P)
            Rs.append(R)
            Fs.append(F)

    if n_cl == nKnovel or nKnovel == 0:
        return np.mean(APs), np.mean(Ps), np.mean(Rs), np.mean(Fs)

    APs_both, Ps_both, Rs_both, Fs_both = np.mean(APs), np.mean(Ps), np.mean(Rs), np.mean(Fs)
    APs_base, Ps_base, Rs_base, Fs_base = np.mean(APs[:-nKnovel]), np.mean(Ps[:-nKnovel]), np.mean(Rs[:-nKnovel]), np.mean(Fs[:-nKnovel]),
    APs_novel, Ps_novel, Rs_novel, Fs_novel = np.mean(APs[-nKnovel:]), np.mean(Ps[-nKnovel:]), np.mean(Rs[-nKnovel:]), np.mean(Fs[-nKnovel:])

    return APs_both, Ps_both, Rs_both, Fs_both, APs_base, Ps_base, Rs_base, Fs_base, APs_novel, Ps_novel, Rs_novel, Fs_novel


def activate_dropout_units(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.training = True


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


class FewShot(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)
        self.nKbase = torch.LongTensor()
        self.activate_dropout = (opt['activate_dropout'] if ('activate_dropout' in opt) else False)
        self.keep_best_model_metric_name = 'loss'
        self.opt = opt
        self.novelLoss = (opt['criterions']['loss']['novelLoss'] if ('novelLoss' in opt['criterions']['loss']) else False)
        self.openl3 = (True if ('openl3' in opt) else False)


    def set_tensors(self, batch):
        self.nKbase = self.dloader.nKbase
        self.nKnovel = self.dloader.nKnovel

        if self.nKnovel > 0:
            train_test_stage = 'fewshot'
            assert (len(batch) == 6)
            images_train, labels_train, images_test, labels_test, K, nKbase = batch
            self.nKbase = nKbase.squeeze()[0].item()
            self.tensors['images_train'].resize_(images_train.size()).copy_(images_train)
            self.tensors['labels_train'].resize_(labels_train.size()).copy_(labels_train)
            labels_train = self.tensors['labels_train']

            nKnovel = 1 + labels_train.max()  - self.nKbase

            self.tensors['labels_train_1hot'] = labels_train.float()
            self.tensors['images_test'].resize_(images_test.size()).copy_(images_test)
            self.tensors['labels_test'].resize_(labels_test.size()).copy_(labels_test)
            self.tensors['Kids'].resize_(K.size()).copy_(K)
        else:
            train_test_stage = 'base_classification'
            assert (len(batch) == 4)
            images_test, labels_test, K, nKbase = batch
            self.nKbase = nKbase.squeeze()[0].item()
            self.tensors['images_test'].resize_(images_test.size()).copy_(images_test)
            self.tensors['labels_test'].resize_(labels_test.size()).copy_(labels_test)
            self.tensors['Kids'].resize_(K.size()).copy_(K)

        return train_test_stage

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train):
        self.nKbase = self.dloader.nKbase
        self.nKnovel = self.dloader.nKnovel


        if self.nKnovel > 0:
            record = self.process_batch_fewshot_without_forgetting(batch, do_train=do_train)

        elif self.nKnovel == 0:
            record = self.process_batch_base_category_classification(batch, do_train=do_train)
        else:
            raise ValueError('Unexpected process type')

        return record

    def process_batch_base_category_classification(self, batch, do_train=True):
        if self.openl3:
            features_test, labels_test, K, nKbase = batch
            batch_size, num_test_examples, n_dim = features_test.size()
            new_batch_dim = batch_size * num_test_examples
            features_test = move_data_to_device(features_test, 'cuda')

            do_train_feat_model = do_train and self.optimizers['feat_model'] is not None
            if not do_train_feat_model:
                self.networks['feat_model'].eval()
                if do_train and self.activate_dropout:
                    # Activate the dropout units of the feature extraction model
                    # even if the feature extraction model is freezed (i.e., it is
                    # in eval mode).
                    activate_dropout_units(self.networks['feat_model'])

            features_test = self.networks['feat_model'](features_test.view(new_batch_dim, n_dim))
            features_test = features_test.view([batch_size, num_test_examples, ] + list(features_test.size()[1:]))

        else:
            images_test, labels_test, K, nKbase = batch
            images_test = move_data_to_device(images_test, 'cuda')

            do_train_feat_model = do_train and self.optimizers['feat_model'] is not None

            if not do_train_feat_model:
                self.networks['feat_model'].eval()
                if do_train and self.activate_dropout:
                    # Activate the dropout units of the feature extraction model
                    # even if the feature extraction model is freezed (i.e., it is
                    # in eval mode).
                    activate_dropout_units(self.networks['feat_model'])

            # ************************* FORWARD PHASE *******************************
            # *********** EXTRACT FEATURES FROM TRAIN & TEST IMAGES *****************
            batch_size, num_test_examples, channels, height, width = images_test.size()
            new_batch_dim = batch_size * num_test_examples
            features_test = self.networks['feat_model'](images_test.view(new_batch_dim, channels, height, width))
            features_test = features_test.view([batch_size, num_test_examples, ] + list(features_test.size()[1:]))

        # zero the gradients
        if do_train:
            self.optimizers['classifier'].zero_grad()
            if do_train_feat_model:
                self.optimizers['feat_model'].zero_grad()

        # Make sure that no gradients are backproagated to the feature
        # extractor when the feature extraction model is freezed.
        if (not do_train_feat_model) and do_train:
            features_test = features_test.detach()

        labels_test = move_data_to_device(labels_test, 'cuda')
        Kids = move_data_to_device(K, 'cuda')
        nKbase = nKbase[0].item()  # for pytorch 1.0.1
        Kbase = (None if (nKbase == 0) else Kids[:, :nKbase].contiguous())

        # ************************ APPLY CLASSIFIER *****************************
        cls_scores, att_coeficients, novel_weights = self.networks['classifier'](features_test=features_test,
                                                                                 Kbase_ids=Kbase)

        cls_scores = cls_scores.view(new_batch_dim, -1)
        labels_test = labels_test.view(new_batch_dim, -1)
        # ***********************************************************************
        # ************************** COMPUTE LOSSES *****************************
        loss_cls_all = self.criterions['loss'](cls_scores, labels_test)

        # add L1 reg
        if do_train and 'l1reg' in self.opt['criterions']['loss']:
            reg_lambda = self.opt['criterions']['loss']['l1reg']
            l1_reg = None
            for W in self.networks['classifier'].parameters():
                if l1_reg is None:
                    l1_reg = W.norm(p=1)
                else:
                    l1_reg = l1_reg + W.norm(p=1)

            # L1 reg on last fc layer in feature model
            if 'feat_reg' in self.opt['criterions']['loss']:
                l1_reg = l1_reg + self.networks['feat_model'].fc1.weight.norm(p=1)

            loss_cls_all = loss_cls_all + reg_lambda * l1_reg

        loss_record = {}
        loss_record['loss'] = loss_cls_all.data.cpu().numpy()
        loss_record['mAPBase'], loss_record['PBase'], loss_record['RBase'], loss_record['FBase'] = mAP((nn.Sigmoid()(cls_scores)).data.cpu().numpy(),
                                                                                                       labels_test.data.cpu().numpy(),                                                                                            self.nKnovel)
        # ***********************************************************************
        # ************************* BACKWARD PHASE ******************************
        if do_train:
            loss_cls_all.backward()
            self.optimizers['classifier'].step()
            if do_train_feat_model:
                self.optimizers['feat_model'].step()
        # ***********************************************************************

        return loss_record

    def process_batch_fewshot_without_forgetting(self, batch, do_train=True):
        if self.openl3:
            features_train_var, labels_train, features_test_var, labels_test, K, nKbase = batch
            batch_size, num_train_examples, n_dim = features_train_var.size()
            num_test_examples = features_test_var.size(1)
            features_train_var = move_data_to_device(features_train_var, 'cuda')
            features_test_var = move_data_to_device(features_test_var, 'cuda')

            do_train_feat_model = do_train and self.optimizers['feat_model'] is not None
            if (not do_train_feat_model):
                self.networks['feat_model'].eval()
                if do_train and self.activate_dropout:
                    # Activate the dropout units of the feature extraction model
                    # even if the feature extraction model is freezed (i.e., it is
                    # in eval mode).
                    activate_dropout_units(self.networks['feat_model'])

            features_train_var = self.networks['feat_model'](features_train_var.view(batch_size * num_train_examples, n_dim))
            features_test_var = self.networks['feat_model'](features_test_var.view(batch_size * num_test_examples, n_dim))
            features_train_var = features_train_var.view([batch_size, num_train_examples, ] + list(features_train_var.size()[1:]))
            features_test_var = features_test_var.view([batch_size, num_test_examples, ] + list(features_test_var.size()[1:]))

        else:
            images_train, labels_train, images_test, labels_test, K, nKbase = batch
            images_train = move_data_to_device(images_train, 'cuda')
            images_test = move_data_to_device(images_test, 'cuda')

            do_train_feat_model = do_train and self.optimizers['feat_model'] is not None
            if (not do_train_feat_model):
                self.networks['feat_model'].eval()
                if do_train and self.activate_dropout:
                    # Activate the dropout units of the feature extraction model
                    # even if the feature extraction model is freezed (i.e., it is
                    # in eval mode).
                    activate_dropout_units(self.networks['feat_model'])

            # ************************* FORWARD PHASE: ******************************
            # ************ EXTRACT FEATURES FROM TRAIN & TEST IMAGES ****************
            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)
            features_train_var = self.networks['feat_model'](images_train.view(batch_size * num_train_examples, channels, height, width))
            features_test_var = self.networks['feat_model'](images_test.view(batch_size * num_test_examples, channels, height, width))
            features_train_var = features_train_var.view([batch_size, num_train_examples, ] + list(features_train_var.size()[1:]))
            features_test_var = features_test_var.view([batch_size, num_test_examples, ] + list(features_test_var.size()[1:]))


        labels_train_1hot = move_data_to_device(labels_train, 'cuda')
        labels_test = move_data_to_device(labels_test, 'cuda')
        Kids = move_data_to_device(K, 'cuda')
        nKbase = nKbase[0].item()
        Kbase = (None if (nKbase == 0) else Kids[:, :nKbase].contiguous())

        if (not do_train_feat_model) and do_train:
            # Make sure that no gradients are backproagated to the feature
            # extractor when the feature extraction model is freezed.
            features_train_var = features_train_var.detach()
            features_test_var = features_test_var.detach()

        if do_train:  # zero the gradients
            self.optimizers['classifier'].zero_grad()
            if do_train_feat_model:  # unfreeze
                for name, param in self.networks['feat_model'].named_parameters():
                    if 'fc1' not in name:
                        param.requires_grad = False

                self.optimizers['feat_model'].zero_grad()

        loss_record = {}
        # ***********************************************************************
        # ************************ APPLY CLASSIFIER *****************************
        if self.nKbase > 0:
            cls_scores_var, att_coeficients, novel_weights = self.networks['classifier'](
                features_test=features_test_var,
                Kbase_ids=Kbase,
                features_train=features_train_var,
                labels_train=labels_train_1hot)
        else:
            cls_scores_var = self.networks['classifier'](
                features_test=features_test_var,
                features_train=features_train_var,
                labels_train=labels_train_1hot)

        cls_scores_novel = cls_scores_var[:, :, -self.nKnovel:]
        cls_scores_var = cls_scores_var.view(batch_size * num_test_examples, -1)
        labels_test = labels_test.view(batch_size * num_test_examples, -1)

        cls_scores_var_novel = cls_scores_novel.view(batch_size * num_test_examples, -1)
        labels_test_novel = labels_test[:, -self.nKnovel:]
        # ***********************************************************************

        # ************************* COMPUTE LOSSES ******************************
        loss_cls_all = self.criterions['loss'](cls_scores_var, labels_test)

        # optimize on novel loss only
        if self.novelLoss:
            loss_cls_novel = self.criterions['loss'](cls_scores_var_novel, labels_test_novel)
            loss_cls_all = loss_cls_novel

        # weight novel losses
        if self.opt['criterions']['loss']['opt'] == 'reduction=none':
            weights = torch.tensor([1]*self.nKbase + [self.nKbase/self.nKnovel]*self.nKnovel).cuda()
            loss_cls_all = (loss_cls_all * weights).mean()

        # add L1 regularization
        if do_train and 'l1reg' in self.opt['criterions']['loss']:
            reg_lambda = self.opt['criterions']['loss']['l1reg']
            l1_reg = torch.from_numpy(att_coeficients).norm(p=1)

            # l1_reg = None
            # for W in self.networks['classifier'].parameters():
            #     if l1_reg is None:
            #         l1_reg = W.norm(p=1)
            #     else:
            #         l1_reg = l1_reg + W.norm(p=1)

            loss_cls_all = loss_cls_all + reg_lambda * l1_reg

        # add L2 regularization
        if do_train and 'l2reg' in self.opt['criterions']['loss']:
            reg_lambda = self.opt['criterions']['loss']['l2reg']
            l2_reg = torch.from_numpy(att_coeficients).norm(p=2)
            loss_cls_all = loss_cls_all + reg_lambda * l2_reg


        loss_record['loss'] = loss_cls_all.data.cpu().numpy()  # for pytorch 1.0.1

        if self.nKbase > 0:
            loss_record['mAPBoth'], loss_record['PBoth'], loss_record['RBoth'], loss_record['FBoth'], \
            loss_record['mAPBase'], loss_record['PBase'], loss_record['RBase'], loss_record['FBase'], \
            loss_record['mAPNovel'], loss_record['PNovel'], loss_record['RNovel'], loss_record['FNovel'] \
                = mAP((nn.Sigmoid()(cls_scores_var)).data.cpu().numpy(), labels_test.data.cpu().numpy(), self.nKnovel)
        else:
            loss_record['mAPNovel'], loss_record['PNovel'], loss_record['RNovel'], loss_record['FNovel'] \
                = mAP((nn.Sigmoid()(cls_scores_var)).data.cpu().numpy(), labels_test.data.cpu().numpy(), self.nKnovel)
        # ***********************************************************************

        # ***********************************************************************
        # ************************* BACKWARD PHASE ******************************
        if do_train:
            loss_cls_all.backward()
            if do_train_feat_model:
                self.optimizers['feat_model'].step()
            self.optimizers['classifier'].step()
        # ***********************************************************************

        if (not do_train):
            if self.biter == 0: self.test_accuracies = {'loss': []}
            self.test_accuracies['loss'].append(
                loss_record['loss'])
            if self.biter == (self.bnumber - 1):
                # Compute the std and the confidence interval of the accuracy of
                # the novel categories.
                stds = np.std(np.array(self.test_accuracies['loss']), 0)
                ci95 = 1.96 * stds / np.sqrt(self.bnumber)
                loss_record['loss_std'] = stds
                loss_record['loss_cnf'] = ci95

        return loss_record
