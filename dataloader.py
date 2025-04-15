import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pickle
import random
import copy
import re


def map_label(label, classes):
    mapped_label = torch.LongTensor(len(label))
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


class DATA_LOADER(object):
    def __init__(
        self,
        dataset,
        fold,
        data_folder,
        aux_datasource,
        batch_size,
        test_classes,
        val_classes,
        device="cuda",
    ):

        if dataset == "ESC-50":
            data_path = "{}/{}.pickle".format(data_folder, fold)
        elif dataset == "FSC22":
            data_path = data_folder

        self.test_classes = test_classes
        self.val_classes = val_classes

        self.data_path = data_path
        self.batch_size = batch_size
        self.device = device
        self.fold = fold
        self.dataset = dataset
        self.auxiliary_data_source = aux_datasource

        self.all_data_sources = ["audio_features"] + [self.auxiliary_data_source]

        self.read_dataset()
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def next_batch(self, batch_size):
        #####################################################################
        # gets batch from train_feature = 7057 samples from 150 train classes
        #####################################################################
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.data["train_seen"]["audio_features"][idx]
        batch_label = self.data["train_seen"]["labels"][idx]
        batch_att = self.data["train_seen"][self.auxiliary_data_source][idx]
        return batch_label, [batch_feature, batch_att]

    def __getitem__(self, idx):
        #####################################################################
        # gets batch from train_feature = 7057 samples from 150 train classes
        #####################################################################
        idx = torch.randperm(self.ntrain)[0 : self.batch_size].long()
        batch_feature = self.data["train_seen"]["audio_features"][idx]
        batch_label = self.data["train_seen"]["labels"][idx]
        batch_att = self.data["train_seen"][self.auxiliary_data_source][idx]
        return batch_label, [batch_feature, batch_att]

    def read_dataset(self):
        # LOAD DATA
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        all_labels = np.array(data["labels"])
        all_features = np.array([list(d.to("cpu")[0]) for d in data["features"]])
        all_auxiliary = np.array(data["auxiliary"])

        train_labels = []
        train_features = []
        train_auxiliary = []

        val_labels = []
        val_features = []
        val_auxiliary = []

        for i in range(len(all_labels)):
            if self.fold == "test":
                if all_labels[i] in self.test_classes:
                    val_labels.append(all_labels[i])
                    val_features.append(all_features[i])
                    val_auxiliary.append(all_auxiliary[i])
                else:
                    train_labels.append(all_labels[i])
                    train_features.append(all_features[i])
                    train_auxiliary.append(all_auxiliary[i])
            else:
                if all_labels[i] in self.test_classes:
                    continue
                elif all_labels[i] in self.val_classes:
                    val_labels.append(all_labels[i])
                    val_features.append(all_features[i])
                    val_auxiliary.append(all_auxiliary[i])
                else:
                    train_labels.append(all_labels[i])
                    train_features.append(all_features[i])
                    train_auxiliary.append(all_auxiliary[i])

        unique_val_labels = list(set(val_labels))
        # for i in range(len(val_labels)):
        #     val_labels[i] = unique_val_labels.index(val_labels[i])

        unique_train_labels = list(set(train_labels))
        # for i in range(len(train_labels)):
        #     train_labels[i] = unique_train_labels.index(train_labels[i])        

        # print(val_labels)
        

        # Create a dictionary to map unique val labels to their corresponding auxiliary vectors
        val_auxiliary_dict = {label: val_auxiliary[val_labels.index(label)] for label in unique_val_labels}
        unique_val_aux = [val_auxiliary_dict[label] for label in unique_val_labels]

        # Create a dictionary to map unique train labels to their corresponding auxiliary vectors
        train_auxiliary_dict = {label: train_auxiliary[train_labels.index(label)] for label in unique_train_labels}
        unique_train_aux = [train_auxiliary_dict[label] for label in unique_train_labels]

        

        # Translate to CADA-VAE format
        seen_labels = train_labels
        seen_aux = train_auxiliary
        seen_features = train_features

        unseen_features = val_features

        # A list of all indices, to be shuffles to make a test/train fold for seen data
        indices = [i for i in range(len(seen_labels))]
        random.shuffle(indices)
        test_num = int(len(indices) * 0.2)  # 20-80 fold on test-train

        # Partition the seen data into test and train
        seen_test_labels = (
            torch.tensor([seen_labels[i] for i in indices[:test_num]])
            .long()
            .to(self.device)
        )
        seen_test_aux = (
            torch.tensor([seen_aux[i] for i in indices[:test_num]])
            .float()
            .to(self.device)
        )
        seen_test_features = (
            torch.tensor([seen_features[i] for i in indices[:test_num]])
            .float()
            .to(self.device)
        )

        seen_train_labels = (
            torch.tensor([seen_labels[i] for i in indices[test_num:]])
            .long()
            .to(self.device)
        )
        seen_train_aux = (
            torch.tensor([seen_aux[i] for i in indices[test_num:]])
            .float()
            .to(self.device)
        )
        seen_train_features = (
            torch.tensor([seen_features[i] for i in indices[test_num:]])
            .float()
            .to(self.device)
        )

        # SET CLASS DATA

        # Set the seen/unseen classes
        self.seenclasses = torch.tensor(unique_train_labels).long().to(self.device)
        self.novelclasses = torch.tensor(unique_val_labels).long().to(self.device)

        # Set the number of training and testing samples and classes
        self.ntrain = len(seen_train_features)
        self.ntrain_class = len(unique_train_labels)
        self.ntest_class = len(unique_val_labels)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.train_mapped_label = map_label(seen_train_labels, self.seenclasses)

        self.data = {}
        self.data["train_seen"] = {}
        self.data["train_seen"]["audio_features"] = seen_train_features
        self.data["train_seen"]["labels"] = seen_train_labels
        self.data["train_seen"][self.auxiliary_data_source] = seen_train_aux

        self.data["train_unseen"] = {}
        self.data["train_unseen"]["audio_features"] = None
        self.data["train_unseen"]["labels"] = None

        self.data["test_seen"] = {}
        self.data["test_seen"]["audio_features"] = seen_test_features
        self.data["test_seen"]["labels"] = seen_test_labels

        self.data["test_unseen"] = {}
        self.data["test_unseen"]["audio_features"] = (
            torch.tensor(unseen_features).float().to(self.device)
        )
        self.data["test_unseen"][self.auxiliary_data_source] = (
            torch.tensor(val_auxiliary).float().to(self.device)
        )
        self.data["test_unseen"]["labels"] = (
            torch.tensor(val_labels).long().to(self.device)
        )

        self.novelclass_aux_data = (
            torch.tensor(unique_val_aux).float().to(self.device)
        )
        self.seenclass_aux_data = (
            torch.tensor(unique_train_aux).float().to(self.device)
        )

    def transfer_features(self, n, num_queries="num_features"):
        print("size before")
        print(self.data["test_unseen"]["audio_features"].size())
        print(self.data["train_seen"]["audio_features"].size())

        print("o" * 100)
        print(self.data["test_unseen"].keys())
        for i, s in enumerate(self.novelclasses):

            features_of_that_class = self.data["test_unseen"]["audio_features"][
                self.data["test_unseen"]["labels"] == s, :
            ]

            if "attributes" == self.auxiliary_data_source:
                attributes_of_that_class = self.data["test_unseen"]["attributes"][
                    self.data["test_unseen"]["labels"] == s, :
                ]
                use_att = True
            else:
                use_att = False

            word2vec_of_that_class = self.data["test_unseen"]["word2vec"][
                self.data["test_unseen"]["labels"] == s, :
            ]

            num_features = features_of_that_class.size(0)

            indices = torch.randperm(num_features)

            if num_queries != "num_features":

                indices = indices[: n + num_queries]

            print(features_of_that_class.size())

            if i == 0:
                new_train_unseen = features_of_that_class[indices[:n], :]

                if use_att:
                    new_train_unseen_att = attributes_of_that_class[indices[:n], :]

                new_train_unseen_w2v = word2vec_of_that_class[indices[:n], :]
                new_train_unseen_label = s.repeat(n)
                new_test_unseen = features_of_that_class[indices[n:], :]
                new_test_unseen_label = s.repeat(len(indices[n:]))

            else:
                new_train_unseen = torch.cat(
                    (new_train_unseen, features_of_that_class[indices[:n], :]), dim=0
                )
                new_train_unseen_label = torch.cat(
                    (new_train_unseen_label, s.repeat(n)), dim=0
                )

                new_test_unseen = torch.cat(
                    (new_test_unseen, features_of_that_class[indices[n:], :]), dim=0
                )
                new_test_unseen_label = torch.cat(
                    (new_test_unseen_label, s.repeat(len(indices[n:]))), dim=0
                )

                if use_att:
                    new_train_unseen_att = torch.cat(
                        (
                            new_train_unseen_att,
                            attributes_of_that_class[indices[:n], :],
                        ),
                        dim=0,
                    )

                new_train_unseen_w2v = torch.cat(
                    (new_train_unseen_w2v, word2vec_of_that_class[indices[:n], :]),
                    dim=0,
                )

        print("new_test_unseen.size(): ", new_test_unseen.size())
        print("new_test_unseen_label.size(): ", new_test_unseen_label.size())
        print("new_train_unseen.size(): ", new_train_unseen.size())
        # print('new_train_unseen_att.size(): ', new_train_unseen_att.size())
        print("new_train_unseen_label.size(): ", new_train_unseen_label.size())
        print(">> num novel classes: " + str(len(self.novelclasses)))

        #######
        ##
        #######

        self.data["test_unseen"]["audio_features"] = copy.deepcopy(new_test_unseen)
        # self.data['train_seen']['audio_features']  = copy.deepcopy(new_train_seen)

        self.data["test_unseen"]["labels"] = copy.deepcopy(new_test_unseen_label)
        # self.data['train_seen']['labels']  = copy.deepcopy(new_train_seen_label)

        self.data["train_unseen"]["audio_features"] = copy.deepcopy(new_train_unseen)
        self.data["train_unseen"]["labels"] = copy.deepcopy(new_train_unseen_label)
        self.ntrain_unseen = self.data["train_unseen"]["audio_features"].size(0)

        if use_att:
            self.data["train_unseen"]["attributes"] = copy.deepcopy(
                new_train_unseen_att
            )

        self.data["train_unseen"]["word2vec"] = copy.deepcopy(new_train_unseen_w2v)

        ####
        self.data["train_seen_unseen_mixed"] = {}
        self.data["train_seen_unseen_mixed"]["audio_features"] = torch.cat(
            (
                self.data["train_seen"]["audio_features"],
                self.data["train_unseen"]["audio_features"],
            ),
            dim=0,
        )
        self.data["train_seen_unseen_mixed"]["labels"] = torch.cat(
            (self.data["train_seen"]["labels"], self.data["train_unseen"]["labels"]),
            dim=0,
        )

        self.ntrain_mixed = self.data["train_seen_unseen_mixed"]["audio_features"].size(
            0
        )

        if use_att:
            self.data["train_seen_unseen_mixed"]["attributes"] = torch.cat(
                (
                    self.data["train_seen"]["attributes"],
                    self.data["train_unseen"]["attributes"],
                ),
                dim=0,
            )

        self.data["train_seen_unseen_mixed"]["word2vec"] = torch.cat(
            (
                self.data["train_seen"]["word2vec"],
                self.data["train_unseen"]["word2vec"],
            ),
            dim=0,
        )

    def __len__(self):
        return int(self.ntrain / self.batch_size)


# d = DATA_LOADER()
