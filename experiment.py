### execute this function to train and test the vae-model

from VAE import VariationalAutoencoder
import numpy as np
import pickle
import torch
import os
import argparse
import random
import torch
import math

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default="ESC-50")
parser.add_argument("--num_shots", type=int, default=0)
parser.add_argument("--generalized", type=str2bool, default="False")
parser.add_argument("--fold", type=str)
parser.add_argument("--folder", type=str)
args = parser.parse_args()

# These are for testing with fold 4, and doing 4-fold cross-validation on the remaining folds
if args.dataset == "ESC-50":
    args.val_classes = []
    if args.fold == "fold04":
        args.val_classes = [27, 46, 38, 3, 29, 48, 40, 31, 2, 35]
    elif args.fold == "fold14":
        args.val_classes = [22, 13, 39, 49, 32, 26, 42, 21, 19, 36]
    elif args.fold == "fold24":
        args.val_classes = [23, 41, 14, 24, 33, 30, 4, 17, 10, 45]
    elif args.fold == "fold34":
        args.val_classes = [47, 34, 20, 44, 25, 6, 7, 1, 28, 18]
    args.test_classes = [43, 5, 37, 12, 9, 0, 11, 8, 15, 16]
elif args.dataset == "FSC22":
    args.val_classes = []
    args.test_classes = [5, 7, 15, 17, 21, 23, 26]
    if args.fold != "test":
        args.val_classes = [6, 8, 9, 12, 13, 18, 22]

########################################
# The basic hyperparameters
########################################
hyperparameters = {
    "num_shots": 0,
    "device": "cuda",
    "model_specifics": {
        "cross_reconstruction": True,
        "name": "CADA",
        "distance": "wasserstein",
        "warmup": {
            "beta": {"factor": 0.25, "end_epoch": 93, "start_epoch": 0},
            "cross_reconstruction": {
                "factor": 2.37,
                "end_epoch": 75,
                "start_epoch": 21,
            },
            "distance": {"factor": 5.13, "end_epoch": 22, "start_epoch": 6},
            "M": 1,
        },
    },
    "lr_gen_model": 0.00015,
    "generalized": True,
    # "batch_size": 1,
    "batch_size": 52,
    "cls_batch_size": 32,
    "lr_cls": 0.001,
    # "epochs": 100,
    "epochs": 80,
    "loss": "l1",
    "auxiliary_data_source": "word2vec",
    "dataset": "ESC-50",
    "hidden_size_rule": {"audio": (1450, 665), "text": (1450, 665)},
    "latent_size": 64,
    "data_size": {"audio": 128, "text": 300},
    "val_classes": args.val_classes,
    "test_classes": args.test_classes,
}


# The training epochs for the final classifier, for early stopping,
# as determined on the validation spit

cls_train_steps = [
    {"dataset": "ESC-50", "num_shots": 0, "generalized": False, "cls_train_steps": 23},
    {"dataset": "FSC22", "num_shots": 0, "generalized": False, "cls_train_steps": 20},
]

##################################
# change some hyperparameters here
##################################
hyperparameters["dataset"] = args.dataset
hyperparameters["num_shots"] = args.num_shots
hyperparameters["generalized"] = args.generalized
hyperparameters["fold"] = args.fold
hyperparameters["folder"] = args.folder

hyperparameters["cls_train_steps"] = [
    x["cls_train_steps"]
    for x in cls_train_steps
    if all(
        [
            hyperparameters["dataset"] == x["dataset"],
            hyperparameters["num_shots"] == x["num_shots"],
            hyperparameters["generalized"] == x["generalized"],
        ]
    )
][0]

print("***")
print(hyperparameters["cls_train_steps"])
if hyperparameters["generalized"]:
    if hyperparameters["num_shots"] == 0:
        hyperparameters["samples_per_class"] = {
            "ESC-50": (100, 0, 200, 0),
            "FSC22": (100, 0, 200, 0),
        }
    else:
        hyperparameters["samples_per_class"] = {
            "ESC-50": (100, 0, 100, 100),
            "FSC22": (100, 0, 100, 100),
        }
else:
    if hyperparameters["num_shots"] == 0:
        hyperparameters["samples_per_class"] = {
            "ESC-50": (0, 0, 100, 0),
            "FSC22": (0, 0, 100, 0),
        }
    else:
        hyperparameters["samples_per_class"] = {
            "ESC-50": (0, 0, 100, 100),
            "FSC22": (0, 0, 100, 100),
        }


"""
########################################
### Load model
########################################
saved_state = torch.load('./saved_models/CADA_trained.pth.tar')
model.load_state_dict(saved_state['state_dict'])
for d in model.all_data_sources_without_duplicates:
    model.encoder[d].load_state_dict(saved_state['encoder'][d])
    model.decoder[d].load_state_dict(saved_state['decoder'][d])
########################################
"""

# Train several times randomly
seeds = random.sample(range(1000), 10)
n_trials = len(seeds)
accuracies = []

for trial, seed in enumerate(seeds):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = VariationalAutoencoder(hyperparameters)
    model.to(hyperparameters["device"])

    losses = model.train_vae()

    u, s, h, history = model.train_classifier()

    if hyperparameters["generalized"] == True:
        acc = [hi[2] for hi in history]
    elif hyperparameters["generalized"] == False:
        acc = [hi[1] for hi in history]

    accuracies.append(acc[-1])

print(accuracies)
mean = 0
for accuracy in accuracies:
    mean += accuracy
mean /= len(accuracies)

std = 0
for accuracy in accuracies:
    std += (accuracy - mean) ** 2
std = math.sqrt(std / len(accuracies))

print("Accuracy mean is {} and std is {}".format(mean, std))
