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

elif args.dataset == "UrbanSound8k":
    # args.train_classes = [0, 1, 2, 4, 5, 7, 8]
    args.val_classes = [3, 6, 9]
    args.test_classes = [3, 6, 9]

elif args.dataset == "TAU2019":
    # args.train_classes = [2, 3, 4, 5, 7, 8, 9]
    args.val_classes = [0, 1, 6]
    args.test_classes = [0, 1, 6]

elif args.dataset == "GTZAN":
    # args.train_classes = [0, 1, 2, 6, 7, 8, 9]
    args.val_classes = [3, 4, 5]
    args.test_classes = [3, 4, 5]

elif args.dataset == "ARCA23K-FSD":
    # args.test_classes = ['Female_singing', 'Wind_chime', 'Dishes_and_pots_and_pans', 'Scratching_(performance_technique)', 'Crying_and_sobbing', 'Waves_and_surf', 'Screaming', 'Bark', 'Camera', 'Organ']
    args.test_classes = np.linspace(60, 69, 10)
    args.val_classes = np.linspace(60, 69, 10)
    if args.fold == "fold0":
        # args.val_classes = ['Crash_cymbal', 'Run', 'Zipper_(clothing)', 'Acoustic_guitar', 'Gong', 'Knock', 'Train', 'Crack', 'Cough', 'Cricket']
        args.val_classes = np.linspace(0, 9, 10)
    elif args.fold == "fold1":
        # args.val_classes = ['Electric_guitar', 'Chewing_and_mastication', 'Keys_jangling', 'Female_speech_and_woman_speaking', 'Crumpling_and_crinkling', 'Skateboard', 'Computer_keyboard', 'Bass_guitar', 'Stream', 'Toilet_flush']
        args.val_classes = np.linspace(10, 19, 10)
    elif args.fold == "fold2":
        # args.val_classes = ['Tap', 'Water_tap_and_faucet', 'Squeak', 'Snare_drum', 'Finger_snapping', 'Walk_and_footsteps', 'Meow', 'Rattle_(instrument)', 'Bowed_string_instrument', 'Sawing']
        args.val_classes = np.linspace(20, 29, 10)
    elif args.fold == "fold3":
        # args.val_classes = ['Rattle', 'Slam', 'Whoosh_and_swoosh_and_swish', 'Hammer', 'Fart', 'Harp', 'Coin_(dropping)', 'Printer', 'Boom', 'Giggle']
        args.val_classes = np.linspace(30, 39, 10)
    elif args.fold == "fold4":
        # args.val_classes = ['Clapping', 'Crushing', 'Livestock_and_farm_animals_and_working_animals', 'Scissors', 'Writing', 'Wind', 'Crackle', 'Tearing', 'Piano', 'Microwave_oven']
        args.val_classes = np.linspace(40, 49, 10)
    elif args.fold == "fold5":
        # args.val_classes = ['Trumpet', 'Wind_instrument_and_woodwind_instrument', 'Child_speech_and_kid_speaking', 'Drill', 'Thump_and_thud', 'Drawer_open_or_close', 'Male_speech_and_man_speaking', 'Gunshot_and_gunfire', 'Burping_and_eructation', 'Splash_and_splatter']
        args.val_classes = np.linspace(50, 59, 10)


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
    "dataset": args.dataset,
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
    {"dataset": "UrbanSound8k", "num_shots": 0, "generalized": False, "cls_train_steps": 20},
    {"dataset": "TAU2019", "num_shots": 0, "generalized": False, "cls_train_steps": 30},
    {"dataset": "GTZAN", "num_shots": 0, "generalized": False, "cls_train_steps": 20},
    {"dataset": "ARCA23K-FSD", "num_shots": 0, "generalized": False, "cls_train_steps": 50},
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
            "UrbanSound8k": (0, 0, 873, 0),
            "TAU2019": (0, 0, 1440, 0),
            "GTZAN": (0, 0, 100, 0),
            "ARCA23K-FSD": (0, 0, 338, 0),
        }
    else:
        hyperparameters["samples_per_class"] = {
            "ESC-50": (0, 0, 100, 100),
            "FSC22": (0, 0, 100, 100),
            "UrbanSound8k": (0, 0, 873, 873),
            "TAU2019": (0, 0, 1440, 1440),
            "GTZAN": (0, 0, 100, 100),
            "ARCA23K-FSD": (0, 0, 338, 338),
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
