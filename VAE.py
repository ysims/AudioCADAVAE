import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import copy
from models import Encoder, Decoder, weights_init
import classifier
from dataloader import DATA_LOADER as dataloader


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction = nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


# Combine encoder and decoder
class VariationalAutoencoder(nn.Module):
    def __init__(self, hyperparameters):
        super(VariationalAutoencoder, self).__init__()

        self.device = hyperparameters["device"]
        self.DATASET = hyperparameters["dataset"]
        self.data_size = hyperparameters["data_size"]
        self.auxiliary_data_source = "word2vec"
        self.num_shots = hyperparameters["num_shots"]
        self.latent_size = hyperparameters["latent_size"]
        self.batch_size = hyperparameters["batch_size"]
        self.hidden_size_rule = hyperparameters["hidden_size_rule"]
        self.warmup = hyperparameters["model_specifics"]["warmup"]
        self.generalized = hyperparameters["generalized"]
        self.classifier_batch_size = hyperparameters["cls_batch_size"]
        self.audio_seen_samples = hyperparameters["samples_per_class"][self.DATASET][0]
        self.text_seen_samples = hyperparameters["samples_per_class"][self.DATASET][1]
        self.text_unseen_samples = hyperparameters["samples_per_class"][self.DATASET][2]
        self.audio_unseen_samples = hyperparameters["samples_per_class"][self.DATASET][
            3
        ]
        self.reco_loss_function = hyperparameters["loss"]
        self.nepoch = hyperparameters["epochs"]
        self.lr_cls = hyperparameters["lr_cls"]
        self.cross_reconstruction = hyperparameters["model_specifics"][
            "cross_reconstruction"
        ]
        self.cls_train_epochs = hyperparameters["cls_train_steps"]
        self.dataset = dataloader(
            self.DATASET,
            hyperparameters["fold"],
            hyperparameters["folder"],
            copy.deepcopy(self.auxiliary_data_source),
            test_classes=hyperparameters["test_classes"],
            val_classes=hyperparameters["val_classes"],
            device=self.device,
            batch_size=self.batch_size,
        )

        if self.DATASET == "ESC-50":
            self.num_classes = 50
            self.num_novel_classes = 10
        elif self.DATASET == "FSC22":
            self.num_classes = 27
            self.num_novel_classes = 7
        else:
            print("Unsupported dataset")

        self.encoder_audio = Encoder(
            self.latent_size,
            self.hidden_size_rule["audio"],
            self.data_size["audio"],
            self.device,
        )
        self.decoder_audio = Decoder(
            self.latent_size,
            self.hidden_size_rule["audio"],
            self.data_size["audio"],
            self.device,
        )
        self.encoder_text = Encoder(
            self.latent_size,
            self.hidden_size_rule["text"],
            self.data_size["text"],
            self.device,
        )
        self.decoder_text = Decoder(
            self.latent_size,
            self.hidden_size_rule["text"],
            self.data_size["text"],
            self.device,
        )

        parameters_to_optimize = list(self.parameters())

        parameters_to_optimize += list(self.encoder_audio.parameters())
        parameters_to_optimize += list(self.decoder_audio.parameters())
        parameters_to_optimize += list(self.encoder_text.parameters())
        parameters_to_optimize += list(self.decoder_text.parameters())

        self.optimizer = optim.Adam(
            parameters_to_optimize,
            lr=hyperparameters["lr_gen_model"],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=True,
        )

        if self.reco_loss_function == "l2":
            self.reconstruction_criterion = nn.MSELoss(reduction="sum")

        elif self.reco_loss_function == "l1":
            self.reconstruction_criterion = nn.L1Loss(reduction="sum")

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())
            return mu + sigma * eps
        else:
            return mu

    def forward(self):
        pass

    def map_label(self, label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label == classes[i]] = i

        return mapped_label

    # Takes an audio vector and text vector (auxiliary data),
    # runs them through the encoder and decoder,
    # calculates the loss and calls the optimiser
    def trainstep(self, audio_vector, text_vector, current_epoch):

        ##############################################
        # Encode image features and additional
        # features
        ##############################################
        mu_audio, logvar_audio = self.encoder_audio(audio_vector)
        z_from_audio = self.reparameterize(mu_audio, logvar_audio)

        mu_text, logvar_text = self.encoder_text(text_vector)
        z_from_text = self.reparameterize(mu_text, logvar_text)

        ##############################################
        # Reconstruct inputs
        ##############################################

        audio_from_audio = self.decoder_audio(z_from_audio)
        text_from_text = self.decoder_text(z_from_text)

        reconstruction_loss = self.reconstruction_criterion(
            audio_from_audio, audio_vector
        ) + self.reconstruction_criterion(text_from_text, text_vector)

        ##############################################
        # Cross Reconstruction Loss
        ##############################################
        audio_from_text = self.decoder_audio(z_from_text)
        text_from_audio = self.decoder_text(z_from_audio)

        cross_reconstruction_loss = self.reconstruction_criterion(
            audio_from_text, audio_vector
        ) + self.reconstruction_criterion(text_from_audio, text_vector)

        ##############################################
        # KL-Divergence
        ##############################################

        KLD = (
            0.5 * torch.sum(1 + logvar_text - mu_text.pow(2) - logvar_text.exp())
        ) + (0.5 * torch.sum(1 + logvar_audio - mu_audio.pow(2) - logvar_audio.exp()))

        ##############################################
        # Distribution Alignment
        ##############################################
        distance = torch.sqrt(
            torch.sum((mu_audio - mu_text) ** 2, dim=1)
            + torch.sum(
                (torch.sqrt(logvar_audio.exp()) - torch.sqrt(logvar_text.exp())) ** 2,
                dim=1,
            )
        )

        distance = distance.sum()

        ##############################################
        # Scale the loss terms according to the warmup
        # schedule
        ##############################################
        current_epoch_mod = current_epoch % int(self.nepoch / self.warmup["M"])

        # Calculate cross-reconstruction
        cr_start = self.warmup["cross_reconstruction"]["start_epoch"] / self.warmup["M"]
        cr_end = self.warmup["cross_reconstruction"]["end_epoch"] / self.warmup["M"]
        cr_factor = self.warmup["cross_reconstruction"]["factor"] / self.warmup["M"]

        f1 = float(current_epoch_mod - cr_start) / float(cr_end - cr_start) * cr_factor

        cross_reconstruction_factor = torch.cuda.FloatTensor(
            [min(max(f1, 0), cr_factor)]
        )

        # Calculate beta
        b_start = self.warmup["beta"]["start_epoch"] / self.warmup["M"]
        b_end = self.warmup["beta"]["end_epoch"] / self.warmup["M"]
        b_factor = self.warmup["beta"]["factor"] / self.warmup["M"]

        f2 = (float(current_epoch_mod - b_start) / float(b_end - b_start)) * b_factor
        beta = torch.cuda.FloatTensor([min(max(f2, 0), b_factor)])

        # Calculate distance factor
        d_start = self.warmup["distance"]["start_epoch"] / self.warmup["M"]
        d_end = self.warmup["distance"]["end_epoch"] / self.warmup["M"]
        d_factor = self.warmup["distance"]["factor"] / self.warmup["M"]

        f3 = (float(current_epoch_mod - d_start) / float(d_end - d_start)) * d_factor
        distance_factor = torch.cuda.FloatTensor([min(max(f3, 0), d_factor)])

        ##############################################
        # Put the loss together and call the optimizer
        ##############################################

        self.optimizer.zero_grad()

        loss = reconstruction_loss - beta * KLD

        if cross_reconstruction_loss > 0:
            loss += cross_reconstruction_factor * cross_reconstruction_loss

        if distance_factor > 0:
            loss += distance_factor * distance

        loss.backward()

        self.optimizer.step()

        return loss.item()

    # Trains the VAE
    def train_vae(self):
        # List of the losses calculated over training
        losses = []

        # Load the data
        self.dataloader = data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        ) 
        self.dataset.novelclasses = self.dataset.novelclasses.long().cuda()
        self.dataset.seenclasses = self.dataset.seenclasses.long().cuda()

        # Set to training mode
        self.train()
        self.reparameterize_with_noise = True

        print("Train for reconstruction")
        # Iterated over each epoch and iteration within epoch
        for epoch in range(0, self.nepoch):
            for iteration, _ in enumerate(
                range(0, self.dataset.ntrain, self.batch_size)
            ):
                # Get batch from the dataloader
                label, data_batch = self.dataset.next_batch(self.batch_size)
                label = label.long().to(self.device)

                # Configure batch
                for j in range(len(data_batch)):
                    data_batch[j] = data_batch[j].to(self.device)
                    data_batch[j].requires_grad = False

                # Do a training step with this batch
                loss = self.trainstep(data_batch[0], data_batch[1], epoch)

                # Print update every 50 iterations
                if iteration % 50 == 0:
                    print(
                        "epoch "
                        + str(epoch)
                        + " | iter "
                        + str(iteration)
                        + "\t"
                        + " | loss "
                        + str(loss)[:5]
                    )

                # Add loss every 50 iterations, but skip the first
                if iteration % 50 == 0 and iteration > 0:
                    losses.append(loss)

        # Turn into evaluation mode:
        self.encoder_audio.eval()
        self.decoder_audio.eval()
        self.encoder_text.eval()
        self.decoder_text.eval()

        return losses

    # Not sure yet
    def train_classifier(self, show_plots=False):

        if self.num_shots > 0:
            print(
                "================  transfer features from test to train =================="
            )
            self.dataset.transfer_features(self.num_shots, num_queries="num_features")

        history = []  # stores accuracies

        # Get the classes
        cls_seenclasses = self.dataset.seenclasses
        cls_novelclasses = self.dataset.novelclasses

        # Get the features and labels of the training data
        train_seen_feat = self.dataset.data["train_seen"]["audio_features"]
        train_seen_label = self.dataset.data["train_seen"]["labels"]

        # Get the auxiliary data for all classes
        novelclass_aux_data = (
            self.dataset.novelclass_aux_data
        )  # access as novelclass_aux_data['audio_features'], novelclass_aux_data['attributes']
        seenclass_aux_data = self.dataset.seenclass_aux_data

        # Get the labels in desired format and on device
        novel_corresponding_labels = cls_novelclasses.long().to(self.device)
        seen_corresponding_labels = cls_seenclasses.long().to(self.device)

        # Testing features and labels for both
        # The audio_features for testing the classifier are loaded here
        novel_test_feat = self.dataset.data["test_unseen"][
            "audio_features"
        ]  # self.dataset.test_novel_feature.to(self.device)
        seen_test_feat = self.dataset.data["test_seen"][
            "audio_features"
        ]  # self.dataset.test_seen_feature.to(self.device)
        test_seen_label = self.dataset.data["test_seen"][
            "labels"
        ]  # self.dataset.test_seen_label.to(self.device)
        test_novel_label = self.dataset.data["test_unseen"][
            "labels"
        ]  # self.dataset.test_novel_label.to(self.device)

        # Training features and labels for unseen
        train_unseen_feat = self.dataset.data["train_unseen"]["audio_features"]
        train_unseen_label = self.dataset.data["train_unseen"]["labels"]

        # If in ZSL mode, set up the label numbers correctly for just unseen
        if self.generalized == False:
            novel_corresponding_labels = self.map_label(
                novel_corresponding_labels, novel_corresponding_labels
            )

            if self.num_shots > 0:
                # not generalized and at least 1 shot means normal FSL setting (use only unseen classes)
                train_unseen_label = self.map_label(
                    train_unseen_label, cls_novelclasses
                )

            test_novel_label = self.map_label(test_novel_label, cls_novelclasses)

            # map cls novelclasses last
            cls_novelclasses = self.map_label(cls_novelclasses, cls_novelclasses)

        if self.generalized:
            # If GZSL, use all the classes
            print("mode: gzsl")
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
        else:
            # If ZSL, use only the unseen classes
            print("mode: zsl")
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_novel_classes)

        # Initialise the weights of the softmax classifier
        clf.apply(weights_init)

        with torch.no_grad():

            ####################################
            # preparing the test set
            # convert raw test data into z vectors
            ####################################

            self.reparameterize_with_noise = False

            mu1, var1 = self.encoder_audio(novel_test_feat)
            test_novel_X = self.reparameterize(mu1, var1).to(self.device).data
            test_novel_Y = test_novel_label.to(self.device)

            mu2, var2 = self.encoder_audio(seen_test_feat)
            test_seen_X = self.reparameterize(mu2, var2).to(self.device).data
            test_seen_Y = test_seen_label.to(self.device)

            ####################################
            # preparing the train set:
            # chose n random image features per
            # class. If n exceeds the number of
            # image features per class, duplicate
            # some. Next, convert them to
            # latent z features.
            ####################################

            self.reparameterize_with_noise = True

            # Real vector, label for that vector, number of samples to get
            def sample_train_data_on_sample_per_class_basis(
                features, label, sample_per_class
            ):
                sample_per_class = int(sample_per_class)

                if sample_per_class == 0 or len(label) == 0:
                    return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])

                classes = label.unique()  # get all the classes

                # i is the index of the class and s is the class
                for i, s in enumerate(classes):
                    # Get all the vectors that are labelled with this class name
                    class_features = features[
                        label == s, :
                    ]  # order of features and labels must coincide
                    # if number of selected features is smaller than the number of features we want per class:
                    multiplier = (
                        torch.ceil(
                            torch.cuda.FloatTensor(
                                [max(1, sample_per_class / class_features.size(0))]
                            )
                        )
                        .long()
                        .item()
                    )

                    class_features = class_features.repeat(multiplier, 1)

                    if i == 0:
                        features_to_return = class_features[:sample_per_class, :]
                        labels_to_return = s.repeat(sample_per_class)
                    else:
                        features_to_return = torch.cat(
                            (features_to_return, class_features[:sample_per_class, :]),
                            dim=0,
                        )
                        labels_to_return = torch.cat(
                            (labels_to_return, s.repeat(sample_per_class)), dim=0
                        )

                return features_to_return, labels_to_return

            # some of the following might be empty tensors if the specified number of
            # samples is zero :

            (
                audio_seen_feat,
                audio_seen_label,
            ) = sample_train_data_on_sample_per_class_basis(
                train_seen_feat, train_seen_label, self.audio_seen_samples
            )

            (
                audio_unseen_feat,
                audio_unseen_label,
            ) = sample_train_data_on_sample_per_class_basis(
                train_unseen_feat, train_unseen_label, self.audio_unseen_samples
            )

            (
                att_unseen_feat,
                att_unseen_label,
            ) = sample_train_data_on_sample_per_class_basis(
                novelclass_aux_data,
                novel_corresponding_labels,
                self.text_unseen_samples,
            )

            att_seen_feat, att_seen_label = sample_train_data_on_sample_per_class_basis(
                seenclass_aux_data, seen_corresponding_labels, self.text_seen_samples
            )

            def convert_datapoints_to_z(features, encoder):
                if features.size(0) != 0:
                    mu_, logvar_ = encoder(features)
                    z = self.reparameterize(mu_, logvar_)
                    return z
                else:
                    return torch.cuda.FloatTensor([])

            z_seen_audio = convert_datapoints_to_z(audio_seen_feat, self.encoder_audio)
            z_unseen_audio = convert_datapoints_to_z(
                audio_unseen_feat, self.encoder_audio
            )

            z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder_text)
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder_text)

            train_Z = [z_seen_audio, z_unseen_audio, z_seen_att, z_unseen_att]
            train_L = [
                audio_seen_label,
                audio_unseen_label,
                att_seen_label,
                att_unseen_label,
            ]

            # empty tensors are sorted out
            train_X = [
                train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0
            ]
            train_Y = [
                train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0
            ]

            train_X = torch.cat(train_X, dim=0)
            train_Y = torch.cat(train_Y, dim=0)

        print(test_novel_X.shape)
        print(test_novel_Y)
        print(test_seen_X.shape)
        print(test_seen_Y)

        ############################################################
        ##### initializing the classifier and train one epoch
        ############################################################

        cls = classifier.CLASSIFIER(
            clf,
            train_X,
            train_Y,
            test_seen_X,
            test_seen_Y,
            test_novel_X,
            test_novel_Y,
            cls_seenclasses,
            cls_novelclasses,
            self.num_classes,
            self.device,
            self.lr_cls,
            0.5,
            1,
            self.classifier_batch_size,
            self.generalized,
        )

        for k in range(self.cls_train_epochs):
            if k > 0:
                if self.generalized:
                    cls.acc_seen, cls.acc_novel, cls.H = cls.fit()
                else:
                    cls.acc = cls.fit_zsl()

            if self.generalized:

                print(
                    "[%.1f]     novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f"
                    % (k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss)
                )

                history.append(
                    [
                        cls.acc_seen.clone().detach().requires_grad_(True).item(),
                        cls.acc_novel.clone().detach().requires_grad_(True).item(),
                        cls.H.clone().detach().requires_grad_(True).item(),
                    ]
                )

            else:
                print("[%.1f]  acc=%.4f " % (k, cls.acc))
                history.append(
                    [0, cls.acc.clone().detach().requires_grad_(True).item(), 0]
                )

        if self.generalized:
            return (
                cls.acc_seen.clone().detach().requires_grad_(True).item(),
                cls.acc_novel.clone().detach().requires_grad_(True).item(),
                cls.H.clone().detach().requires_grad_(True).item(),
                history,
            )
        else:
            return 0, cls.acc.clone().detach().requires_grad_(True).item(), 0, history
