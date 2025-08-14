import copy

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    BertTokenizer,
    BertModel,
)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = [
    "BatchGenerator",
    "TripletDataset",
    "choice_accuracy",
    "cross_entropy_loss",
    "compute_kld",
    "compare_modalities",
    "corr_mat",
    "compute_trils",
    "cos_mat",
    "cross_correlate_latent_dims",
    "encode_as_onehot",
    "get_cut_off",
    "get_digits",
    "get_nneg_dims",
    "get_ref_indices",
    "get_results_files",
    "get_nitems",
    "kld_online",
    "kld_offline",
    "load_batches",
    "load_concepts",
    "load_data",
    "load_inds_and_item_names",
    "load_model",
    "load_sparse_codes",
    "load_ref_images",
    "load_targets",
    "load_weights",
    "l2_reg_",
    "matmul",
    "merge_dicts",
    "pickle_file",
    "unpickle_file",
    "pearsonr",
    "prune_weights",
    "rsm",
    "rsm_pred",
    "save_weights_",
    "sparsity",
    "spose2rsm_odd_one_out",
    "avg_sparsity",
    "softmax",
    "sort_weights",
    "trinomial_loss",
    "trinomial_probs",
    "validation",
]

import json
import logging
import math
import os
import pickle
import re
import torch
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd
import skimage.io as io
import torch.nn.functional as F

from collections import defaultdict, Counter
from itertools import combinations, permutations
from numba import njit, jit, prange
from os.path import join as pjoin
from skimage.transform import resize
from torch.optim import Adam, AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from typing import Tuple, Iterator, List, Dict
from models import model as md

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    BertTokenizer,
    BertModel,
)

from numba import njit


class TripletDataset(Dataset):

    def __init__(self, I: torch.tensor, dataset: torch.Tensor):
        self.I = I
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = encode_as_onehot(self.I, self.dataset[idx])
        return sample


class TripletDataset_ID(Dataset):

    def __init__(self, average_reps: torch.tensor, dataset: torch.Tensor):
        self.average_reps = average_reps
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.average_reps[idx,]
        return sample


class BatchGenerator(object):

    def __init__(
        self,
        I: torch.tensor,
        dataset: torch.Tensor,
        batch_size: int,
        sampling_method: str = "normal",
        p=None,
        method: str = "average",
    ):
        self.I = I
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampling_method = sampling_method
        self.p = p
        self.method = method

        if sampling_method == "soft":
            assert isinstance(self.p, float)
            self.n_batches = int(len(self.dataset) * self.p) // self.batch_size
        else:
            self.n_batches = len(self.dataset) // self.batch_size

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self.get_batches(self.I, self.dataset, self.method)

    def sampling(self, triplets: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """randomly sample training data during each epoch"""
        rnd_perm = torch.randperm(len(triplets))
        if self.sampling_method == "soft":
            rnd_perm = rnd_perm[: int(len(rnd_perm) * self.p)]
        return triplets[rnd_perm], ids[rnd_perm]

    def get_batches(
        self, I: torch.Tensor, triplets: torch.Tensor, method: str
    ) -> Iterator[torch.Tensor]:
        if not isinstance(self.sampling_method, type(None)):
            triplets, ids = self.sampling(triplets[:, 0:3], triplets[:, 3])
        else:
            ids = triplets[:, 3]
            triplets = triplets[:, 0:3]
        for i in range(self.n_batches):
            batch = encode_as_onehot(
                I, triplets[i * self.batch_size : (i + 1) * self.batch_size]
            )
            ids_batch = ids[i * self.batch_size : (i + 1) * self.batch_size]
            ids_batch_triplet = np.repeat(ids_batch, 3)
            if method == "average":
                yield batch
            elif method == "ids":
                yield batch, ids_batch_triplet


class BatchGenerator_ID(object):

    def __init__(
        self,
        average_reps: torch.tensor,
        dataset: torch.Tensor,
        batch_size: int,
        sampling_method: str = "normal",
        p=None,
        method: str = "two_step",
        within_subjects: bool = False,
    ):
        self.average_reps = average_reps
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampling_method = sampling_method
        self.p = p
        self.method = method
        self.within_subjects = within_subjects

        if sampling_method == "soft":
            assert isinstance(self.p, float)
            self.n_batches = int(len(self.dataset) * self.p) // self.batch_size
        else:
            self.n_batches = len(self.dataset) // self.batch_size

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self.get_batches(self.average_reps, self.dataset, self.method)

    def sampling(
        self, triplets: torch.Tensor, ids: torch.Tensor, within_subjects: bool = False
    ) -> torch.Tensor:
        """randomly sample training data during each epoch"""
        if within_subjects:
            df_triplets = pd.DataFrame(
                np.concatenate((triplets[:, 0:3], ids[:, np.newaxis]), axis=1)
            )
            df_triplets.columns = [0, 1, 2, "ID"]
            df_triplets_random = (
                df_triplets.groupby("ID")[[0, 1, 2]]
                .apply(lambda x: x.sample(frac=1))
                .reset_index(drop=False)
            )
            return (
                triplets[df_triplets_random["level_1"]],
                ids[df_triplets_random["level_1"]],
            )

        elif within_subjects == False:
            rnd_perm = torch.randperm(len(triplets))
            if self.sampling_method == "soft":
                rnd_perm = rnd_perm[: int(len(rnd_perm) * self.p)]
            return triplets[rnd_perm], ids[rnd_perm]

    def get_batches(
        self,
        average_reps: torch.Tensor,
        triplets: torch.Tensor,
        method: str = "two_step",
    ) -> Iterator[torch.Tensor]:
        if not isinstance(self.sampling_method, type(None)):
            triplets, ids = self.sampling(
                triplets[:, 0:3], triplets[:, 3], self.within_subjects
            )
        else:
            ids = triplets[:, 3]
            triplets = triplets[:, 0:3]
        for i in range(self.n_batches):
            # batch = encode_as_onehot(I, triplets[i*self.batch_size: (i+1)*self.batch_size])
            batch = average_reps[
                triplets.flatten()[
                    i * 3 * self.batch_size : (i + 1) * 3 * self.batch_size
                ],
                :,
            ]
            ids_batch = ids[i * self.batch_size : (i + 1) * self.batch_size]
            ids_batch_triplet = np.repeat(ids_batch, 3)
            if method == "two_step":
                yield batch
            elif method == "embedding":
                yield batch, ids_batch_triplet


def pickle_file(file: dict, out_path: str, file_name: str) -> None:
    with open(os.path.join(out_path, "".join((file_name, ".txt"))), "wb") as f:
        f.write(pickle.dumps(file))


def unpickle_file(in_path: str, file_name: str) -> dict:
    return pickle.loads(
        open(os.path.join(in_path, "".join((file_name, ".txt"))), "rb").read()
    )


def assert_nneg(X: np.ndarray, thresh: float = 1e-5) -> np.ndarray:
    """if data matrix X contains negative real numbers, transform matrix into R+ (i.e., positive real number(s) space)"""
    if np.any(X < 0):
        X -= np.amin(X, axis=0)
        return X + thresh
    return X


def load_inds_and_item_names(folder: str = "./data") -> Tuple[np.ndarray]:
    item_names = pd.read_csv(
        pjoin(folder, "item_names.tsv"), encoding="utf-8", sep="\t"
    ).uniqueID.values
    sortindex = pd.read_table(pjoin(folder, "sortindex"), header=None)[0].values
    return item_names, sortindex


def load_ref_images(img_folder: str, item_names: np.ndarray) -> np.ndarray:
    ref_images = np.array(
        [
            resize(
                io.imread(pjoin("./reference_images", name + ".jpg")),
                (400, 400),
                anti_aliasing=True,
            )
            for name in item_names
        ]
    )
    return ref_images


def load_concepts(folder: str = "./data") -> pd.DataFrame:
    concepts = pd.read_csv(
        pjoin(folder, "category_mat_manual.tsv"), encoding="utf-8", sep="\t"
    )
    return concepts


def load_data(
    device: torch.device, triplets_dir: str, inference: bool = False
) -> Tuple[torch.Tensor]:
    """load train and test triplet datasets into memory"""
    if inference:
        with open(pjoin(triplets_dir, "test_triplets.npy"), "rb") as test_triplets:
            test_triplets = (
                torch.from_numpy(np.load(test_triplets))
                .to(device)
                .type(torch.LongTensor)
            )
            return test_triplets
    try:
        with open(pjoin(triplets_dir, "train_90.npy"), "rb") as train_file:
            train_triplets = (
                torch.from_numpy(np.load(train_file)).to(device).type(torch.LongTensor)
            )

        with open(pjoin(triplets_dir, "test_10.npy"), "rb") as test_file:
            test_triplets = (
                torch.from_numpy(np.load(test_file)).to(device).type(torch.LongTensor)
            )
    except FileNotFoundError:
        print("\n...Could not find any .npy files for current modality.")
        print("...Now searching for .txt files.\n")
        train_triplets = (
            torch.from_numpy(np.loadtxt(pjoin(triplets_dir, "train_90.txt")))
            .to(device)
            .type(torch.LongTensor)
        )
        test_triplets = (
            torch.from_numpy(np.loadtxt(pjoin(triplets_dir, "test_10.txt")))
            .to(device)
            .type(torch.LongTensor)
        )
    return train_triplets, test_triplets


def load_data_ID(
    device: torch.device,
    triplets_dir: str,
    inference: bool = False,
    testcase: bool = False,
    use_shuffled_subjects: str = "actual",
    splithalf: str = "no",
) -> Tuple[torch.Tensor]:
    """load train and test triplet datasets with associated participant ID into memory"""
    if inference:
        with open(pjoin(triplets_dir, "test_triplets_ID.npy"), "rb") as test_triplets:
            test_triplets = (
                torch.from_numpy(np.load(test_triplets))
                .to(device)
                .type(torch.LongTensor)
            )
            return test_triplets
    try:
        with open(pjoin(triplets_dir, "train_90_ID.npy"), "rb") as train_file:
            train_triplets = (
                torch.from_numpy(np.load(train_file)).to(device).type(torch.LongTensor)
            )

        with open(pjoin(triplets_dir, "test_10_ID.npy"), "rb") as test_file:
            test_triplets = (
                torch.from_numpy(np.load(test_file)).to(device).type(torch.LongTensor)
            )
    except FileNotFoundError:
        print("\n...Could not find any .npy files for current modality.")
        print("...Now searching for .txt files.\n")
        if splithalf == "no":
            if testcase:
                train_triplets = (
                    torch.from_numpy(
                        np.loadtxt(pjoin(triplets_dir, "train_90_ID_smallsample.txt"))
                    )
                    .to(device)
                    .type(torch.LongTensor)
                )
                test_triplets = (
                    torch.from_numpy(
                        np.loadtxt(pjoin(triplets_dir, "test_10_ID_smallsample.txt"))
                    )
                    .to(device)
                    .type(torch.LongTensor)
                )
            elif testcase == False:
                if use_shuffled_subjects == "actual":
                    train_triplets = (
                        torch.from_numpy(
                            np.loadtxt(pjoin(triplets_dir, "train_90_ID_item.txt"))
                        )
                        .to(device)
                        .type(torch.LongTensor)
                    )
                    test_triplets = (
                        torch.from_numpy(
                            np.loadtxt(pjoin(triplets_dir, "test_10_ID_item.txt"))
                        )
                        .to(device)
                        .type(torch.LongTensor)
                    )
                elif use_shuffled_subjects == "shuffled":
                    train_triplets = (
                        torch.from_numpy(
                            np.loadtxt(
                                pjoin(triplets_dir, "train_shuffled_90_ID_item.txt")
                            )
                        )
                        .to(device)
                        .type(torch.LongTensor)
                    )
                    test_triplets = (
                        torch.from_numpy(
                            np.loadtxt(
                                pjoin(triplets_dir, "test_shuffled_10_ID_item.txt")
                            )
                        )
                        .to(device)
                        .type(torch.LongTensor)
                    )
        elif splithalf == "1":
            train_triplets = (
                torch.from_numpy(
                    np.loadtxt(pjoin(triplets_dir, "splithalf_1_ID_item.txt"))
                )
                .to(device)
                .type(torch.LongTensor)
            )
            test_triplets = (
                torch.from_numpy(
                    np.loadtxt(pjoin(triplets_dir, "splithalf_2_ID_item.txt"))
                )
                .to(device)
                .type(torch.LongTensor)
            )
        elif splithalf == "2":
            train_triplets = (
                torch.from_numpy(
                    np.loadtxt(pjoin(triplets_dir, "splithalf_2_ID_item.txt"))
                )
                .to(device)
                .type(torch.LongTensor)
            )
            test_triplets = (
                torch.from_numpy(
                    np.loadtxt(pjoin(triplets_dir, "splithalf_1_ID_item.txt"))
                )
                .to(device)
                .type(torch.LongTensor)
            )

    return train_triplets, test_triplets


def load_data_combined(
    device: torch.device, triplets_dir: str, testcase: bool = False
) -> Tuple[torch.Tensor]:
    """load train and test triplet datasets with associated participant ID into memory"""
    if testcase:
        train_triplets = (
            torch.from_numpy(
                np.loadtxt(
                    pjoin(triplets_dir, "ooo_data_modeling_old_and_new_testcase.txt")
                )
            )
            .to(device)
            .type(torch.LongTensor)
        )
    elif testcase == False:
        train_triplets = (
            torch.from_numpy(
                np.loadtxt(pjoin(triplets_dir, "ooo_data_modeling_old_and_new.txt"))
            )
            .to(device)
            .type(torch.LongTensor)
        )

    return train_triplets


def get_nitems(train_triplets: torch.Tensor) -> int:
    # number of unique items in the data matrix
    n_items = torch.max(train_triplets).item()
    if torch.min(train_triplets).item() == 0:
        n_items += 1
    return n_items


def load_batches(
    train_triplets: torch.Tensor,
    test_triplets: torch.Tensor,
    n_items: int,
    batch_size: int,
    inference: bool = False,
    sampling_method: str = None,
    rnd_seed: int = None,
    multi_proc: bool = False,
    n_gpus: int = None,
    p=None,
    method="average",
):
    # initialize an identity matrix of size n_items x n_items for one-hot-encoding of triplets
    I = torch.eye(n_items)
    if inference:
        assert train_triplets is None
        test_batches, test_ids = BatchGenerator(
            I=I,
            dataset=test_triplets,
            batch_size=batch_size,
            sampling_method=None,
            p=None,
            method=method,
        )
        return test_batches, test_ids
    if multi_proc and n_gpus > 1:
        if sampling_method == "soft":
            warnings.warn(
                f"...Soft sampling cannot be used in a multi-process distributed training setting.",
                RuntimeWarning,
            )
            warnings.warn(
                f"...Processes will equally distribute the entire training dataset amongst each other.",
                RuntimeWarning,
            )
            warnings.warn(
                f"...If you want to use soft sampling, you must switch to single GPU or CPU training.",
                UserWarning,
            )
        train_set = TripletDataset(I=I, dataset=train_triplets)
        val_set = TripletDataset(I=I, dataset=test_triplets)
        train_sampler = DistributedSampler(
            dataset=train_set, shuffle=True, seed=rnd_seed
        )
        train_batches = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=n_gpus,
        )
        val_batches = DataLoader(
            dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=n_gpus
        )
    else:
        # create two iterators of train and validation mini-batches respectively
        train_batches = BatchGenerator(
            I=I,
            dataset=train_triplets,
            batch_size=batch_size,
            sampling_method=sampling_method,
            p=p,
            method=method,
        )
        val_batches = BatchGenerator(
            I=I,
            dataset=test_triplets,
            batch_size=batch_size,
            sampling_method=None,
            p=None,
            method=method,
        )
    return train_batches, val_batches


def load_batches_ID(
    train_triplets: torch.Tensor,
    test_triplets: torch.Tensor,
    average_reps: torch.Tensor,
    n_items: int,
    batch_size: int,
    inference: bool = False,
    sampling_method: str = None,
    rnd_seed: int = None,
    multi_proc: bool = False,
    n_gpus: int = None,
    p=None,
    method: str = "two_step",
    within_subjects: bool = False,
):

    if inference:
        assert train_triplets is None
        test_batches = BatchGenerator_ID(
            average_reps=average_reps,
            dataset=test_triplets,
            batch_size=batch_size,
            sampling_method=None,
            p=None,
            method=method,
            within_subjects=within_subjects,
        )
        return test_batches
    if multi_proc and n_gpus > 1:
        if sampling_method == "soft":
            warnings.warn(
                f"...Soft sampling cannot be used in a multi-process distributed training setting.",
                RuntimeWarning,
            )
            warnings.warn(
                f"...Processes will equally distribute the entire training dataset amongst each other.",
                RuntimeWarning,
            )
            warnings.warn(
                f"...If you want to use soft sampling, you must switch to single GPU or CPU training.",
                UserWarning,
            )
        train_set = TripletDataset_ID(I=I, dataset=train_triplets)
        val_set = TripletDataset_ID(I=I, dataset=test_triplets)
        train_sampler = DistributedSampler(
            dataset=train_set, shuffle=True, seed=rnd_seed
        )
        train_batches = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=n_gpus,
        )
        val_batches = DataLoader(
            dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=n_gpus
        )
    else:
        # create two iterators of train and validation mini-batches respectively
        train_batches = BatchGenerator_ID(
            average_reps=average_reps,
            dataset=train_triplets,
            batch_size=batch_size,
            sampling_method=sampling_method,
            p=p,
            method=method,
            within_subjects=within_subjects,
        )
        val_batches = BatchGenerator_ID(
            average_reps=average_reps,
            dataset=test_triplets,
            batch_size=batch_size,
            sampling_method=None,
            p=None,
            method=method,
            within_subjects=within_subjects,
        )
    return train_batches, val_batches


def l2_reg_(model, weight_decay: float = 1e-5) -> torch.Tensor:
    loc_norms_squared = 0.5 * (
        model.encoder_mu[0].weight.pow(2).sum() + model.encoder_mu[0].bias.pow(2).sum()
    )
    scale_norms_squared = (
        model.encoder_b[0].weight.pow(2).sum() + model.encoder_mu[0].bias.pow(2).sum()
    )
    l2_reg = weight_decay * (loc_norms_squared + scale_norms_squared)
    return l2_reg


def encode_as_onehot(I: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
    """encode item triplets as one-hot-vectors"""
    return I[triplets.flatten(), :]


def softmax(sims: tuple, t: torch.Tensor) -> torch.Tensor:
    return torch.exp(sims[0] / t) / torch.sum(
        torch.stack([torch.exp(sim / t) for sim in sims]), dim=0
    )


def cross_entropy_loss(sims: tuple, t: torch.Tensor) -> torch.Tensor:
    sims_scaled = torch.stack(sims, dim=-1) / t
    return torch.mean(-torch.log(F.softmax(sims_scaled, dim=1)[:, 0]))
    # return torch.mean(-F.log_sotmax(torch.stack(sims, dim=-1)/t, dim=1)[:, 0])
    # replaced by torch softmax function with temperature == 1 to avoid Nan values
    # return torch.mean(-torch.log(softmax(sims, t)))


def temperature_softmax(logits, temperature=1.0, dim=-1):
    return F.softmax(logits / temperature, dim=dim)


def compute_similarities(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    method: str,
    distance_metric: str = "dot",
) -> Tuple:
    if distance_metric == "dot":
        pos_sim = torch.sum(anchor * positive, dim=1)
        neg_sim = torch.sum(anchor * negative, dim=1)
        if method == "odd_one_out":
            neg_sim_2 = torch.sum(positive * negative, dim=1)
            return pos_sim, neg_sim, neg_sim_2
        else:
            return pos_sim, neg_sim
    elif distance_metric == "euclidean":
        pos_sim = -1 * torch.sqrt(
            torch.sum(torch.square(torch.sub(anchor, positive)), dim=1)
        )
        neg_sim = -1 * torch.sqrt(
            torch.sum(torch.square(torch.sub(anchor, negative)), dim=1)
        )

        if method == "odd_one_out":
            neg_sim_2 = -1 * torch.sqrt(
                torch.sum(torch.square(torch.sub(positive, negative)), dim=1)
            )
            return pos_sim, neg_sim, neg_sim_2
        else:
            return pos_sim, neg_sim


def accuracy_(probas: torch.Tensor) -> float:
    choices = np.where(
        probas.mean(axis=1) == probas.max(axis=1), -1, np.argmax(probas, axis=1)
    )
    acc = np.where(choices == 0, 1, 0).mean()
    return acc


def choice_accuracy(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    method: str,
    distance_metric: str = "dot",
    scalingfactors: torch.Tensor = torch.Tensor([1]),
) -> float:
    similarities_prep = compute_similarities(
        anchor, positive, negative, method, distance_metric
    )
    similarities = torch.stack(similarities_prep, dim=-1)
    similarities_scaled = similarities / scalingfactors
    probas = F.softmax(similarities_scaled, dim=1).detach().cpu().numpy()
    # the following uses the softmax policy
    proba_correct = probas[:, 0].mean()
    # accuracy_ uses an argmax policy
    max_correct = accuracy_(probas)
    return proba_correct, max_correct


def trinomial_probs(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    method: str,
    t: torch.Tensor,
    distance_metric: str = "dot",
) -> torch.Tensor:
    sims = compute_similarities(anchor, positive, negative, method, distance_metric)
    return softmax(sims, t)


def trinomial_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    method: str,
    t: torch.Tensor,
    distance_metric: str = "dot",
) -> torch.Tensor:
    sims = compute_similarities(anchor, positive, negative, method, distance_metric)
    return cross_entropy_loss(sims, t)


def kld_online(
    mu_1: torch.Tensor, l_1: torch.Tensor, mu_2: torch.Tensor, l_2: torch.Tensor
) -> torch.Tensor:
    return torch.mean(
        torch.log(l_1 / l_2)
        + (l_2 / l_1) * torch.exp(-l_1 * torch.abs(mu_1 - mu_2))
        + l_2 * torch.abs(mu_1 - mu_2)
        - 1
    )


def kld_offline(
    mu_1: torch.Tensor, b_1: torch.Tensor, mu_2: torch.Tensor, b_2: torch.Tensor
) -> torch.Tensor:
    return (
        torch.log(b_2 / b_1)
        + (b_1 / b_2) * torch.exp(-torch.abs(mu_1 - mu_2) / b_1)
        + torch.abs(mu_1 - mu_2) / b_2
        - 1
    )


def get_nneg_dims(W: torch.Tensor, eps: float = 0.1) -> int:
    w_max = W.max(dim=1)[0]
    nneg_d = len(w_max[w_max > eps])
    return nneg_d


def remove_zeros(W: np.ndarray, eps: float = 0.1) -> np.ndarray:
    w_max = np.max(W, axis=1)
    W = W[np.where(w_max > eps)]
    return W


################################################
######### helper functions for evaluation ######
################################################


def get_seeds(PATH: str) -> List[str]:
    return [
        dir.name
        for dir in os.scandir(PATH)
        if dir.is_dir() and dir.name.startswith("seed")
    ]


def seed_sampling(PATH: str) -> np.ndarray:
    seed = np.random.choice(get_seeds(PATH))
    with open(os.path.join(PATH, seed, "test_probas.npy"), "rb") as f:
        probas = np.load(f)
    return probas


def instance_sampling(probas: np.ndarray) -> np.ndarray:
    rnd_sample = np.random.choice(
        np.arange(len(probas)), size=len(probas), replace=True
    )
    probas_draw = probas[rnd_sample]
    return probas_draw


def get_global_averages(avg_probas: dict) -> np.ndarray:
    sorted_bins = dict(sorted(avg_probas.items()))
    return np.array([np.mean(p) for p in sorted_bins.values()])


def compute_pm(probas: np.ndarray) -> Tuple[np.ndarray, dict]:
    """compute probability mass for every choice"""
    avg_probas = defaultdict(list)
    count_vector = np.zeros((2, 11))
    for pmf in probas:
        indices = np.round(pmf * 10).astype(int)
        count_vector[0, indices[0]] += 1
        count_vector[1, indices] += 1
        for k, p in enumerate(pmf):
            avg_probas[int(indices[k])].append(p)
    model_confidences = count_vector[0] / count_vector[1]
    avg_probas = get_global_averages(avg_probas)
    return model_confidences, avg_probas


def mse(avg_p: np.ndarray, confidences: np.ndarray) -> float:
    return np.mean((avg_p - confidences) ** 2)


def bootstrap_calibrations(
    PATH: str, alpha: float, n_bootstraps: int = 1000
) -> np.ndarray:
    mses = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        probas = seed_sampling(PATH)
        probas_draw = instance_sampling(probas)
        confidences, avg_p = compute_pm(probas_draw, alpha)
        mses[i] += mse(avg_p, confidences)
    return mses


def get_model_confidence_(PATH: str) -> Tuple[np.ndarray, np.ndarray]:
    seeds = get_seeds(PATH)
    confidence_scores = np.zeros((len(seeds), 11))
    avg_probas = np.zeros((len(seeds), 11))
    for i, seed in enumerate(seeds):
        with open(os.path.join(PATH, seed, "test_probas.npy"), "rb") as f:
            confidence, avg_p = compute_pm(np.load(f))
            confidence_scores[i] += confidence
            avg_probas[i] += avg_p
    return confidence_scores, avg_probas


def smoothing_(p: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    return (p + alpha) / np.sum(p + alpha)


def entropy_(p: np.ndarray) -> np.ndarray:
    return np.sum(np.where(p == 0, 0, p * np.log(p)))


def cross_entropy_(p: np.ndarray, q: np.ndarray, alpha: float) -> float:
    return -np.sum(p * np.log(smoothing_(q, alpha)))


def kld_(p: np.ndarray, q: np.ndarray, alpha: float) -> float:
    return entropy_(p) + cross_entropy_(p, q, alpha)


def compute_divergences(
    human_pmfs: dict, model_pmfs: dict, metric: str = "kld"
) -> np.ndarray:
    assert len(human_pmfs) == len(
        model_pmfs
    ), "\nNumber of triplets in human and model distributions must correspond.\n"
    divergences = np.zeros(len(human_pmfs))
    for i, (triplet, p) in enumerate(human_pmfs.items()):
        q = model_pmfs[triplet]
        div = kld_(p, q) if metric == "kld" else cross_entropy_(p, q)
        divergences[i] += div
    return divergences


def mat2py(triplet: tuple) -> tuple:
    return tuple(np.asarray(triplet) - 1)


def pmf(hist: dict) -> np.ndarray:
    values = np.array(list(hist.values()))
    return values / np.sum(values)


def histogram(choices: list, behavior: bool = False) -> dict:
    hist = {i + 1 if behavior else i: 0 for i in range(3)}
    for choice in choices:
        hist[choice if behavior else choice.item()] += 1
    return hist


def compute_pmfs(choices: dict, behavior: bool) -> dict:
    pmfs = {
        mat2py(t) if behavior else t: pmf(histogram(c, behavior))
        for t, c in choices.items()
    }
    return pmfs


def get_choice_distributions(test_set: pd.DataFrame) -> dict:
    """function to compute human choice distributions and corresponding pmfs"""
    triplets = test_set[["trip.1", "trip.2", "trip.3"]]
    test_set["triplets"] = list(map(tuple, triplets.to_numpy()))
    unique_triplets = test_set.triplets.unique()
    choice_distribution = defaultdict(list)
    for triplet in unique_triplets:
        choices = list(test_set[test_set["triplets"] == triplet].choice.values)
        sorted_choices = [
            np.where(np.argsort(triplet) + 1 == c)[0][0] + 1 for c in choices
        ]
        sorted_triplet = tuple(sorted(triplet))
        choice_distribution[sorted_triplet].extend(sorted_choices)
    choice_pmfs = compute_pmfs(choice_distribution, behavior=True)
    return choice_pmfs


def collect_choices(
    probas: np.ndarray, human_choices: np.ndarray, model_choices: dict
) -> dict:
    """collect model choices at inference time"""
    probas = probas.flip(dims=[1])
    for pmf, choices in zip(probas, human_choices):
        sorted_choices = tuple(np.sort(choices))
        model_choices[sorted_choices].append(np.argmax(pmf[np.argsort(choices)]))
    return model_choices


def logsumexp_(logits: torch.Tensor) -> torch.Tensor:
    return torch.exp(logits - torch.logsumexp(logits, dim=1)[..., None])


def test(
    model,
    test_batches,
    version: str,
    task: str,
    device: torch.device,
    batch_size=None,
    n_samples=None,
    distance_metric: str = "dot",
    temperature: float = 1.0,
) -> Tuple:
    probas = torch.zeros(int(len(test_batches) * batch_size), 3)
    temperature = temperature.clone().detach()
    # temperature = torch.tensor(temperature).to(device)
    model_choices = defaultdict(list)
    model.eval()
    with torch.no_grad():
        batch_accs = torch.zeros(len(test_batches))
        for j, batch in enumerate(test_batches):
            batch = batch.to(device)
            if version == "variational":
                assert isinstance(
                    n_samples, int
                ), "\nOutput logits of variational neural networks have to be averaged over different samples through mc sampling.\n"
                test_acc, _, batch_probas = mc_sampling(
                    model=model,
                    batch=batch,
                    temperature=temperature,
                    task=task,
                    n_samples=n_samples,
                    device=device,
                )
            else:
                logits = model(batch)
                anchor, positive, negative = torch.unbind(
                    torch.reshape(logits, (-1, 3, logits.shape[-1])), dim=1
                )
                similarities = compute_similarities(
                    anchor, positive, negative, task, distance_metric
                )
                # stacked_sims = torch.stack(similarities, dim=-1)
                # batch_probas = F.softmax(logsumexp_(stacked_sims), dim=1)
                batch_probas = F.softmax(torch.stack(similarities, dim=-1), dim=1)
                test_acc = choice_accuracy(anchor, positive, negative, task)

            probas[j * batch_size : (j + 1) * batch_size] += batch_probas
            batch_accs[j] += test_acc
            human_choices = (
                batch.nonzero(as_tuple=True)[-1].view(batch_size, -1).numpy()
            )
            model_choices = collect_choices(batch_probas, human_choices, model_choices)

    probas = probas.cpu().numpy()
    probas = probas[np.where(probas.sum(axis=1) != 0.0)]
    model_pmfs = compute_pmfs(model_choices, behavior=False)
    test_acc = batch_accs.mean().item()
    return test_acc, probas, model_pmfs


def validation(
    model,
    val_batches,
    task: str,
    device: torch.device,
    temperature: torch.tensor,
    sampling: bool = False,
    batch_size=None,
    distance_metric: str = "dot",
    level_explanation="avg",
    modeltype="free_weights",
):
    if sampling:
        assert isinstance(batch_size, int), "batch size must be defined"
        sampled_choices = np.zeros((int(len(val_batches) * batch_size), 3), dtype=int)

    model.eval()
    with torch.no_grad():
        batch_losses_val = torch.zeros(len(val_batches))
        batch_accs_val_max = torch.zeros(len(val_batches))
        batch_accs_val_proba = torch.zeros(len(val_batches))

        for j, batch in enumerate(val_batches):
            if level_explanation == "avg":
                batch = batch.to(device)
                logits = model(batch)
                anchor, positive, negative = torch.unbind(
                    torch.reshape(logits, (-1, 3, logits.shape[-1])), dim=1
                )
                c_entropy = trinomial_loss(
                    anchor, positive, negative, task, temperature, distance_metric
                )

            elif level_explanation == "ID":
                if modeltype in [
                    "random_weights_free_scaling",
                    "random_weights_random_scaling",
                ]:
                    b = batch[0].to(device)
                    id = batch[1].to(device)
                    c_entropy, anchor, positive, negative = model(
                        b, id, distance_metric
                    )
                    temperature = model.model2(id[::3])
                else:
                    b = batch[0].to(device)
                    id = batch[1].to(device)
                    logits = model(b, id)
                    anchor, positive, negative = torch.unbind(
                        torch.reshape(logits, (-1, 3, logits.shape[-1])), dim=1
                    )
                    c_entropy = trinomial_loss(
                        anchor, positive, negative, task, temperature, distance_metric
                    )

            if sampling:
                sims_prep = compute_similarities(
                    anchor, positive, negative, task, distance_metric
                )
                sims = torch.stack(sims_prep, dim=-1)

                sims_scaled = sims / temperature
                probas = F.softmax(sims_scaled, dim=1).numpy()
                probas = probas[:, ::-1]
                human_choices = (
                    b.nonzero(as_tuple=True)[-1].view(batch_size, -1).numpy()
                )
                model_choices = np.array(
                    [
                        np.random.choice(h_choice, size=len(p), replace=False, p=p)[
                            ::-1
                        ]
                        for h_choice, p in zip(human_choices, probas)
                    ]
                )
                sampled_choices[j * batch_size : (j + 1) * batch_size] += model_choices

            else:
                val_loss = c_entropy
                val_acc_proba, val_acc_max = choice_accuracy(
                    anchor,
                    positive,
                    negative,
                    task,
                    distance_metric,
                    scalingfactors=temperature,
                )
                batch_losses_val[j] += val_loss.item()
                batch_accs_val_proba[j] += val_acc_proba
                batch_accs_val_max[j] += val_acc_max

    if sampling:
        return sampled_choices

    avg_val_loss = torch.mean(batch_losses_val).item()
    avg_val_acc_max = torch.mean(batch_accs_val_max).item()
    avg_val_acc_proba = torch.mean(batch_accs_val_proba).item()
    return avg_val_loss, avg_val_acc_proba, avg_val_acc_max


def get_digits(string: str) -> int:
    c = ""
    nonzero = False
    for i in string:
        if i.isdigit():
            if (int(i) == 0) and (not nonzero):
                continue
            else:
                c += i
                nonzero = True
    return int(c)


def get_results_files(
    results_dir: str,
    modality: str,
    version: str,
    subfolder: str,
    vision_model=None,
    layer=None,
) -> list:
    if modality == "visual":
        assert isinstance(vision_model, str) and isinstance(
            layer, str
        ), "name of vision model and layer are required"
        PATH = pjoin(
            results_dir, modality, vision_model, layer, version, f"{dim}d", f"{lmbda}"
        )
    else:
        PATH = pjoin(results_dir, modality, version, f"{dim}d", f"{lmbda}")
    files = [
        pjoin(PATH, seed, f)
        for seed in os.listdir(PATH)
        for f in os.listdir(pjoin(PATH, seed))
        if f.endswith(".json")
    ]
    return files


def sort_results(results: dict) -> dict:
    return dict(sorted(results.items(), key=lambda kv: kv[0], reverse=False))


def merge_dicts(files: list) -> dict:
    """merge multiple .json files efficiently into a single dictionary"""
    results = {}
    for f in files:
        with open(f, "r") as f:
            results.update(dict(json.load(f)))
    results = sort_results(results)
    return results


def load_model(
    model,
    results_dir: str,
    modality: str,
    version: str,
    data: str,
    dim: int,
    lmbda: float,
    rnd_seed: int,
    device: torch.device,
    subfolder: str = "model",
):
    model_path = pjoin(
        results_dir,
        modality,
        version,
        data,
        f"{dim}d",
        f"{lmbda}",
        f"seed{rnd_seed:02d}",
        subfolder,
    )
    models = os.listdir(model_path)
    checkpoints = list(map(get_digits, models))
    last_checkpoint = np.argmax(checkpoints)
    PATH = pjoin(model_path, models[last_checkpoint])
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def save_weights_(out_path: str, W_mu: torch.tensor) -> None:
    W_mu = W_mu.detach().cpu().numpy()
    W_mu = remove_zeros(W_mu)
    W_sorted = np.abs(W_mu[np.argsort(-np.linalg.norm(W_mu, ord=1, axis=1))]).T
    with open(pjoin(out_path, "weights_sorted.npy"), "wb") as f:
        np.save(f, W_sorted)


def load_weights(model, version: str) -> Tuple[torch.Tensor]:
    if version == "variational":
        W_mu = model.encoder_mu[0].weight.data.T.detach()
        if hasattr(model.encoder_mu[0].bias, "data"):
            W_mu += model.encoder_mu[0].bias.data.detach()
        W_b = model.encoder_b[0].weight.data.T.detach()
        if hasattr(model.encoder_b[0].bias, "data"):
            W_b += model.encoder_b[0].bias.data.detach()
        W_mu = F.relu(W_mu)
        W_b = F.softplus(W_b)
        return W_mu, W_b
    else:
        return model.fc.weight.T.detach()


def prune_weights(model, version: str, indices: torch.Tensor, fraction: float):
    indices = indices[: int(len(indices) * fraction)]
    for n, m in model.named_parameters():
        if version == "variational":
            if re.search(r"encoder", n):
                # prune output weights and biases of encoder
                m.data = m.data[indices]
            else:
                # only prune input weights of decoder
                if re.search(r"weight", n):
                    m.data = m.data[:, indices]
        else:
            # prune output weights of fc layer
            m.data = m.data[indices]
    return model


def sort_weights(model, aggregate: bool) -> np.ndarray:
    """sort latent dimensions according to their l1-norm in descending order"""
    W = load_weights(model, version="deterministic").cpu()
    l1_norms = W.norm(p=1, dim=0)
    sorted_dims = torch.argsort(l1_norms, descending=True)
    if aggregate:
        l1_sorted = l1_norms[sorted_dims]
        return sorted_dims, l1_sorted.numpy()
    return sorted_dims, W[:, sorted_dims].numpy()


def get_cut_off(klds: np.ndarray) -> int:
    klds /= klds.max(axis=0)
    cut_off = np.argmax(
        [np.var(klds[i - 1]) - np.var(kld) for i, kld in enumerate(klds.T) if i > 0]
    )
    return cut_off


def compute_kld(model, lmbda: float, aggregate: bool, reduction=None) -> np.ndarray:
    mu_hat, b_hat = load_weights(model, version="variational")
    mu = torch.zeros_like(mu_hat)
    lmbda = torch.tensor(lmbda)
    b = torch.ones_like(b_hat).mul(lmbda.pow(-1))
    kld = kld_offline(mu_hat, b_hat, mu, b)
    if aggregate:
        assert isinstance(
            reduction, str
        ), "\noperator to aggregate KL divergences must be defined\n"
        if reduction == "sum":
            # use sum as to aggregate KLDs for each dimension
            kld_sum = kld.sum(dim=0)
            sorted_dims = torch.argsort(kld_sum, descending=True)
            klds_sorted = kld_sum[sorted_dims].cpu().numpy()
        else:
            # use max to aggregate KLDs for each dimension
            kld_max = kld.max(dim=0)[0]
            sorted_dims = torch.argsort(kld_max, descending=True)
            klds_sorted = kld_max[sorted_dims].cpu().numpy()
    else:
        # use mean KLD to sort dimensions in descending order (highest KLDs first)
        sorted_dims = torch.argsort(kld.mean(dim=0), descending=True)
        klds_sorted = kld[:, sorted_dims].cpu().numpy()
    return sorted_dims, klds_sorted


#############################################################################################
######### helper functions to load weight matrices and compare RSMs across modalities #######
#############################################################################################


def load_sparse_codes(PATH) -> np.ndarray:
    Ws = [f for f in os.listdir(PATH) if f.endswith(".txt")]
    max_epoch = np.argmax(list(map(get_digits, Ws)))
    W = np.loadtxt(pjoin(PATH, Ws[max_epoch]))
    W = remove_zeros(W)
    l1_norms = np.linalg.norm(W, ord=1, axis=1)
    sorted_dims = np.argsort(l1_norms)[::-1]
    W = W[sorted_dims]
    return W.T, sorted_dims


def load_targets(model: str, layer: str, folder: str = "./visual") -> np.ndarray:
    PATH = pjoin(folder, model, layer)
    with open(pjoin(PATH, "targets.npy"), "rb") as f:
        targets = np.load(f)
    return targets


def get_ref_indices(targets: np.ndarray) -> np.ndarray:
    n_items = len(np.unique(targets))
    cats = np.zeros(n_items, dtype=int)
    indices = np.zeros(n_items, dtype=int)
    for idx, cat in enumerate(targets):
        if cat not in cats:
            cats[cat] = cat
            indices[cat] = idx
    assert (
        len(indices) == n_items
    ), "\nnumber of indices for reference images must be equal to number of unique objects\n"
    return indices


def pearsonr(
    u: np.ndarray, v: np.ndarray, a_min: float = -1.0, a_max: float = 1.0
) -> np.ndarray:
    u_c = u - np.mean(u)
    v_c = v - np.mean(v)
    num = u_c @ v_c
    denom = np.linalg.norm(u_c) * np.linalg.norm(v_c)
    rho = (num / denom).clip(min=a_min, max=a_max)
    return rho


def cos_mat(W: np.ndarray, a_min: float = -1.0, a_max: float = 1.0) -> np.ndarray:
    num = matmul(W, W.T)
    l2_norms = np.linalg.norm(W, axis=1)  # compute l2-norm across rows
    denom = np.outer(l2_norms, l2_norms)
    cos_mat = (num / denom).clip(min=a_min, max=a_max)
    return cos_mat


def corr_mat(W: np.ndarray, a_min: float = -1.0, a_max: float = 1.0) -> np.ndarray:
    W_c = W - W.mean(axis=1)[:, np.newaxis]
    cov = matmul(W_c, W_c.T)
    l2_norms = np.linalg.norm(W_c, axis=1)  # compute l2-norm across rows
    denom = np.outer(l2_norms, l2_norms)
    # counteract potential rounding errors
    corr_mat = (cov / denom).clip(min=a_min, max=a_max)
    return corr_mat


@njit(parallel=False, fastmath=False)
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    I, K = A.shape
    K, J = B.shape
    C = np.zeros((I, J))
    for i in prange(I):
        for j in prange(J):
            for k in prange(K):
                C[i, j] += A[i, k] * B[k, j]
    return C


@njit(parallel=False, fastmath=False)
def rsm_pred(W: np.ndarray) -> np.ndarray:
    """convert weight matrix corresponding to the mean of each dim distribution for an object into a RSM"""
    N = W.shape[0]
    S = matmul(W, W.T)
    S_e = np.exp(S)  # exponentiate all elements in the inner product matrix S
    rsm = np.zeros((N, N))
    for i in prange(N):
        for j in prange(i + 1, N):
            for k in prange(N):
                if k != i and k != j:
                    rsm[i, j] += S_e[i, j] / (S_e[i, j] + S_e[i, k] + S_e[j, k])
    rsm /= N - 2
    rsm += rsm.T  # make similarity matrix symmetric
    np.fill_diagonal(rsm, 1)
    return rsm


def spose2rsm_odd_one_out(W: np.ndarray) -> np.ndarray:
    rsm = rsm_pred(W)
    rsm[rsm > 1] = 1
    assert np.allclose(rsm, rsm.T), "\nRSM is required to be a symmetric matrix\n"
    return rsm


def rsm(W: np.ndarray, metric: str) -> np.ndarray:
    rsm = corr_mat(W) if metric == "rho" else cos_mat(W)
    return rsm


def compute_trils(W_mod1: np.ndarray, W_mod2: np.ndarray, metric: str) -> float:
    metrics = ["cos", "pred", "rho"]
    assert metric in metrics, f"\nMetric must be one of {metrics}.\n"
    if metric == "pred":
        rsm_1 = spose_rsm(W_mod1)
        rsm_2 = spose_rsm(W_mod2)
    else:
        rsm_1 = rsm(W_mod1, metric)  # RSM wrt first modality (e.g., DNN)
        rsm_2 = rsm(W_mod2, metric)  # RSM wrt second modality (e.g., behavior)
    assert rsm_1.shape == rsm_2.shape, "\nRSMs must be of equal size.\n"
    # since RSMs are symmetric matrices, we only need to compare their lower triangular parts (main diagonal can be omitted)
    tril_inds = np.tril_indices(len(rsm_1), k=-1)
    tril_1 = rsm_1[tril_inds]
    tril_2 = rsm_2[tril_inds]
    return tril_1, tril_2, tril_inds


def compare_modalities(
    W_mod1: np.ndarray, W_mod2: np.ndarray, duplicates: bool = False
) -> Tuple[np.ndarray]:
    assert (
        W_mod1.shape[0] == W_mod2.shape[0]
    ), "\nNumber of items in weight matrices must align.\n"
    mod1_mod2_corrs = np.zeros(W_mod1.shape[1])
    mod2_dims = []
    for d_mod1, w_mod1 in enumerate(W_mod1.T):
        corrs = np.array([pearsonr(w_mod1, w_mod2) for w_mod2 in W_mod2.T])
        if duplicates:
            mod2_dims.append(np.argmax(corrs))
        else:
            for d_mod2 in np.argsort(-corrs):
                if d_mod2 not in mod2_dims:
                    mod2_dims.append(d_mod2)
                    break
        mod1_mod2_corrs[d_mod1] = corrs[mod2_dims[-1]]
    mod1_dims_sorted = np.argsort(-mod1_mod2_corrs)
    mod2_dims_sorted = np.asarray(mod2_dims)[mod1_dims_sorted]
    corrs = mod1_mod2_corrs[mod1_dims_sorted]
    return mod1_dims_sorted, mod2_dims_sorted, corrs


def sparsity(A: np.ndarray) -> float:
    return 1.0 - (A[A > 0].size / A.size)


def avg_sparsity(Ws: list) -> np.ndarray:
    return np.mean(list(map(sparsity, Ws)))


def robustness(corrs: np.ndarray, thresh: float) -> float:
    return len(corrs[corrs > thresh]) / len(corrs)


def cross_correlate_latent_dims(X, thresh: float = None) -> float:
    if isinstance(X, np.ndarray):
        W_mu_i = np.copy(X)
        W_mu_j = np.copy(X)
    else:
        W_mu_i, W_mu_j = X
    corrs = np.zeros(min(W_mu_i.shape))
    for i, w_i in enumerate(W_mu_i):
        if np.all(W_mu_i == W_mu_j):
            corrs[i] = np.max(
                [pearsonr(w_i, w_j) for j, w_j in enumerate(W_mu_j) if j != i]
            )
        else:
            corrs[i] = np.max([pearsonr(w_i, w_j) for w_j in W_mu_j])
    if thresh:
        return robustness(corrs, thresh)
    return np.mean(corrs)


def train_ID_model(
    l_train_triplets_ID: list,
    l_test_triplets_ID: list,
    array_avg_reps: np.array,
    n_items_ID: int,
    batch_size: int,
    sampling_method: str,
    task: str,
    temperature: float,
    embed_dim: int,
    distance_metric: str,
    rnd_seed: int,
    p: float,
    lr: float,
    epochs: int,
    device: str,
    model_dir: str,
) -> dict:
    tensor_avg_reps = torch.Tensor(array_avg_reps)
    train_batches_ID, val_batches_ID = load_batches_ID(
        train_triplets=l_train_triplets_ID,
        test_triplets=l_test_triplets_ID,
        average_reps=tensor_avg_reps,
        n_items=n_items_ID,
        batch_size=batch_size,
        sampling_method=sampling_method,
        rnd_seed=rnd_seed,
        p=p,
    )
    _, val_batches_avg = load_batches(
        train_triplets=l_train_triplets_ID,
        test_triplets=l_test_triplets_ID,
        n_items=n_items_ID,
        batch_size=batch_size,
        sampling_method=sampling_method,
        rnd_seed=rnd_seed,
        p=p,
    )
    model_ID = md.IndependentLinearLayerWithIntercept(array_avg_reps.shape[1])
    model_ID.to(device)
    optim = Adam(model_ID.parameters(), lr=lr)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    start = 0
    train_accs_ID, val_accs_ID = [], []
    train_losses_ID, val_losses_ID = [], []
    loglikelihoods_ID = []

    iter = 0

    for epoch in range(start, epochs):
        model_ID.train()
        batch_llikelihoods = torch.zeros(len(train_batches_ID))
        batch_losses_train = torch.zeros(len(train_batches_ID))
        batch_accs_train = torch.zeros(len(train_batches_ID))
        for i, batch in enumerate(train_batches_ID):
            optim.zero_grad()  # zero out gradients
            batch = batch.to(device)
            logits = model_ID(batch)
            anchor, positive, negative = torch.unbind(
                torch.reshape(logits, (-1, 3, embed_dim)), dim=1
            )
            loss = trinomial_loss(
                anchor, positive, negative, task, temperature, distance_metric
            )

            ICs = model_ID.intercepts
            Slopes = model_ID.weights

            loss.backward()
            optim.step()

            batch_losses_train[i] += loss.item()
            batch_llikelihoods[i] += loss.item()
            batch_accs_train[i] += choice_accuracy(
                anchor, positive, negative, task, distance_metric
            )
            iter += 1

        avg_llikelihood_ID = torch.mean(batch_llikelihoods).item()
        avg_train_loss_ID = torch.mean(batch_losses_train).item()
        avg_train_acc_ID = torch.mean(batch_accs_train).item()

        loglikelihoods_ID.append(avg_llikelihood_ID)
        train_losses_ID.append(avg_train_loss_ID)
        train_accs_ID.append(avg_train_acc_ID)

    return {
        "model": model_ID,
        "train_batches_ID": train_batches_ID,
        "val_batches_ID": val_batches_ID,
        "val_batches_avg": val_batches_avg,
        "train_accs": train_accs_ID,
        "train_losses": train_losses_ID,
        "train_ll": loglikelihoods_ID,
        "ics": ICs,
        "Slopes": Slopes,
        "n_choices_train": l_train_triplets_ID.shape[0],
        "n_choices_test": l_test_triplets_ID.shape[0],
    }


def process_ID_results(
    l_train_ID: list, l_val_ID: list, l_val_avg: list, n_ID: int, embed_dim: int
) -> dict:
    df_eval = pd.DataFrame(
        np.column_stack((np.arange(0, n_ID), l_val_ID, l_val_avg)),
        columns=["id", "avg_loss_ID", "avg_acc_ID", "avg_loss_avg", "avg_acc_avg"],
    )
    df_eval_long = pd.melt(
        df_eval, id_vars="id", var_name="variable", value_name="value"
    )
    df_swarm = df_eval_long.query("variable in ['avg_acc_avg', 'avg_acc_ID']").copy()
    df_swarm.loc[df_swarm["variable"] == "avg_acc_avg", "x_position"] = 0.0
    df_swarm.loc[df_swarm["variable"] == "avg_acc_ID", "x_position"] = 0.5

    l_train_acc = []
    l_params = []
    for id, tr in enumerate(l_train_ID):
        tmp = np.column_stack(
            (
                np.repeat(id, len(tr["train_accs"])),
                np.arange(0, len(tr["train_accs"])),
                tr["train_accs"],
            )
        )
        l_train_acc.append(tmp)
        tmp = np.column_stack(
            (
                np.repeat(id, embed_dim),
                np.arange(0, embed_dim),
                tr["ics"].detach().numpy(),
                tr["Slopes"].detach().numpy(),
            )
        )
        l_params.append(tmp)
    m_train_acc = np.concatenate(l_train_acc, axis=0)
    df_train_acc = pd.DataFrame(m_train_acc, columns=["id", "epoch", "train_acc"])
    m_params = np.concatenate(l_params, axis=0)
    df_params = pd.DataFrame(m_params, columns=["id", "dim", "intercept", "slope"])

    df_train_acc["epoch_bin"] = pd.cut(
        df_train_acc["epoch"],
        bins=20,
        labels=False,
    )
    df_train_acc_agg = (
        df_train_acc.groupby(["id", "epoch_bin"], observed=False)
        .agg({"train_acc": ["mean"]})
        .reset_index()
    )
    df_train_acc_agg.columns = [
        "_".join(col).strip() for col in df_train_acc_agg.columns.values
    ]
    df_train_acc_agg.columns = ["id", "epoch_bin", "train_acc_mean"]
    df_train_acc_agg = (
        df_train_acc_agg.groupby("epoch_bin", observed=False)["train_acc_mean"]
        .agg(["mean", "std"])
        .reset_index()
    )
    df_train_acc_agg["ci_95"] = df_train_acc_agg["std"] / np.sqrt(n_ID)
    return {
        "df_swarm": df_swarm,
        "df_train_acc_agg": df_train_acc_agg,
        "df_params": df_params,
    }


def load_avg_embeddings(model_id: str, device: str) -> list:
    if model_id == "Word2Vec":
        l_embeddings = np.load("data/word2vec-embeddings.npy")
    elif model_id == "clip-vit-base-p32":
        l_embeddings = np.load("data/clip-vit-base-p32-embeddings.npy")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)

        tbl_labels = pd.read_csv("data/unique_id.txt", delimiter="\\", header=None)
        tbl_labels["label_id"] = np.arange(1, tbl_labels.shape[0] + 1)
        tbl_labels.columns = ["label", "label_id"]
        new_order = ["label_id", "label"]
        tbl_labels = tbl_labels[new_order]

        l_embeddings = []

        for prompt in tbl_labels["label"]:
            tokenized_input = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(tokenized_input)
            embedding = output.last_hidden_state[0]
            emb_flat = torch.mean(embedding, axis=0).cpu().detach().numpy()
            l_embeddings.append(emb_flat)
    return l_embeddings


def delta_avg_id(
    anchors,
    positives,
    negatives,
    anchors_weighted,
    positives_weighted,
    negatives_weighted,
    ids,
    idx,
):
    # avg reps for current idx
    anchors_0 = torch.Tensor(
        np.array([anchor for anchor, i in zip(anchors, ids) if i == idx])
    )
    positives_0 = torch.Tensor(
        np.array([positive for positive, i in zip(positives, ids) if i == idx])
    )
    negatives_0 = torch.Tensor(
        np.array([negative for negative, i in zip(negatives, ids) if i == idx])
    )
    # id reps for current idx
    anchors_weighted_0 = torch.Tensor(
        np.array(
            [
                anchor_weighted
                for anchor_weighted, i in zip(anchors_weighted, ids)
                if i == idx
            ]
        )
    )
    positives_weighted_0 = torch.Tensor(
        np.array(
            [
                positive_weighted
                for positive_weighted, i in zip(positives_weighted, ids)
                if i == idx
            ]
        )
    )
    negatives_weighted_0 = torch.Tensor(
        np.array(
            [
                negative_weighted
                for negative_weighted, i in zip(negatives_weighted, ids)
                if i == idx
            ]
        )
    )
    # compute similarities for both models
    sims_avg = compute_similarities(
        anchors_0, positives_0, negatives_0, method="odd_one_out"
    )
    sims_id = compute_similarities(
        anchors_weighted_0,
        positives_weighted_0,
        negatives_weighted_0,
        method="odd_one_out",
    )
    # calculate accuracies on test set
    one_avg = (sims_avg[0] > sims_avg[1]).numpy() & (sims_avg[0] > sims_avg[2]).numpy()
    acc_eval_avg = one_avg.sum() / np.sum(ids == idx)
    one_id = (sims_id[0] > sims_id[1]).numpy() & (sims_id[0] > sims_id[2]).numpy()
    acc_eval_id = one_id.sum() / np.sum(ids == idx)
    delta = acc_eval_id - acc_eval_avg
    return delta, one_avg, one_id


def delta_avg_triplet(
    anchors,
    positives,
    negatives,
    anchors_weighted,
    positives_weighted,
    negatives_weighted,
    array_weights_items,
    array_weights_id,
    df_diagnostic_data,
):
    # prepare avg and id weights
    anchors = torch.Tensor(
        np.array(
            [array_weights_items[i, :] for i in list(df_diagnostic_data.loc[:, 0])]
        )
    )
    positives = torch.Tensor(
        np.array(
            [array_weights_items[i, :] for i in list(df_diagnostic_data.loc[:, 1])]
        )
    )
    negatives = torch.Tensor(
        np.array(
            [array_weights_items[i, :] for i in list(df_diagnostic_data.loc[:, 2])]
        )
    )
    anchors_weighted = [
        a * array_weights_id.numpy()[df_diagnostic_data.loc[id, "id_subject"]]
        for id, a in enumerate(anchors)
    ]
    positives_weighted = [
        a * array_weights_id.numpy()[df_diagnostic_data.loc[id, "id_subject"]]
        for id, a in enumerate(positives)
    ]
    negatives_weighted = [
        a * array_weights_id.numpy()[df_diagnostic_data.loc[id, "id_subject"]]
        for id, a in enumerate(negatives)
    ]
    anchors_weighted = torch.vstack(anchors_weighted)
    positives_weighted = torch.vstack(positives_weighted)
    negatives_weighted = torch.vstack(negatives_weighted)
    # compute similarities for every triplet
    sims_avg = compute_similarities(anchors, positives, negatives, method="odd_one_out")
    sims_id = compute_similarities(
        anchors_weighted, positives_weighted, negatives_weighted, method="odd_one_out"
    )
    # mark correct and incorrect decisions given similarities (argmax)
    one_avg = (sims_avg[0] > sims_avg[1]).numpy() & (sims_avg[0] > sims_avg[2]).numpy()
    one_id = (sims_id[0] > sims_id[1]).numpy() & (sims_id[0] > sims_id[2]).numpy()
    # and add back into df
    df_diagnostic_data["correct_avg"] = one_avg
    df_diagnostic_data["correct_id"] = one_id
    # group by triplet and calculate prop correct
    df_items_delta = (
        df_diagnostic_data.groupby("triplet_id")
        .agg(
            correct_avg=("correct_avg", "mean"),
            correct_id=("correct_id", "mean"),
        )
        .reset_index()
    )
    # calculate prop improvement: delta
    df_items_delta["delta"] = (
        df_items_delta["correct_id"] - df_items_delta["correct_avg"]
    )
    df_items_delta["Accuracy Avg. Model"] = pd.cut(
        df_items_delta["correct_avg"], 10, labels=False
    )
    return df_items_delta


def gini(x):
    x = np.array(x, dtype=np.float64)
    if np.amin(x) < 0:
        raise ValueError("Values cannot be negative")
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))


def max_epoch(l_files):
    """extract the epoch saved last"""
    the_number = []
    for f in l_files:
        # extract number and store
        mtch = re.search(r"epoch(\d{4})\.txt$", f)
        try:
            the_number.append(mtch.groups()[0])
        except:
            the_number.append("0")
    the_number_int = [int(n) for n in the_number]
    max_value = max(the_number_int)
    indices = [
        index for index, value in enumerate(the_number_int) if value == max_value
    ]
    file_to_extract = l_files[indices[0]]
    file_to_extract = file_to_extract.replace(".txt", ".tar")
    file_to_extract = file_to_extract.replace("sparse_embed", "model")
    return file_to_extract


def extract_results_id(
    lmbda,
    lmbda_hierarchical,
    rnd_seed,
    modelversion,
    l_n,
    l_sparse,
    l_subjecttype,
    l_splithalf,
    modeltype="weightsonly_only_weights",
    l_temperature=[],
):
    """load the results from the ID model from disk"""
    all_dirs = []
    l_all_results = []
    l_all_models = []
    l_sparsity = []
    l_subject = []
    for n in l_n:
        for splithalf in l_splithalf:
            for la in lmbda:
                for la_h in lmbda_hierarchical:
                    for sp in l_sparse:
                        for st in l_subjecttype:
                            if l_temperature == []:
                                results_dir_ID = os.path.join(
                                    "./results",
                                    modelversion,
                                    f"modeltype_{modeltype}",
                                    f"splithalf_{splithalf}",
                                    f"{n}d",
                                    str(la),
                                    str(la_h),
                                    sp,
                                    st,
                                    f"seed{rnd_seed}",
                                )
                                all_dirs.append(results_dir_ID)
                                l_sparsity.append(sp)
                                l_subject.append(st)
                            else:
                                for temp in l_temperature:
                                    results_dir_ID = os.path.join(
                                        "./results",
                                        modelversion,
                                        f"modeltype_{modeltype}",
                                        f"splithalf_{splithalf}",
                                        f"temperature_{temp}",
                                        f"{n}d",
                                        str(la),
                                        str(la_h),
                                        sp,
                                        st,
                                        f"seed{rnd_seed}",
                                    )
                                    all_dirs.append(results_dir_ID)
                                    l_sparsity.append(sp)
                                    l_subject.append(st)
    for i, d in enumerate(all_dirs):
        file_path = os.path.join(d, "results.json")
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                l_results = json.load(f)
                l_all_results.append(l_results)
        else:
            print(file_path + ": not found")
        l_files = os.listdir(results_dir_ID)
        latest_epoch = max_epoch(l_files)
        p = os.path.join(d, "model", latest_epoch)
        if os.path.isfile(p):
            m = torch.load(p, weights_only=True, map_location=torch.device("cpu"))
            m["sparsity"] = l_sparsity[i]
            m["subject_type"] = l_subject[i]
            l_all_models.append(m)
        else:
            print(f"{p} does not exist")
    return l_all_models


def model_detail_dict(ls, embedding_only):
    """extract model details as a df and append in a list"""
    l_df = [
        pd.DataFrame(
            {
                "modeltype": l["modeltype"],
                "sparsity": l["sparsity"],
                "subject_type": l["subject_type"],
                "nr_epochs": l["epoch"],
                "ndim": l["n_embed"],
                "lambda": l["lambda"],
                "lambda_hierarchical": l["lmbda_hierarchical"],
                "train_acc_max": l["train_accs_max"],
                "val_acc_max": l["val_accs_max"],
                "train_acc_proba": l["train_accs_proba"],
                "val_acc_proba": l["val_accs_proba"],
            }
        )
        for l in ls
    ]
    if embedding_only:
        for i, l in enumerate(l_df):
            l["temperature"] = ls[i]["temperature"].detach().numpy()
    return l_df


def extract_split_half(
    lmbda,
    lmbda_hierarchical,
    l_rnd_seed,
    modelversion,
    l_n,
    l_sparse,
    l_subjecttype,
    l_splithalf,
    modeltype="weightsonly_only_weights",
    l_temperature=[],
):
    """load models run on different data splits from disk"""
    all_dirs = []
    l_all_results = []
    l_all_models = []
    l_sparsity = []
    l_subject = []
    l_all_splithalf = []
    l_all_rnd_seed = []
    for rnd_seed in l_rnd_seed:
        for n in l_n:
            for splithalf in l_splithalf:
                for la in lmbda:
                    for la_h in lmbda_hierarchical:
                        for sp in l_sparse:
                            for st in l_subjecttype:
                                if l_temperature == []:
                                    results_dir_ID = os.path.join(
                                        "./results",
                                        modelversion,
                                        f"modeltype_{modeltype}",
                                        f"splithalf_{splithalf}",
                                        f"{n}d",
                                        str(la),
                                        str(la_h),
                                        sp,
                                        st,
                                        f"seed{rnd_seed}",
                                    )
                                else:
                                    for temp in l_temperature:
                                        results_dir_ID = os.path.join(
                                            "./results",
                                            modelversion,
                                            f"modeltype_{modeltype}",
                                            f"splithalf_{splithalf}",
                                            f"temperature_{temp}",
                                            f"{n}d",
                                            str(la),
                                            str(la_h),
                                            sp,
                                            st,
                                            f"seed{rnd_seed}",
                                        )
                                all_dirs.append(results_dir_ID)
                                l_sparsity.append(sp)
                                l_subject.append(st)
                                l_all_rnd_seed.append(rnd_seed)
                                l_all_splithalf.append(splithalf)
    for i, d in enumerate(all_dirs):
        file_path = os.path.join(d, "results.json")
        l_files = os.listdir(results_dir_ID)
        latest_epoch = max_epoch(l_files)
        p = os.path.join(d, "model", latest_epoch)
        if os.path.isfile(p):
            m = torch.load(p, weights_only=True, map_location=torch.device("cpu"))
            if modeltype in (
                "random_weights_free_scaling",
                "random_weights_random_scaling",
            ):
                item_embeddings = m["model_state_dict"]["model1.fc.weight"]
                decision_weights = m["model_state_dict"][
                    "model1.individual_slopes.weight"
                ]
                temperature_scalings = m["model_state_dict"][
                    "model2.individual_temps.weight"
                ]
                dict_out = {
                    "item_embeddings": item_embeddings,
                    "decision_weights": decision_weights,
                    "temperature_scalings": temperature_scalings,
                }
            else:
                item_embeddings = m["model_state_dict"]["fc.weight"]
                decision_weights = m["model_state_dict"]["individual_slopes.weight"]
                dict_out = {
                    "item_embeddings": item_embeddings,
                    "decision_weights": decision_weights,
                }
            dict_out["modeltype"] = m["modeltype"]
            dict_out["lmbda"] = m["lambda"]
            dict_out["lmbda_hierarchical"] = m["lmbda_hierarchical"]
            dict_out["n_embed"] = m["n_embed"]
            dict_out["splithalf"] = l_all_splithalf[i]
            dict_out["rnd_seed"] = l_all_rnd_seed[i]

            l_all_models.append(dict_out)
        else:
            print(f"{p} does not exist")
    return l_all_models


def extract_decision_weights(l, tp):
    """extract decision weights from a model with by-participant softmax"""
    df_sh = pd.DataFrame(l["decision_weights"].detach().numpy()).reset_index(drop=False)
    df_sh = pd.melt(
        df_sh, id_vars=["index"], var_name="dimension", value_name="decision_weight"
    )
    df_sh["timepoint"] = tp
    df_sh.rename(columns={"index": "id"}, inplace=True)
    return df_sh


def split_half_reliabilities(l_splithalf, idxs, ndims):
    """extract and plot split-half reliabilities"""

    df_sh1 = extract_decision_weights(l_splithalf[idxs[0]], 1)
    df_sh2 = extract_decision_weights(l_splithalf[idxs[1]], 2)
    df_sh = pd.merge(
        df_sh1, df_sh2, how="left", on=["id", "dimension"], suffixes=["_1", "_2"]
    )
    # Create a faceted scatterplot: one for each day

    def scatter_with_corr(data, x, y, **kwargs):
        r, _ = np.corrcoef(data[x], data[y])[0, 1], None
        z_val = data["dimension"].iloc[0]  # safely grab the facet value
        sns.scatterplot(data=data, x=x, y=y, **kwargs)
        plt.title(f"dimension = {z_val}, r = {r:.2f}")
        # Plot identity line
        min_val = min(data[x].min(), data[y].min())
        max_val = max(data[x].max(), data[y].max())
        _ = plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="gray",
            linestyle="--",
            linewidth=1,
        )

    g = sns.FacetGrid(df_sh, col="dimension", col_wrap=5)
    g.map_dataframe(scatter_with_corr, x="decision_weight_1", y="decision_weight_2")

    df_corr = pd.DataFrame(
        df_sh.groupby("dimension")
        .apply(lambda g: g["decision_weight_1"].corr(g["decision_weight_2"]))
        .sort_values(ascending=False)
    )
    df_corr["ndims"] = ndims
    df_corr.columns = ["r", "ndims"]

    return df_sh, df_corr, g


def extract_image(l_concepts_filtered, i):
    """load the relevant image from all images from disk"""

    imagename = l_concepts_filtered[i]
    path = os.path.join("data", "images", imagename)
    all_dirs = os.listdir(path)

    l_names = [re.match("^[a-z]", all_dirs[i]) for i in range(0, len(all_dirs))]
    l_names_filter = [l_names[i] != None for i in range(0, len(l_names))]
    l_names_filtered = [value for value, flag in zip(all_dirs, l_names_filter) if flag]
    just_first = l_names_filtered[0]
    path_keep = os.path.join(path, just_first)
    return path_keep


def index_ndim_split(l, ndim, splitnr):
    """
    Find the index of the dictionary in the list `l` that matches the given
    embedding dimensionality (`ndim`) and split identifier (`splitnr`).

    Parameters:
        l (list of dict): A list containing split metadata dictionaries.
        ndim (int): The embedding dimensionality to match.
        splitnr (str): The split identifier to match, e.g. "1" or "2".

    Returns:
        int: Index of the matching split in the list.
    """
    l_idxs = []
    for i, l in enumerate(l):
        filt_embed = l["n_embed"] == ndim
        filt_split = l["splithalf"] == splitnr
        if filt_embed & filt_split:
            l_idxs.append(i)
    return l_idxs


def iterate_similarity(embed1, embed2, ndim):
    """
    Compute cosine similarity between every pair of dimensions from two embedding matrices.

    Parameters:
        embed1 (ndarray): Embedding matrix from split 1 (shape: [n_items, ndim]).
        embed2 (ndarray): Embedding matrix from split 2 (shape: [n_items, ndim]).
        ndim (int): Number of embedding dimensions.

    Returns:
        ndarray: A (ndim x ndim) matrix of cosine similarities.
    """
    similarities = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            similarities[i, j] = cosine_similarity([embed1[:, i]], [embed2[:, j]])[0, 0]
    return similarities


def dimensional_similarities(l, ndim):
    """
    Calculate the cosine similarity matrix of embedding dimensions
    between two splits for the specified dimensionality.

    Parameters:
        l (list of dict): A list containing metadata and embeddings for each split.
        ndim (int): The dimensionality of embeddings to compare.

    Returns:
        tuple:
            - similarities (ndarray): Cosine similarity matrix.
            - idx1 (int): Index of split "1".
            - idx2 (int): Index of split "2".
    """
    l_similarities = []
    l_idx1 = index_ndim_split(l, ndim, "1")
    l_idx2 = index_ndim_split(l, ndim, "2")
    for idx1, idx2 in zip(l_idx1, l_idx2):
        embed1 = l[idx1]["item_embeddings"].t().numpy()
        embed2 = l[idx2]["item_embeddings"].t().numpy()
        similarities = iterate_similarity(embed1, embed2, ndim)
        l_similarities.append(similarities)
    return l_similarities, l_idx1, l_idx2


def reorder_dimensions(l, ndim):
    """
    Reorder the embedding dimensions in split "2" so they best match the ordering
    of split "1", based on maximum cosine similarity.

    Parameters:
        l (list of dict): A list containing metadata and embeddings for each split.
        ndim (int): The number of embedding dimensions to reorder.

    Returns:
        ndarray: Index mapping from split "2" dimensions to split "1" dimensions.
    """
    l_max_sims = []
    l_sims_dimensionality, l_idx1, l_idx2 = dimensional_similarities(l, ndim)
    for sims_dimensionality, idx1, idx2 in zip(l_sims_dimensionality, l_idx1, l_idx2):
        max_sims = np.argmax(sims_dimensionality, axis=1)
        l[idx2]["item_embeddings"] = l[idx2]["item_embeddings"][:, max_sims]
        l[idx2]["decision_weights"] = l[idx2]["decision_weights"][:, max_sims]
        l_max_sims.append(max_sims)
    return l_max_sims, l_idx1, l_idx2


def max_sim_dimensions_per_ndim(l_max_sims, range_dims):
    """
    Identifies the index of the simulation result for each dimensionality in `range_dims`
    where the number of unique values equals the number of dimensions.
    I.e., outputs None when there is at least one dimension in the second half
    that is maximally similar to at least two dimensions in the first half

    Parameters:
        l_max_sims (list): A list of simulation result tuples. Each tuple contains:
                           - [0]: list of arrays with shape (ndim, ...)
        range_dims (iterable): A sequence of dimensionalities to evaluate (e.g., range(2, 11)).

    Returns:
        dict: A dictionary mapping each dimensionality in `range_dims` to the index of the
              simulation that satisfies the uniqueness condition. If no match is found,
              the value is None.
    """

    dict_idxs_use = {i: None for i in range_dims}
    for idx_outer, max_sims in enumerate(l_max_sims):
        for idx_inner, ms in enumerate(max_sims[0]):
            ndim = ms.shape[0]
            n_unique = len(np.unique_values(ms))
            if ndim == n_unique:
                dict_idxs_use[idx_outer + 2] = idx_inner
                break
    return dict_idxs_use


def max_sim_results_per_ndim(dict_idxs_use, l_max_sims, range_dims):
    """
    Returns indexes for the list of simulation results where each dimension
    in the second half is maximally correlated with a unique dimension in the first half.

    Parameters:
        dict_idxs_use (dict): A dictionary mapping dimensionality to the index of the simulation to use.
        l_max_sims (list): A list of simulation result tuples. Each tuple contains:
                           - [1]: first half results
                           - [2]: second half results
        range_dims (iterable): A sequence of dimensionalities to include in the output.

    Returns:
        dict: A dictionary mapping each dimensionality in `range_dims` to a list of two arrays:
              one from the first half and one from the second half of the simulation data.
              If no index is specified for a dimensionality, the list will be empty.
    """
    dict_idxs_list_use = {i: [] for i in range_dims}
    for ndim, idx in dict_idxs_use.items():
        if idx is not None:
            dict_idxs_list_use[ndim].append(
                l_max_sims[ndim - 2][1][idx]
            )  # idxs first half
            dict_idxs_list_use[ndim].append(
                l_max_sims[ndim - 2][2][idx]
            )  # idxs second half
    return dict_idxs_list_use


def max_cors(range_dims, dict_idxs_use, l_cors):
    """
    Computes the maximum correlation values across simulations for each dimensionality.

    Parameters:
        range_dims (iterable): A sequence of dimensionalities to evaluate (e.g., range(2, 11)).
        dict_idxs_use (dict): A dictionary mapping dimensionality to the index of the simulation to use.
        l_cors (list): A list of correlation result tuples. Each tuple contains:
                       - [0]: list of correlation matrices for each simulation.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a dimensionality in `range_dims`,
                      and each row contains the maximum correlation values for the selected simulation.
                      If no index is specified for a dimensionality, the column will contain NaNs.
    """
    dict_cors = {i: None for i in range(2, 11)}
    for k, v in dict_idxs_use.items():
        if v is not None:
            tmp = l_cors[k - list(dict_cors.keys())[0]][0][v]
            dict_cors[k] = tmp.max(axis=0)
    df_cors = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict_cors.items()]))
    return df_cors


def load_ID_lowdim(
    l_lmbda,
    l_lmbda_hierarchical,
    l_rnd_seed,
    l_embed_dim,
    l_sparse,
    modelversion,
    modeltype="random_weights_random_scaling",
):
    """
    Load trained model runs from disk based on combinations of hyperparameters.

    This function constructs paths to saved model results using combinations of
    lambda values, embedding dimensions, sparsity settings, and random seeds.
    It then loads the latest model checkpoint from each path and extracts relevant
    parameters and weights.

    Parameters:
        l_lmbda (list[float]): List of regularization strengths (λ).
        l_lmbda_hierarchical (list[float]): List of hierarchical regularization strengths (λ_h).
        l_rnd_seed (list[int]): List of random seeds used during training.
        l_embed_dim (list[int]): List of embedding dimensions (e.g., 10, 50, 100).
        l_sparse (list[str]): List of sparsity configuration names (e.g., "sparse", "dense").
        modelversion (str): Version identifier for the model (used in directory structure).
        modeltype (str, optional): Type of model architecture or training scheme.
            Defaults to "random_weights_random_scaling".

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - item_embeddings (Tensor): Learned item embedding weights.
            - decision_weights (Tensor): Individual slope weights.
            - temperature_scalings (Tensor): Individual temperature scaling weights.
            - modeltype (str): Model type used during training.
            - lmbda (float): Regularization strength used.
            - lmbda_hierarchical (float): Hierarchical regularization strength used.
            - n_embed (int): Embedding dimension used.
            - rnd_seed (int): Random seed used.

    Notes:
        - If a model checkpoint is missing, a warning is printed and that run is skipped.
        - Assumes model files are stored in a structured directory under "./results".
    """
    l_dirs = []
    l_models = []
    l_sparsity = []
    l_rnd_seeds_flat = []

    for rnd_seed in l_rnd_seed:
        for n in l_embed_dim:
            for lmbda in l_lmbda:
                for lmbda_h in l_lmbda_hierarchical:
                    for sp in l_sparse:
                        results_dir_ID = os.path.join(
                            "./results",
                            modelversion,
                            f"modeltype_{modeltype}",
                            f"{n}d",
                            str(lmbda),
                            str(lmbda_h),
                            sp,
                            f"seed{rnd_seed}",
                        )
                        l_dirs.append(results_dir_ID)
                        l_sparsity.append(sp)
                        l_rnd_seeds_flat.append(rnd_seed)

    for i, d in enumerate(l_dirs):
        l_files = os.listdir(d)
        latest_epoch = max_epoch(l_files)
        model_path = os.path.join(d, "model", latest_epoch)

        if os.path.isfile(model_path):
            m = torch.load(
                model_path, weights_only=True, map_location=torch.device("cpu")
            )
            dict_out = {
                "item_embeddings": m["model_state_dict"]["model1.fc.weight"],
                "decision_weights": m["model_state_dict"][
                    "model1.individual_slopes.weight"
                ],
                "temperature_scalings": m["model_state_dict"][
                    "model2.individual_temps.weight"
                ],
                "modeltype": m["modeltype"],
                "lmbda": m["lambda"],
                "lmbda_hierarchical": m["lmbda_hierarchical"],
                "n_embed": m["n_embed"],
                "rnd_seed": l_rnd_seeds_flat[i],
            }
            l_models.append(dict_out)
        else:
            print(f"{model_path} does not exist")

    return l_models


def reverse_code_big5(df_qs_num):
    """
    Reverse-codes specific items from the Big Five Short Form questionnaire.

    Parameters:
        df_qs_num (pd.DataFrame): A DataFrame containing numeric responses to the Big Five items.
                                  Columns should be named in the format 'BIG5_SF_{i}' where i is the item index (0-based).

    Returns:
        pd.DataFrame: The input DataFrame with specified Big Five items reverse-coded.
                      Reverse coding is done by subtracting the original value from 4.
    """
    # Reverse-coded item indices
    reverse_items_qs = [1, 21, 26, 7, 17, 27, 3, 8, 28, 14, 19, 24, 29, 10, 20, 30]
    reverse_items_python = [re - 1 for re in reverse_items_qs]

    # Reverse-code items
    for i in reverse_items_python:
        col = f"BIG5_SF_{i}"
        df_qs_num[col] = 4 - df_qs_num[col]

    return df_qs_num


def reverse_code_i8(df_qs_num):
    """
    Reverse-codes specific items from the I-8 personality inventory.

    Parameters:
        df_qs_num (pd.DataFrame): A DataFrame containing numeric responses to the I-8 items.
                                  Columns should be named in the format 'I_8_{i}' where i is the item index (0-based).

    Returns:
        pd.DataFrame: The input DataFrame with specified I-8 items reverse-coded.
                      Reverse coding is done by subtracting the original value from 4.
    """
    reverse_items_qs = [3, 4, 5, 6]
    reverse_items_python = [re - 1 for re in reverse_items_qs]

    for i in reverse_items_python:
        col = f"I_8_{i}"
        df_qs_num[col] = 4 - df_qs_num[col]

    return df_qs_num


def reverse_code_q_items(df_qs_num):
    """
    Applies reverse-coding to questionnaire items across multiple instruments.

    Specifically:
    - Reverse-codes selected items from the Big Five Short Form.
    - Reverse-codes selected items from the I-8 inventory.
    - Assumes no reverse-coded items in the PID-5 Brief Form.

    Parameters:
        df_qs_num (pd.DataFrame): A DataFrame containing numeric questionnaire responses.

    Returns:
        pd.DataFrame: The input DataFrame with all applicable items reverse-coded.
    """
    df_qs_num = reverse_code_big5(df_qs_num)
    df_qs_num = reverse_code_i8(df_qs_num)
    # no reversely coded items in pid5
    return df_qs_num


def scales_and_facets_big5(df_qs_num):
    """
    Computes scale and facet scores for the Big Five Short Form (BIG5-SF) questionnaire.

    Each of the five personality traits—Extraversion, Agreeableness, Conscientiousness,
    Negative Emotionality, and Open-Mindedness—is scored by averaging responses to six items.
    Each trait also includes three facets, each scored by averaging two items.

    Parameters:
        df_qs_num (pd.DataFrame): A DataFrame containing numeric responses to BIG5-SF items.
                                  Columns must be named 'BIG5_SF_{i}' where i is the item index (0-based).

    Returns:
        pd.DataFrame: The input DataFrame with additional columns for each trait and facet score.
    """
    # Define scales and their facets
    trait_structure = {
        "Extraversion": {
            "items": [1, 6, 11, 16, 21, 26],
            "facets": {
                "Sociability": [1, 16],
                "Assertiveness": [6, 21],
                "Energy_Level": [11, 26],
            },
        },
        "Agreeableness": {
            "items": [2, 7, 12, 17, 22, 27],
            "facets": {
                "Compassion": [2, 17],
                "Respectfulness": [7, 22],
                "Trust": [12, 27],
            },
        },
        "Conscientiousness": {
            "items": [3, 8, 13, 18, 23, 28],
            "facets": {
                "Organization": [3, 18],
                "Productiveness": [8, 23],
                "Responsibility": [13, 28],
            },
        },
        "Negative_Emotionality": {
            "items": [4, 9, 14, 19, 24, 29],
            "facets": {
                "Anxiety": [4, 19],
                "Depression": [9, 24],
                "Emotional_Volatility": [14, 29],
            },
        },
        "Open_Mindedness": {
            "items": [5, 10, 15, 20, 25, 30],
            "facets": {
                "Aesthetic_Sensitivity": [5, 20],
                "Intellectual_Curiosity": [10, 25],
                "Creative_Imagination": [15, 30],
            },
        },
    }

    # -1 for python logic
    trait_structure_python = copy.deepcopy(trait_structure)
    for scale, content in trait_structure_python.items():
        for idx_scale, item_scale in enumerate(content["items"]):
            content["items"][idx_scale] = item_scale - 1
        for name_facet, items_facet in content["facets"].items():
            for idx_facet, item_facet in enumerate(items_facet):
                content["facets"][name_facet][idx_facet] = item_facet - 1

    # Compute scores
    for scale, content in trait_structure_python.items():
        df_qs_num[scale] = df_qs_num[[f"BIG5_SF_{i}" for i in content["items"]]].mean(
            axis=1
        )
        for facet, items in content["facets"].items():
            df_qs_num[facet] = df_qs_num[[f"BIG5_SF_{i}" for i in items]].mean(axis=1)

    return df_qs_num


def scales_and_facets_pid5(df_qs_num):
    """
    Computes scale scores for the PID-5 Brief Form (PID5-BF) questionnaire.

    The PID-5-BF assesses five maladaptive personality traits—Negative Affect, Detachment,
    Antagonism, Disinhibition, and Psychoticism—each scored by averaging five items.

    Parameters:
        df_qs_num (pd.DataFrame): A DataFrame containing numeric responses to PID5-BF items.
                                  Columns must be named 'PID5_BF_{i}' where i is the item index (0-based).

    Returns:
        pd.DataFrame: The input DataFrame with additional columns for each PID-5 trait score.
    """
    # Define scales and their facets
    trait_structure = {
        "Negative Affect": {
            "items": [8, 9, 10, 11, 15],
        },
        "Detachment": {
            "items": [4, 13, 14, 16, 18],
        },
        "Antagonism": {
            "items": [17, 19, 20, 22, 25],
        },
        "Disinhibition": {
            "items": [1, 2, 3, 5, 6],
        },
        "Psychoticism": {
            "items": [7, 12, 21, 23, 24],
        },
    }

    # -1 for python logic
    trait_structure_python = copy.deepcopy(trait_structure)
    for scale, content in trait_structure_python.items():
        for idx_scale, item_scale in enumerate(content["items"]):
            content["items"][idx_scale] = item_scale - 1

    # Compute scores
    for scale, content in trait_structure_python.items():
        df_qs_num[scale] = df_qs_num[[f"PID5_BF_{i}" for i in content["items"]]].mean(
            axis=1
        )

    return df_qs_num


def scales_and_facets(df_qs_num):
    """
    Computes scale and facet scores for both the Big Five Short Form and PID-5 Brief Form questionnaires.

    This function applies:
    - `scales_and_facets_big5()` to compute Big Five traits and facets.
    - `scales_and_facets_pid5()` to compute PID-5 maladaptive trait scores.

    Parameters:
        df_qs_num (pd.DataFrame): A DataFrame containing numeric responses to both BIG5-SF and PID5-BF items.

    Returns:
        pd.DataFrame: The input DataFrame with added columns for all trait and facet scores.
    """
    df_qs_num = scales_and_facets_big5(df_qs_num)
    df_qs_num = scales_and_facets_pid5(df_qs_num)
    return df_qs_num


def keep_scales_only(df_qs_num):
    """
    Filters a DataFrame to retain only scale-level columns by removing item-level and facet-level scores.

    Parameters:
    -----------
    df_qs_num : pandas.DataFrame
        A DataFrame containing quantitative scores, including item-level, facet-level, and scale-level columns.

    Returns:
    --------
    pandas.DataFrame
        A filtered DataFrame containing only the columns that do not start with predefined item or facet prefixes.

    Notes:
    ------
    The function excludes columns that start with any of the following prefixes:
    - "BIG5_SF_", "PID5_BF_", "I_8_"
    - Specific facet names such as 'Sociability', 'Assertiveness', 'Energy_Level', etc.

    This is useful for isolating higher-order scale scores from detailed questionnaire data.
    """
    col_starts = (
        "BIG5_SF_",
        "PID5_BF_",
        "I_8_",
        "Sociability",
        "Assertiveness",
        "Energy_Level",
        "Compassion",
        "Respectfulness",
        "Trust",
        "Organization",
        "Productiveness",
        "Responsibility",
        "Anxiety",
        "Depression",
        "Emotional_Volatility",
        "Aesthetic_Sensitivity",
        "Intellectual_Curiosity",
        "Creative_Imagination",
    )
    all_cols = df_qs_num.columns
    idxs_keep = [not c.startswith(col_starts) for c in all_cols]
    df_qs_num = df_qs_num[all_cols[idxs_keep]]

    return df_qs_num


def load_model(model_id):
    """
    Loads a pretrained transformer model and tokenizer from Hugging Face.

    Automatically selects GPU if available, otherwise defaults to CPU.

    Parameters:
        model_id (str): The identifier of the pretrained model to load (e.g., "bert-base-uncased").

    Returns:
        tuple:
            - model (torch.nn.Module): The loaded transformer model.
            - tokenizer (transformers.PreTrainedTokenizer): The corresponding tokenizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    return model, tokenizer, device


def add_prefixes_and_concatenate_cols(df_qs_txt):
    """
    Adds predefined textual prefixes to a questionnaire DataFrame and concatenates selected columns into a single text string.

    This function:
    - Adds five prefix columns describing life, work history, and interests.
    - Concatenates these prefixes with user-provided text fields into a new column 'txt_concat'.

    Parameters:
        df_qs_txt (pd.DataFrame): A DataFrame containing text fields such as 'workHistory', 'interests1', 'interests2', and 'interests3'.

    Returns:
        pd.DataFrame: The input DataFrame with added prefix columns and a new 'txt_concat' column containing the full concatenated text.
    """
    # add prefix cols
    prefix = "Here are some details about my life: "
    work_history_prefix = "I have worked as "
    interests1_prefix = ". The topics or activities I am most passionate about are: "
    interests2_prefix = ". The topics or activities I love doing in my free time are: "
    interests3_prefix = (
        ". The subjects, which fascinate me and make me want to learn more are: "
    )
    df_qs_txt["prefix"] = prefix
    df_qs_txt["work_history_prefix"] = work_history_prefix
    df_qs_txt["interests1_prefix"] = interests1_prefix
    df_qs_txt["interests2_prefix"] = interests2_prefix
    df_qs_txt["interests3_prefix"] = interests3_prefix

    # concatenate text cols
    df_qs_txt["txt_concat"] = df_qs_txt[
        [
            "prefix",
            "work_history_prefix",
            "workHistory",
            "interests1_prefix",
            "interests1",
            "interests2_prefix",
            "interests2",
            "interests1_prefix",
            "interests3",
        ]
    ].agg("".join, axis=1)

    return df_qs_txt


def tokenize_col(txt, prefix1, prefix2, tokenizer, model, device):
    """
    Tokenizes a concatenated text string using a Hugging Face tokenizer and extracts the embedding of the [CLS] token.

    Parameters:
        txt (str): The main text content to be tokenized.
        prefix1 (str): A prefix string to prepend to the text.
        prefix2 (str): A second prefix string to prepend after prefix1.
        tokenizer (transformers.PreTrainedTokenizer): A Hugging Face tokenizer instance.

    Returns:
        np.ndarray: A NumPy array representing the embedding of the [CLS] token from the model's output.

    Notes:
        - Assumes a global variable `model` is already loaded and accessible.
        - Assumes a global variable `device` is defined for model execution.
    """
    all_together = prefix1 + prefix2 + txt
    tokenized_input = tokenizer.encode(all_together, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(tokenized_input)
    e = output.last_hidden_state[:, 0, :].numpy()

    return e
