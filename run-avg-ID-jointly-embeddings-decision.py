#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Mirko Thalmann 2024/2025 Helmholtz Munich, based on Code written by:
# Author: Lukas Muttenthaler
# Copyright 2020 Max Planck Institute for Human Cognitive and Brain Sciences


import argparse
import json
import logging
import os
import random
import re
from turtle import distance
import torch
import warnings
import pickle

from tqdm import tqdm
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from os.path import join as pjoin
from collections import defaultdict
from scipy.stats import linregress
from torch.optim import Adam, AdamW

import plotting as pl
from models import model as md
import utils as ut


os.environ['PYTHONIOENCODING'] = 'UTF-8'
os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--task', type=str, default='odd_one_out',
        choices=['odd_one_out', 'similarity_task'])
    aa('--loggername', type=str, default='avg-ID-joint-logger',
       help='name of the logger to be used')
    aa('--triplets_dir', type=str,
        help='directory from where to load triplets')
    aa('--modeltype', type=str, default="random_weights_free_scaling",
        choices=["random_weights_free_scaling"], help='only by-participant slopes or by-participant intercepts as well')
    aa('--results_dir', type=str, default='./results/',
        help='optional specification of results directory (if not provided will resort to ./results/lambda/rnd_seed/)')
    aa('--plots_dir', type=str, default='./plots/',
        help='optional specification of directory for plots (if not provided will resort to ./plots/lambda/rnd_seed/)')
    aa('--learning_rate', type=float, default=0.001,
        help='learning rate to be used in optimizer')
    aa('--lmbda', type=float,
        help='lambda value determines weight of l1-regularization')
    aa('--lmbda_hierarchical', type=float,
        help='value determining weight of gaussian regularization on individual weights')
    aa('--temperature', type=float, default=1.,
        help='softmax temperature (beta param) for choice randomness')
    aa('--embed_dim', metavar='D', type=int, default=90,
        help='dimensionality of the embedding matrix')
    aa('--batch_size', metavar='B', type=int, default=100,
        choices=[16, 25, 32, 50, 64, 100, 128, 150, 200, 256],
        help='number of triplets in each mini-batch')
    aa('--epochs', metavar='T', type=int, default=500,
        help='maximum number of epochs to optimize SPoSE model for')
    aa('--window_size', type=int, default=50,
        help='window size to be used for checking convergence criterion with linear regression')
    aa('--steps', type=int, default=10,
        help='save model parameters and create checkpoints every <steps> epochs')
    aa('--sampling_method', type=str, default='normal',
        choices=['normal', 'soft'],
        help='whether random sampling of the entire training set or soft sampling of some fraction of the training set will be performed during each epoch')
    aa('--p', type=float, default=None,
        choices=[None, 0.5, 0.6, 0.7, 0.8, 0.9],
        help='this argument is only necessary for soft sampling. specifies the fraction of *train* to be sampled during an epoch')
    aa('--resume', action='store_true',
        help='whether to resume training at last checkpoint; if not set training will restart')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    aa('--distance_metric', type=str, default='dot',
       choices=['dot', 'euclidean'], help='distance metric')
    aa('--early_stopping', type=str, default='No',
       choices=["No", "Yes"], help='early stopping')
    aa('--num_threads', type=int, default=20,
       help='number of threads used by PyTorch multiprocessing')
    aa('--use_shuffled_subjects', type=str, default='actual',
        choices=['actual', 'shuffled'], help='actual subjects or subjects with randomly shuffled trials from all subjects')
    aa('--sparsity', type=str, 
        choices=['both', 'items', 'items_and_random_ids'], help='sparsity constraint setting')
    aa('--splithalf', type=str, 
        choices=['1', '2', 'no'], help='use first or second half of trials or use default train test split')
    args = parser.parse_args()
    return args


def setup_logging(file: str, dir: str = './log_files/', loggername: str = "ID-sem-reps"):
    if not os.path.exists(dir):
        os.makedirs(dir)
    # create logger at root level (no need to provide specific name, as our logger won't have children)
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.INFO)
    # add console handler to logger
    if len(logger.handlers) < 1:
        # only file handler, no console handler
        handler = logging.FileHandler(os.path.join(dir, file))
        handler.setLevel(logging.INFO)

        # create formatter to configure order, structure, and content of log messages
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S'
        )  # add formatter to handler
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def run(
        task: str,
        loggername: str,
        rnd_seed: int,
        results_dir: str,
        plots_dir: str,
        triplets_dir: str,
        modeltype: str,
        device: torch.device,
        batch_size: int,
        embed_dim: int,
        epochs: int,
        window_size: int,
        sampling_method: str,
        lmbda: float,
        lmbda_hierarchical: float,
        lr: float,
        steps: int,
        early_stopping: str = "No",
        p: float = None,
        resume: bool = False,
        show_progress: bool = True,
        distance_metric: str = 'dot',
        temperature: float = 1.,
        use_shuffled_subjects: str = 'actual',
        sparsity: str = "both",
        splithalf: str = "no"
):
    # initialise logger and start logging events
    logger = setup_logging(file='avg-ID-jointly-embeddings-decision.log',
                           dir=f'./log_files/{modeltype}/splithalf_{splithalf}/ndim_{embed_dim}/lr_{lr}/lmbda_{lmbda}/lmbda_hierarchical_{lmbda_hierarchical}/sparsity_{sparsity}/{use_shuffled_subjects}_subjects', loggername=loggername)
    logger.info("modeltype = ", f'{modeltype}')
    # load triplets into memory
    train_triplets_ID, test_triplets_ID = ut.load_data_ID(
        device=device, triplets_dir=triplets_dir, testcase=False, use_shuffled_subjects=use_shuffled_subjects, splithalf=splithalf)
    n_items_ID = ut.get_nitems(train_triplets_ID)
    logger.info("n_items = " + str(n_items_ID))

    # load train and test mini-batches
    n_participants = len(np.unique(train_triplets_ID.numpy()[:, 3]))
    train_batches, val_batches = ut.load_batches(
        train_triplets=train_triplets_ID,
        test_triplets=test_triplets_ID,
        n_items=n_items_ID,
        batch_size=batch_size,
        sampling_method=sampling_method,
        rnd_seed=rnd_seed,
        p=p, method="ids"
    )
    logger.info(
        f'\nNumber of train batches in current process: {len(train_batches)}\n')

    ###############################
    ########## settings ###########
    ###############################

    temperature = torch.tensor(temperature).clone().detach()
    if modeltype == "random_weights_free_scaling":
        model = md.CombinedModel(
            in_size=n_items_ID, out_size=embed_dim,
            num_participants=n_participants, init_weights=True
        )
    else:
        # so far, only random weights and free scaling implemented
        exit()
    
    model.to(device)
    optim = Adam(model.parameters(), lr=lr)

    ################################################
    ############# Creating PATHs ###################
    ################################################

    logger.info(f'...Creating PATHs')
    if results_dir == './results/':
        results_dir = os.path.join(
            results_dir, "avg-ID-jointly-embeddings-decision", f'modeltype_{modeltype}', f'splithalf_{splithalf}', f'{embed_dim}d', str(lmbda), str(lmbda_hierarchical), sparsity, f'subjects_{use_shuffled_subjects}', f'seed{rnd_seed}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if plots_dir == './plots/':
        plots_dir = os.path.join(
            plots_dir, "avg-ID-jointly-embeddings-decision", f'modeltype_{modeltype}', f'splithalf_{splithalf}', f'{embed_dim}d', str(lmbda), str(lmbda_hierarchical), sparsity, f'subjects_{use_shuffled_subjects}', f'seed{rnd_seed}')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    model_dir = os.path.join(results_dir, 'model')

    #####################################################################
    ######### Load model from previous checkpoint, if available #########
    #####################################################################

    if resume:
        if os.path.exists(model_dir):
            models = sorted([m.name for m in os.scandir(
                model_dir) if m.name.endswith('.tar')])
            if len(models) > 0:
                try:
                    PATH = os.path.join(model_dir, models[-1])
                    map_location = device
                    checkpoint = torch.load(PATH, map_location=map_location)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optim.load_state_dict(checkpoint['optim_state_dict'])
                    start = checkpoint['epoch'] + 1
                    loss = checkpoint['loss']
                    train_accs_max = checkpoint['train_accs_max']
                    train_accs_proba = checkpoint['train_accs_proba']
                    val_accs_max = checkpoint['val_accs_max']
                    val_accs_proba = checkpoint['val_accs_proba']
                    train_losses = checkpoint['train_losses']
                    val_losses = checkpoint['val_losses']
                    nneg_d_over_time = checkpoint['nneg_d_over_time']
                    loglikelihoods = checkpoint['loglikelihoods']
                    complexity_losses = checkpoint['complexity_costs']
                    print(
                        f'...Loaded model and optimizer state dicts from previous run. Starting at epoch {start}.\n')
                except RuntimeError:
                    print(f'...Loading model and optimizer state dicts failed. Check whether you are currently using a different set of model parameters.\n')
                    start = 0
                    train_accs_max, train_accs_proba, val_accs_max, val_accs_proba = [], [], [], []
                    train_losses, val_losses = [], []
                    loglikelihoods, complexity_losses = [], []
                    nneg_d_over_time = []
            else:
                raise Exception(
                    'No checkpoints found. Cannot resume training.')
        else:
            raise Exception(
                'Model directory does not exist. Cannot resume training.')
    else:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        start = 0
        train_accs_max, train_accs_proba, val_accs_max, val_accs_proba = [], [], [], []
        train_losses, val_losses = [], []
        loglikelihoods, complexity_losses = [], []
        nneg_d_over_time = []

    ################################################
    ################## Training ####################
    ################################################

    start = 0
    train_accs_max, train_accs_proba, val_accs_max, val_accs_proba = [], [], [], []
    train_losses, val_losses = [], []
    loglikelihoods, complexity_losses_ID, complexity_losses_avg = [], [], []
    nneg_d_over_time = []

    iter = 0
    results = {}
    logger.info(f'Optimization started for lambda = {lmbda} and hierarchical lambda = {lmbda_hierarchical}\n')

    # Early stopping parameters
    patience = 10
    best_val_accuracy = 0.0
    counter = 0

    print(f'Optimization started for lambda = {lmbda} and hierarchical lambda = {lmbda_hierarchical}\n')
    for epoch in tqdm(range(start, epochs)):
        model.train()
        batch_llikelihoods = torch.zeros(len(train_batches))
        batch_closses_avg = torch.zeros(len(train_batches))
        batch_closses_ID = torch.zeros(len(train_batches))
        batch_losses_train = torch.zeros(len(train_batches))
        batch_accs_max_train = torch.zeros(len(train_batches))
        batch_accs_proba_train = torch.zeros(len(train_batches))
        for i, batch in enumerate(train_batches):
            optim.zero_grad()  # zero out gradients
            b = batch[0].to(device)
            id = batch[1].to(device)
            c_entropy, anchor, positive, negative = model(b, id, distance_metric)

            # few-dimensional representations of the items
            l1_pen_avg = md.l1_regularization(model, "fc.weight", agreement="few").to(
                device)  # L1-norm to enforce sparsity (many 0s)
            # mostly agreement with item reps, but few dimensions may be downweighted
            l1_pen_ID = md.l1_regularization(model, "individual_", agreement="most").to(
                device)  # L1-norm to enforce sparsity (many 0s)
            W = model.model1.fc.weight
            Bs = model.model1.individual_slopes.weight
            temperature = model.model2(id[::3])
            # positivity constraint to enforce non-negative values in embedding matrix
            # pos_pen = torch.sum(F.relu(-W))
            pos_pen = torch.sum(
                F.relu(-W)) + torch.sum(F.relu(-Bs))
            complexity_loss_avg = (lmbda/n_items_ID) * l1_pen_avg

            # Gaussian loss on individual differences for each dimension
            # is only computed by random model
            gaussian_pen = model.model1.hierarchical_loss(id)
            gaussian_loss = gaussian_pen * lmbda_hierarchical
            loss = c_entropy + 0.01 * pos_pen + complexity_loss_avg + gaussian_loss

            loss.backward()
            optim.step()
            batch_losses_train[i] += loss.item()
            batch_llikelihoods[i] += c_entropy.item()
            # batch_closses_ID[i] += complexity_loss_ID.item()
            batch_closses_avg[i] += complexity_loss_avg.item()
            accs_train_proba, accs_train_max = ut.choice_accuracy(
                anchor, positive, negative, task, distance_metric, scalingfactors=temperature)
            batch_accs_proba_train[i] += accs_train_proba
            batch_accs_max_train[i] += accs_train_max
            iter += 1

        avg_llikelihood = torch.mean(batch_llikelihoods).item()
        avg_closs_avg = torch.mean(batch_closses_avg).item()
        avg_train_loss = torch.mean(batch_losses_train).item()
        avg_train_acc_max = torch.mean(batch_accs_max_train).item()
        avg_train_acc_proba = torch.mean(batch_accs_proba_train).item()

        loglikelihoods.append(avg_llikelihood)
        complexity_losses_avg.append(avg_closs_avg)
        train_losses.append(avg_train_loss)
        train_accs_max.append(avg_train_acc_max)
        train_accs_proba.append(avg_train_acc_proba)

        ################################################
        ################ validation ####################
        ################################################

        avg_val_loss, avg_val_acc_proba, avg_val_acc_max = ut.validation(
            model, val_batches, task, device, 
            level_explanation="ID", modeltype=modeltype,
            temperature = torch.Tensor([0]) # is overwritten in ut.validation()
            )
        val_losses.append(avg_val_loss)
        val_accs_max.append(avg_val_acc_max)
        val_accs_proba.append(avg_val_acc_proba)

        logger.info(f'Epoch: {epoch+1}/{epochs}')
        logger.info(f'Train acc max: {avg_train_acc_max:.5f}')
        logger.info(f'Train acc proba: {avg_train_acc_proba:.5f}')
        logger.info(f'Train loss: {avg_train_loss:.5f}')
        logger.info(f'Val acc max: {avg_val_acc_max:.5f}')
        logger.info(f'Val acc proba: {avg_val_acc_proba:.5f}')
        logger.info(f'Val loss: {avg_val_loss:.5f}\n')

        if show_progress:
            print("\n========================================================================================================")
            print(
                f'====== Epoch: {epoch+1}, Train acc max: {avg_train_acc_max:.5f}, Train acc proba: {avg_train_acc_proba:.5f}, Train loss: {avg_train_loss:.5f}, \nVal acc max: {avg_val_acc_max:.5f}, Val acc proba: {avg_val_acc_proba:.5f}, Val loss: {avg_val_loss:.5f} ======')
            print("========================================================================================================\n")
            current_d = ut.get_nneg_dims(W)
            nneg_d_over_time.append((epoch+1, current_d))
            print("\n========================================================================================================")
            print(
                f"========================= Current number of non-negative dimensions: {current_d} =========================")
            print("========================================================================================================\n")

        if (epoch + 1) % steps == 0:
            W = model.model1.fc.weight
            id_slopes = model.model1.individual_slopes.weight
            id_decision_scaling = model.model2.individual_temps.weight
            np.savetxt(os.path.join(
                results_dir, f'sparse_embed_epoch{epoch+1:04d}.txt'), W.detach().cpu().numpy())
            logger.info(f'Saving model weights at epoch {epoch+1}')
            np.savetxt(os.path.join(
                results_dir, f'individual_slopes{epoch+1:04d}.txt'), id_slopes.detach().cpu().numpy())
            logger.info(f'Saving individual decision weights at epoch {epoch+1}')
            np.savetxt(os.path.join(
                results_dir, f'individual_scalings{epoch+1:04d}.txt'), id_decision_scaling.detach().cpu().numpy())
            logger.info(f'Saving individual decision scaling factors at epoch {epoch+1}')

            # save model and optim parameters for inference or to resume training
            # PyTorch convention is to save checkpoints as .tar files
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'modeltype': modeltype,
                'sparsity': sparsity,
                'subject_type': use_shuffled_subjects,
                'splithalf': splithalf,
                'n_embed': embed_dim,
                'lambda': lmbda,
                'lmbda_hierarchical': lmbda_hierarchical,
                'loss': loss,
                'train_losses': train_losses,
                'train_accs_max': train_accs_max,
                'train_accs_proba': train_accs_proba,
                'val_losses': val_losses,
                'val_accs_max': val_accs_max,
                'val_accs_proba': val_accs_proba,
                'nneg_d_over_time': nneg_d_over_time,
                'loglikelihoods': loglikelihoods,
                'complexity_costs_ID': complexity_losses_ID,
                'complexity_costs_avg': complexity_losses_avg,
            }, os.path.join(model_dir, f'model_epoch{epoch+1:04d}.tar'))

            logger.info(f'Saving model parameters at epoch {epoch+1}\n')

        if early_stopping == "Yes" and (epoch + 1) > window_size and epoch >= 100:
            # check termination condition (we want to train until convergence)
            # Early stopping check
            if avg_val_acc_max > best_val_accuracy:
                best_val_accuracy = avg_val_acc_max
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # save final model weights
    ut.save_weights_(results_dir, model.model1.fc.weight)
    results = {
        'epoch': len(train_accs_max), 
        'train_acc_max': train_accs_max[-1], 
        'val_acc_max': val_accs_max[-1], 
        'train_acc_proba': train_accs_proba[-1], 
        'val_acc_proba': val_accs_proba[-1], 
        'val_loss': val_losses[-1]
        }
    logger.info(
        f'\nOptimization finished after {epoch+1} epochs for lambda: {lmbda}\n')

    logger.info(
        f'\nPlotting number of non-negative dimensions as a function of time for lambda: {lmbda}\n')
    pl.plot_nneg_dims_over_time(
        plots_dir=plots_dir, nneg_d_over_time=nneg_d_over_time)

    logger.info(f'\nPlotting model performances over time for lambda: {lmbda}')
    # plot train and validation performance alongside each other to examine a potential overfit to the training data
    pl.plot_single_performance(
        plots_dir=plots_dir, val_accs=val_accs_max, train_accs=train_accs_max, max_or_proba="max")
    pl.plot_single_performance(
        plots_dir=plots_dir, val_accs=val_accs_proba, train_accs=train_accs_proba, max_or_proba="proba")
    logger.info(f'\nPlotting losses over time for lambda: {lmbda}')
    # plot both log-likelihood of the data (i.e., cross-entropy loss) and complexity loss (i.e., l1-norm in DSPoSE and KLD in VSPoSE)
    pl.plot_complexities_and_loglikelihoods(
        plots_dir=plots_dir, loglikelihoods=loglikelihoods, complexity_losses=complexity_losses)

    PATH = os.path.join(results_dir, 'results.json')
    with open(PATH, 'w') as results_file:
        json.dump(results, results_file)


if __name__ == "__main__":
    # parse all arguments and set random seeds
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    torch.set_num_threads(args.num_threads)

    if re.search(r'^cuda', args.device):
        device = torch.device(args.device)
        torch.cuda.manual_seed_all(args.rnd_seed)
        torch.backends.cudnn.benchmark = False
        # try:
        #     torch.cuda.set_device(int(args.device[-1]))
        # except:
        #     torch.cuda.set_device(1)
        torch.cuda.set_device(0)
        print(f'\nPyTorch CUDA version: {torch.version.cuda}\n')
    else:
        device = torch.device(args.device)

    run(
        task=args.task,
        loggername=args.loggername,
        rnd_seed=args.rnd_seed,
        results_dir=args.results_dir,
        modeltype=args.modeltype,
        plots_dir=args.plots_dir,
        triplets_dir=args.triplets_dir,
        device=device,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        window_size=args.window_size,
        sampling_method=args.sampling_method,
        lmbda=args.lmbda,
        lmbda_hierarchical=args.lmbda_hierarchical,
        lr=args.learning_rate,
        steps=args.steps,
        resume=args.resume,
        p=args.p,
        distance_metric=args.distance_metric,
        temperature=args.temperature,
        early_stopping=args.early_stopping,
        use_shuffled_subjects=args.use_shuffled_subjects,
        sparsity=args.sparsity,
        splithalf=args.splithalf
    )
