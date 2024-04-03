"""
use_alpha_3_4
=============

Implements pipeline for the experiments of the Alpha 3 and Alpha 4
models.
"""

import argparse
import json
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/")
sys.path.append(current_path + "/../../models/quantum/alpha_3_4/")

import embedder as emb
import dim_reduction as dimred
import circuit as circ

from alpha_3_4 import Alpha3, Alpha4
from save_json_output import JsonOutputer


parser = argparse.ArgumentParser()

# Model-related arguments

parser.add_argument(
    "-m",
    "--model",
    help="Choice between alpha_3 and alpha_4 model",
    type=str,
    default="alpha_3",
)
parser.add_argument(
    "-tr",
    "--train",
    help="Path to the training dataset",
    type=str,
    default="./../datasets/toy_datasets/multiclass_toy_train_sentence_bert.csv",  # Change?
)
parser.add_argument(
    "-val",
    "--validation",
    help="Path to the validation dataset",
    type=str,
    default="./../datasets/toy_datasets/multiclass_toy_validation_sentence_bert.csv",  # Change?
)
parser.add_argument(
    "-te",
    "--test",
    help="Path to the test dataset",
    type=str,
    default="./../datasets/toy_datasets/multiclass_toy_test_sentence_bert.csv",  # Change?
)
parser.add_argument(
    "-nc",
    "--n_classes",
    help="Number of classes in the classification task",
    type=int,
)
parser.add_argument(
    "-s",
    "--seed",
    help="Seed for the initial parameters",
    type=int,
    default=150298,
)
parser.add_argument(
    "-i",
    "--iterations",
    help="Number of iterations/epochs",
    type=int,
    default=150,
)
parser.add_argument(
    "-r",
    "--runs",
    help="Number of runs of the model",
    type=int,
    default=1,
)
parser.add_argument(
    "-o",
    "--output",
    help="Path of directory to store model output",
    type=str,
    default="./../../benchmarking/results/raw/",
)

# Optimiser arguments

parser.add_argument(
    "-op",
    "--optimiser",
    help="Choice of torch optimiser",
    type=str,
    default="Adam",
)
parser.add_argument(
    "-b",
    "--batch_size",
    help="Batch size",
    type=int,
    default=512,
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    help="Learning rate for the optimiser",
    type=float,
    default=0.001,
)

# Arguments for data vectorisation

parser.add_argument(
    "-e",
    "--embedder",
    help="Choice of embedder",
    type=str,
    default="Bert",
)
parser.add_argument(
    "-dr",
    "--dim_reduction",
    help="Choice of dimensionality reduction mechanism",
    type=str,
    default="PCA",
)

# Arguments for quantum circuit architecture

parser.add_argument(
    "-c",
    "--circuit",
    help="Choice of quantum circuit architecture",
    type=str,
    default="Sim14",
)
parser.add_argument(
    "-nq",
    "--n_qubits",
    help="Number of qubits in the circuit",
    type=int,
    default=3,
)
parser.add_argument(
    "-nl",
    "--n_layers",
    help="Number of layers in the circuit",
    type=int,
    default=1,
)
parser.add_argument(
    "-qi",
    "--qubit_init",
    help="",  # Need clarification here
    type=str,
)
parser.add_argument(
    "-res",
    "--data_rescaling",
    help="",  # Need clarification
    type=str,
)

# Arguments for MLP layer (Alpha 3 only)

parser.add_argument(
    "-mi",
    "--mlp_init",
    help="Distribution to initialise the post-processing layer optimisable parameters",
    type=str,
    default=None,
)

args = parser.parse_args()


def main(args):
    """
    Main
    """
    # Set optimiser
    lr = args.learning_rate

    if args.optimiser == "Adadelta":
        opt = torch.optim.Adadelta(lr=lr)
    elif args.optimiser == "Adagrad":
        opt = torch.optim.Adagrad(lr=lr)
    elif args.optimiser == "Adam":
        opt = torch.optim.Adam(lr=lr)
    elif args.optimiser == "AdamW":
        opt = torch.optim.AdamW(lr=lr)
    elif args.optimiser == "Adamax":
        opt = torch.optim.Adamax(lr=lr)
    elif args.optimiser == "ASGD":
        opt = torch.optim.ASGD(lr=lr)
    elif args.optimiser == "NAdam":
        opt = torch.optim.NAdam(lr=lr)
    elif args.optimiser == "RAdam":
        opt = torch.optim.RAdam(lr=lr)
    elif args.optimiser == "RMSprop":
        opt = torch.optim.RMSprop(lr=lr)
    elif args.optimiser == "Rprop":
        opt = torch.optim.Rprop(lr=lr)
    elif args.optimiser == "SGD":
        opt = torch.optim.SGD(lr=lr)
    else:
        raise ValueError("Unrecognised torch optimiser given.")

    """
    # Set embedder
    embedder_arg = args.embedder.casefold()
    if embedder_arg == "bert":
        embedder = emb.Bert()
    elif embedder_arg == "fasttext":
        embedder = emb.FastText()
    elif embedder_arg == "ember":
        embedder = emb.Ember()
    else:
        raise ValueError("Unrecognised embedder type given.")
    """

    # Set dimensionality reduction techninique
    dimred_arg = args.dim_reduction.casefold()
    if dimred_arg == "pca":
        dim_red = dimred.PCA()
    elif dimred_arg == "ica":
        dim_red = dimred.ICA()
    elif dimred_arg == "tsvd":
        dim_red = dimred.TSVD()
    elif dimred_arg == "umap":
        dim_red = dimred.UMAP()
    elif dimred_arg == "tsne":
        dim_red = dimred.TSNE()
    else:
        raise ValueError("Unrecognised embedder was given.")

    # et quantum circuit
    if args.circuit == "Sim14":
        ansatz = circ.Sim14()
    elif args.circuit == "Sim15":
        ansatz = circ.Sim15()
    elif args.circuit == "StronglyEntangling":
        ansatz = circ.StronglyEntangling()
    else:
        raise ValueError("Unrecognised circuit type was given.")

    # Set the rescaling init functions for the circuit inputs,
    # circuit parameters, and MLP parameters

    param_init_functions = [("small", torch.nn.init.uniform_(a=0, b=2*np.pi)), ("unif", torch.nn.init.uniform_(a=0, b=2*np.pi)), ("norm", torch.nn.init.normal_(mean=np.pi, std=np.sqrt(np.pi)))]

    for param_init_function in param_init_functions
        if args.qubit_init[0] == param_init_function[0]:
            qubit_init = param_init_functions[1]
        elif args.qubit_init[0] == "const":
            qubit_init = torch.nn.init.constant_(val=args.qubit_init[1])
        else:
            raise ValueError("Unrecognised qubit init function.")
        
    for param_init_function in param_init_functions
        if args.data_rescaling[0] == param_init_function[0]:
            data_rescaling = param_init_functions[1]
        elif args.data_rescaling[0] == "const":
            data_rescaling = torch.nn.init.constant_(val=args.data_rescaling[1])
        else:
            raise ValueError("Unrecognised data rescaling function.")
        
    for param_init_function in param_init_functions:
        if args.mlp_init[0] == param_init_function[0]:
            mlp_init = param_init_functions[1]
        elif args.mlp_init[0] == "const":
            mlp_init = torch.nn.init.constant_(val=args.mlp_init[1])
        else:
            raise ValueError("Unrecognised MLP init function.")

    # Initialise objects (ths includes embedder, dim red, etc.)

    #Prepare data

    sentence_vectors = []
    labels = []
    data_paths = [args.train, args.val, args.test]
    for path in data_paths:
        if args.embedder not in path.split("_"):
            raise ValueError("The dataset provided does not correspond"
                             " to the choice of embedder.")
    train = pd.read_csv(args.train, delimiter='\t')
    val = pd.read_csv(args.val, delimiter="\t")
    test = pd.read_csv(args.test, delimiter="\t")
    for df in [train, val, test]:
        classes = df["class"].to_numpy()
        labels.append(classes)
        dim_red(dataset=df).reduce_dimension(dim_out=args.n_qubits)
        reduced_embeddings = dim_red.dataset["reduced_sentence_embedding"].to_numpy()
        sentence_vectors.append(reduced_embeddings)
    
    #Initialise model with all of its attributes
    if args.model == "alpha_3":
        circuit = ansatz(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            data_rescaling=data_rescaling,
            output_probabilities=False,
        )
        model = Alpha3(
            sentence_vectors=sentence_vectors,
            labels=labels,
            n_classes=args.n_classes,
            circuit = circuit,
            optimiser = opt,
            epochs = args.iterations,
            batch_size = args.batch_size,
            seed = args.seed,
            circuit_params_initialisation = qubit_init,
            mlp_params_initilaisation = mlp_init,
        )
    elif args.model == "alpha_4":
        circuit = ansatz(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            data_rescaling=data_rescaling,
            output_probabilities=True,
        )
        model = Alpha4(
            sentence_vectors=sentence_vectors,
            labels=labels,
            n_classes=args.n_classes,
            circuit = circuit,
            optimiser = opt,
            epochs = args.iterations,
            batch_size = args.batch_size,
            seed = args.seed,
            circuit_params_initialisation = qubit_init,
        )
    else:
        raise ValueError("An unrecognised model type was specified.")

    
    #Run pipeline
    model.train()
    model.compute_preds_prob_test()
    #Store variables resulting from train and test and store in JSON


    
