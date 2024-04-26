"""
use_alpha_3_4
=============

Implements pipeline for the experiments of the Alpha 3 and Alpha 4
models.
"""

import argparse
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch

import pennylane as qml

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/")
sys.path.append(current_path + "/../../models/quantum/alpha_3_4/")

import dim_reduction as dimred
import circuit as circ

from alpha_3_4 import Alpha3, Alpha3

parser = argparse.ArgumentParser()

# Model-related arguments

parser.add_argument(
    "-m",
    "--model",
    help="Choice between alpha_3 and alpha_4 model.",
    type=str,
    choices=["alpha_3", "alpha_4"],
    default="alpha_3",
)
parser.add_argument(
    "-dat",
    "--dataset",
    help="Name of dataset.",
    type=str,
    choices=["ag_news", "newscatcher", "reviews", "bert_test"],
)
parser.add_argument(
    "-sp",
    "--split_id",
    help="Split ID - indicates what split will be used as validation data."
    "The split with split ID + 1 will be used as test data.",
    type=int,
    default=0,
)
parser.add_argument(
    "-s",
    "--seed",
    help="Seed for the initial parameters",
    type=int,
    default=150298,
)
parser.add_argument(
    "-e",
    "--epochs",
    help="Number of iterations/epochs",
    type=int,
    default=150,
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
    "-em",
    "--embedder",
    help="Choice of embedder",
    type=str,
    default="Bert",
)
# In current pipeline we pass an already vectorised dataset
# as argument, so this argument is for book-keeping

parser.add_argument(
    "-dr",
    "--dim_reduction",
    help="Choice of dimensionality reduction mechanism",
    type=str,
    choices=["PCA", "ICA", "TSVD", "UMAP"],
    default="PCA",
)

# Arguments for quantum circuit architecture

parser.add_argument(
    "-c",
    "--circuit",
    help="Choice of quantum circuit architecture",
    type=str,
    choices=["Sim14", "Sim15", "StronglyEntanglingAnsatz"],
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
    "-res",
    "--data_rescaling",
    help="Function to apply to rescale the inputs that will be encoded \
        in the first wall of the quantum circuit.",
    type=str,
    default="none",
    choices=["none", "rescaled_unif", "unif", "norm"],
)
parser.add_argument(
    "-ci",
    "--circuit_init",
    help="Function to be used to initialise circuit optimisable parameters.",
    type=str,
    default=None,
    choices=["none", "rescaled_unif", "unif", "norm"],
)

# Arguments for MLP layer (Alpha 3 only)

parser.add_argument(
    "-mi",
    "--mlp_init",
    help="Distribution to initialise the post-processing layer optimisable parameters.",
    type=str,
    default=None,
    choices=["none", "rescaled_unif", "unif", "norm"],
)

args = parser.parse_args()


def main(args) -> dict:
    """
    Main

    Parameters
    ----------

    Returns
    -------
    """

    # Set optimiser
    lr = args.learning_rate
    optimiser_args = {"lr": lr}

    if args.optimiser == "Adadelta":
        opt = torch.optim.Adadelta
    elif args.optimiser == "Adagrad":
        opt = torch.optim.Adagrad
    elif args.optimiser == "Adam":
        opt = torch.optim.Adam
    elif args.optimiser == "AdamW":
        opt = torch.optim.AdamW
    elif args.optimiser == "Adamax":
        opt = torch.optim.Adamax
    elif args.optimiser == "ASGD":
        opt = torch.optim.ASGD
    elif args.optimiser == "NAdam":
        opt = torch.optim.NAdam
    elif args.optimiser == "RAdam":
        opt = torch.optim.RAdam
    elif args.optimiser == "RMSprop":
        opt = torch.optim.RMSprop
    elif args.optimiser == "Rprop":
        opt = torch.optim.Rprop
    elif args.optimiser == "SGD":
        opt = torch.optim.SGD
    else:
        raise ValueError("Unrecognised torch optimiser given.")

    # Set dimensionality reduction techninique
    dimred_arg = args.dim_reduction
    dim_out = args.n_qubits
    if dimred_arg == "PCA":
        dim_red = dimred.PCA(dim_out)
    elif dimred_arg == "ICA":
        dim_red = dimred.ICA(dim_out)
    elif dimred_arg == "TSVD":
        dim_red = dimred.TSVD(dim_out)
    elif dimred_arg == "UMAP":
        dim_red = dimred.UMAP(dim_out)
    else:
        raise ValueError(
            "Unrecognised dimensionality reduction technique was given."
        )

    # Set quantum circuit
    if args.circuit == "Sim14":
        ansatz = circ.Sim14
    elif args.circuit == "Sim15":
        ansatz = circ.Sim15
    elif args.circuit == "StronglyEntangling":
        ansatz = circ.StronglyEntangling
    else:
        raise ValueError("Unrecognised circuit type was given.")

    # Set the rescaling init functions for the circuit inputs,
    # circuit parameters, and MLP parameters

    param_init_functions = [
        ("rescaled_unif", torch.nn.init.uniform_),  # (a=0, b=2 * np.pi)
        ("unif", torch.nn.init.uniform_),  # (a=0, b=2 * np.pi)
        ("norm", torch.nn.init.normal_),  # (mean=np.pi, std=np.sqrt(np.pi))
        ("none", None),
    ]

    error = True
    for param_init_function in param_init_functions:
        if args.circuit_init == param_init_function[0]:
            circuit_init = param_init_function[1]
            error = False
            break
    if error:
        raise ValueError("Unrecognised qubit init function.")

    error = True
    for param_init_function in param_init_functions:
        if args.data_rescaling == param_init_function[0]:
            data_rescaling = param_init_function[1]
            error = False
            break
    if error:
        raise ValueError("Unrecognised data rescaling function.")

    error = True
    for param_init_function in param_init_functions:
        if args.mlp_init == param_init_function[0]:
            mlp_init = param_init_function[1]
            error = False
            break
    if error:
        raise ValueError("Unrecognised MLP init function.")

    # Prepare data

    # Set dataset
    if args.dataset == "ag_news":
        n_classes = 4
        pass
    elif args.dataset == "newscatcher":
        n_classes = 7
        pass
    elif args.dataset == "reviews":
        n_classes = 3
        pass
    elif args.dataset == "bert_test":
        n_classes = 2
        dataset_path = "../datasets/bert_test_pipeline_dataset.pkl"

    sentence_vectors = []
    labels = []
    split_id = args.split_id

    if args.embedder.lower() not in dataset_path:
        raise ValueError(
            "The dataset provided does not correspond"
            " to the choice of embedder."
        )

    dataset = pd.read_pickle(dataset_path)
    dim_red.fit(data_to_fit=dataset)

    # Split train and val
    val = dataset[dataset["split"] == split_id]
    train = dataset[dataset["split"] != split_id]

    for df in [train, val]:
        classes = df["class"].astype(int).tolist()
        print(classes)
        labels.append(classes)
        reduced_dataset = dim_red.reduce_dimension(data_to_reduce=df)
        reduced_embeddings = reduced_dataset[
            "reduced_sentence_vector"
        ].tolist()
        sentence_vectors.append(reduced_embeddings)

    # Initialise model with all of its attributes
    if args.model == "alpha_3":
        circuit = ansatz(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            axis_embedding="X",
            observables={key: qml.PauliY for key in range(args.n_qubits)},
            data_rescaling=data_rescaling,
            output_probabilities=False,
        )
        model = Alpha3(
            sentence_vectors=sentence_vectors,
            labels=labels,
            n_classes=n_classes,
            circuit=circuit,
            optimiser=opt,
            optimiser_args=optimiser_args,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            circuit_params_initialisation=circuit_init,
            mlp_params_initialisation=mlp_init,
        )
    elif args.model == "alpha_4":
        circuit = ansatz(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            # Need to add observables and axis emb
            data_rescaling=data_rescaling,
            output_probabilities=True,
        )
        model = Alpha4(
            sentence_vectors=sentence_vectors,
            labels=labels,
            n_classes=n_classes,
            circuit=circuit,
            optimiser=opt,
            optimiser_args=optimiser_args,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            circuit_params_initialisation=circuit_init,
        )
    else:
        raise ValueError("An unrecognised model type was specified.")

    # Train model and store model outputs
    """
    print(
        f"----------\n"
        "Beginning training for the set of parameters with ID: {args.exp_id}\n"
        "----------\n"
    )"""
    t_before = time.time()
    model.train()
    t_after = time.time()
    # print("\n----- Training completed! -----\n")

    train_loss_list = model.loss_train
    train_preds_list = model.preds_train
    train_probs_list = model.probs_train
    val_loss_list = model.loss_val
    val_preds_list = model.preds_val
    val_probs_list = model.probs_val

    time_taken = t_after - t_before
    # print(f"Time taken to run = {time_taken}\n")

    results_dict = {
        "split_id": args.split_id,
        "seed": args.seed,
        "dataset": args.dataset.split("/")[-1],
        "epochs": args.epochs,
        "optimiser": args.optimiser,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "embedder": args.embedder,
        "dim_reduction": args.dim_reduction,
        "ansatz": args.circuit,
        "num_qubits": args.n_qubits,
        "num_layers": args.n_layers,
        "data_rescaling": args.data_rescaling,
        "circuit_params_init": args.circuit_init,
        "mlp_params_init": args.mlp_init,
        "train_labels": labels[0],
        "train_loss": train_loss_list,
        "train_preds": train_preds_list,
        "train_probs": train_probs_list,
        "val_labels": labels[1],
        "val_loss": val_loss_list,
        "val_preds": val_preds_list,
        "val_probs": val_probs_list,
        "train_runtime": time_taken,
    }

    # Return model outputs

    return results_dict


if __name__ == "__main__":
    main(args)


# Finish documenting

# Test with ag_news dataset, test outputs look good
