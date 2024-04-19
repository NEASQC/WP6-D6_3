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

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/")
sys.path.append(current_path + "/../../models/quantum/alpha_3_4/")

import dim_reduction as dimred
import circuit as circ

from alpha_3_4 import Alpha3, Alpha4

parser = argparse.ArgumentParser()

parser.add_argument(
    "-id",
    "--exp_id",
    help="Experiment ID for the hyperparameter configuration.",
    type=int,
)

# Model-related arguments

parser.add_argument(
    "-m",
    "--model",
    help="Choice between alpha_3 and alpha_4 model.",
    type=str,
    choices=['alpha_3', 'alpha_4']
    default="alpha_3",
)
parser.add_argument(
    "-dat",
    "--dataset",
    help="Path to the full dataset.",
    type=str,
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
    choices=['PCA', 'ICA', 'TSVD', 'UMAP'],
    default="PCA",
)

# Arguments for quantum circuit architecture

parser.add_argument(
    "-c",
    "--circuit",
    help="Choice of quantum circuit architecture",
    type=str,
    choices=['Sim14', 'Sim15', 'StronglyEntanglingAnsatz'],
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
    default=None,
    choices=['None', 'rescaled_unif', 'unif', 'norm'],
)
parser.add_argument(
    "-ci",
    "--circuit_init",
    help="Function to be used to initialise circuit optimisable parameters.",  
    type=str,
    default=None,
    choices=['None', 'rescaled_unif', 'unif', 'norm'],
)

# Arguments for MLP layer (Alpha 3 only)

parser.add_argument(
    "-mi",
    "--mlp_init",
    help="Distribution to initialise the post-processing layer optimisable parameters.",
    type=str,
    default=None,
    choices=['None', 'rescaled_unif', 'unif', 'norm'],
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
        raise ValueError("Unrecognised dimensionality reduction technique was given.")

    # Set quantum circuit
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

    param_init_functions = [("rescaled_unif", torch.nn.init.uniform_(a=0, b=2*np.pi)), ("unif", torch.nn.init.uniform_(a=0, b=2*np.pi)), ("norm", torch.nn.init.normal_(mean=np.pi, std=np.sqrt(np.pi)))]

    for param_init_function in param_init_functions:
        if args.circuit_init[0] == param_init_function[0]:
            circuit_init = param_init_functions[1]
        else:
            raise ValueError("Unrecognised qubit init function.")
        
    for param_init_function in param_init_functions:
        if args.data_rescaling[0] == param_init_function[0]:
            data_rescaling = param_init_functions[1]
        else:
            raise ValueError("Unrecognised data rescaling function.")
        
    for param_init_function in param_init_functions:
        if args.mlp_init[0] == param_init_function[0]:
            mlp_init = param_init_functions[1]
        else:
            raise ValueError("Unrecognised MLP init function.")

    # Prepare data

    sentence_vectors = []
    labels = []
    dataset_path = args.dataset
    split_id = args.split_id

    split_data_path = dataset_path.split('_')
    split_data_path = [item.split('.') for item in split_data_path]

    if args.embedder not in split_data_path:
            raise ValueError("The dataset provided does not correspond"
                             " to the choice of embedder.")

    dataset = pd.read_pickle(dataset_path)
    dim_red.fit(data_to_fit=dataset)

    # Split train and val
    val = dataset[dataset['split'] == split_id]
    train = dataset[dataset['split'] != split_id]

    for df in [train, val]:
        classes = df["class"].astype(int).tolist()
        labels.append(classes)
        reduced_dataset = dim_red.reduce_dimension(data_to_reduce=df)
        reduced_embeddings = reduced_dataset["reduced_sentence_vector"].tolist()
        sentence_vectors.append(reduced_embeddings)
    
    # Initialise model with all of its attributes
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
            circuit_params_initialisation = circuit_init,
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
            circuit_params_initialisation = circuit_init,
        )
    else:
        raise ValueError("An unrecognised model type was specified.")

    # Train model and store model outputs
    
    model.seed = args.seed

    print(f'----------\n /
           Beginning training for the set of parameters with ID: {args.exp_id}\n /
           ----------\n')
    t_before = time.time()
    model.train()
    t_after = time.time()
    print('\n----- Training completed! -----\n')

    train_loss_list = model.loss_train
    train_preds_list = model.preds_train
    train_probs_list = model.probs_train
    val_loss_list =  model.loss_val
    val_preds_list = model.preds_val
    val_probs_list = model.probs_val

    time_taken = t_after - t_before
    print(f"Time taken to run ID {args.exp_id} = {time_taken}\n")

    results_dict = {
    'Params_ID': args.exp_id,
    'Split': args.split,
    'Seed': args.seed,
    'Dataset': args.dataset.split('/')[-1],
    'Epochs': args.epochs,
    'Optimiser': args.optimiser,
    'Batch Size': args.batch_size,
    'Learning Rate': args.learning_rate,
    'Embedder': args.embedder,
    'Dim Reduction': args.dim_reduction,
    'Ansatz': args.circuit,
    'Num Qubits': args.n_qubits,
    'Num Layers': args.n_layers,
    'Data Rescaling': args.data_rescaling,
    'Circuit Params Init': args.circuit_init,
    'MLP Params Init': args.mlp_init,
    'Train Loss': train_loss_list,
    'Train Preds': train_preds_list,
    'Train Probs': train_probs_list,
    'Val Loss': val_loss_list,
    'Val Preds': val_preds_list,
    'Val Probs': val_probs_list,
    'Runtime': time_taken
}
    
    # Return model outputs
        
    return results_dict


if __name__ == "__main__":
    main()

    
