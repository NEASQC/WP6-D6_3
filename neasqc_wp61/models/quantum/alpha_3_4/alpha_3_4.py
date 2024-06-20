"""
Alpha3_4
======
Module containing the class for Alpha3 model.
"""

import os
import sys
from typing import Callable

import numpy as np
import pennylane as qml
import torch

from abc import ABC, abstractmethod
from numpy.core.multiarray import array as array
from torch.utils.data import DataLoader

# The two lines below will be removed when converting the library to a package.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../")
from circuit import Circuit
from utils import Dataset


class NewAlphaModel(ABC, torch.nn.Module):
    """
    An abstract class to implement the new Alpha models, Alpha 3 and
    Alpha 4, which are structurally very similar. New models that
    exploit this structure can be developed as children classes of this
    class.
    """

    def __init__(
        self,
        sentence_vectors: list[list[np.array]],
        labels: list[list[int]],
        n_classes: int,
        circuit: Circuit,
        optimiser: torch.optim.Optimizer,
        epochs: int,
        batch_size: int,
        loss_function: Callable = torch.nn.CrossEntropyLoss,
        optimiser_args: dict = {},
        device: str = "cpu",
        seed: int = 1906,
    ) -> None:
        """
        Initialise the alpha3 class.

        Parameters
        ----------
        sentence_vectors : list[list[np.array]]
            List with the train, validation and test sentence vectors.
        labels : list[list[int]]
            List with the train, validation and test labels.
        n_classes : int
            Total number of classes in our datasets.
        circuit : Circuit
            Circuit to use in the neural network. Its number
            of qubits must be equal to the length of the
            vectors. Also the argument output_probabilities
            of the circuit class
            must be set to False for this model.
        optimiser : torch.optim.Optimizer
            Optimiser to use for training the model.
        epochs : int
            Number of epochs to be done in training.
        batch_size : int
            Batch size to use in training.
        loss_function : Callable
            Loss function to use in training.
            (Default = torch.nn.CrossEntropyLoss)
        optimiser_args : dict
            Optional arguments for the optmiser. (Default = {}).
        device : str
            CUDA device ID used for tensor operation speed-up.
            (Default = "cpu').
        seed : int
            Random seed for paramter initialisation.
            (Default = 1906).
        circuit_params_initialisation : torch.nn.init
            Function to be used to initialise circuit
            optimisable parameters.
        """
        torch.manual_seed(seed)
        super().__init__()
        self.data_loader_train = DataLoader(
            Dataset(sentence_vectors[0], labels[0]), batch_size=batch_size
        )
        self.data_loader_val = DataLoader(
            Dataset(sentence_vectors[1], labels[1]), batch_size=batch_size
        )
        """
        self.data_loader_test = DataLoader(
            Dataset(sentence_vectors[2], labels[2]), batch_size=batch_size
        )
        """
        self.n_classes = n_classes
        self.circuit = circuit
        self.optimiser = optimiser
        self.epochs = epochs
        self.loss_function = loss_function()
        self.optimiser_args = optimiser_args
        self.device = device
        vectors_match_n_qubits = all(
            len(vector_instance) == self.circuit.n_qubits
            for vector_partition in sentence_vectors
            for vector_instance in vector_partition
        )
        if not vectors_match_n_qubits:
            raise ValueError(
                "All vector lengths must be equal to the number of qubits"
                " in the circuit."
            )

    @abstractmethod
    def forward(self, sentence_tensors: torch.tensor) -> torch.tensor:
        """
        Compute the output of the model for a tensor containing a number
        of sentence vectors equal to the batch size.

        Parameters
        ----------
        sentence_tensors : torch.tensor
            Sentence tensors to be input to the model.

        Returns
        -------
        torch.tensor
            The output of the model.
        """
        raise NotImplementedError(
            "Subclasses must implement the forward method."
        )

    @abstractmethod
    def compute_probs(self, sentence_tensors: torch.tensor) -> torch.tensor:
        """
        Compute the output probabilities of the model for a tensor
        containing a number of sentence vectors equal to the batch size.

        Parameters
        ----------
        sentence_tensors : torch.tensor
            Sentence tensors to be input to the model.

        Returns
        -------
        torch.tensor
            The output probabilities of the model.
        """
        raise NotImplementedError(
            "Subclasses must implement the compute_probs method."
        )

    def compute_preds(self, sentence_tensors: torch.tensor) -> torch.tensor:
        """
        Compute the output predictions of the model for a tensor
        containing a number of sentence vectors equal to the batch size.

        Parameters
        ----------
        sentence_tensors : torch.tensor
            Sentence tensors to be input to the model.

        Returns
        -------
        preds: torch.tensor
            The model's predictions for the given input tensor.
        """
        model_output = self.forward(sentence_tensors)
        _, preds = torch.max(model_output, 1)
        return preds

    def train(self) -> None:
        """
        Trains the model.
        """
        loss_train_val = [[], []]
        preds_train_val = [[], []]
        probs_train_val = [[], []]
        data_loaders = [self.data_loader_train, self.data_loader_val]
        opt = self.optimiser(self.parameters(), **self.optimiser_args)
        dataset_size = [
            len(self.data_loader_train.dataset),
            len(self.data_loader_val.dataset),
        ]
        for _ in range(self.epochs):
            loss_epoch_train_val = [0, 0]
            preds_epoch_train_val = [[], []]
            probs_epoch_train_val = [[], []]
            for i, data_loader in enumerate(data_loaders):
                for vector, label in data_loader:
                    vector = vector.to(self.device)
                    label = label.to(self.device)
                    opt.zero_grad()
                    batch_loss = self.loss_function(
                        self.forward(vector), label.long()
                    )
                    loss_epoch_train_val[i] += (
                        batch_loss.item() * vector.shape[0]
                    )
                    for j in range(vector.shape[0]):
                        preds_epoch_train_val[i].append(
                            self.compute_preds(vector)[j].item()
                        )
                        probs_epoch_train_val[i].append(
                            self.compute_probs(vector)[j, :].tolist()
                        )
                    if i == 0:
                        batch_loss.backward()
                        opt.step()
                loss_train_val[i].append(
                    loss_epoch_train_val[i] / dataset_size[i]
                )
                preds_train_val[i].append(preds_epoch_train_val[i])
                probs_train_val[i].append(probs_epoch_train_val[i])
        self.loss_train = loss_train_val[0]
        self.loss_val = loss_train_val[1]
        self.preds_train = preds_train_val[0]
        self.preds_val = preds_train_val[1]
        self.probs_train = probs_train_val[0]
        self.probs_val = probs_train_val[1]

    def compute_preds_probs_test(self) -> None:
        """
        Once the model is trained, compute the predictions and
        probabilities for each epoch for the test dataset.
        """
        self.preds_test = []
        self.probs_test = []
        with torch.no_grad():
            for vector, label in self.data_loader_test:
                label = label.to(self.device)
                vector = vector.to(self.device)
                for i in range(vector.shape[0]):
                    self.preds_test.append(
                        self.compute_preds(vector)[i].item()
                    )
                    self.probs_test.append(
                        self.compute_probs(vector)[i, :].tolist()
                    )


class Alpha3(NewAlphaModel):
    """
    A class for the Alpha3 model, a classifier using a circuit with a
    post-processing linear layer.
    """

    def __init__(
        self,
        sentence_vectors: list[list],
        labels: list[list[int]],
        n_classes: int,
        circuit: Circuit,
        optimiser: torch.optim.Optimizer,
        epochs: int,
        batch_size: int,
        loss_function: Callable = torch.nn.CrossEntropyLoss,
        optimiser_args: dict = {},
        device: str = "cpu",
        seed: int = 1906,
        circuit_params_initialisation: torch.nn.init = None,
        mlp_params_initialisation: torch.nn.init = None,
    ) -> None:
        """
        Initialises the Alpha3 class with the attributes specified in
        the abstrac class, as well as:

        circuit_params_initialisation : torch.nn.init
            Function to be used to initialise circuit optimisable
            parameters (default: None).
        mlp_params_initialisation : torch.nn.init
            Function to be used to initialise post-processing layer
            optimisable parameters, i.e. weights and biases (default:
            None).
        """
        super().__init__(
            sentence_vectors,
            labels,
            n_classes,
            circuit,
            optimiser,
            epochs,
            batch_size,
            loss_function,
            optimiser_args,
            device,
            seed,
        )
        self.n_outputs_quantum_circuit = len(self.circuit.observables.keys())
        if self.circuit.output_probabilities == True:
            raise ValueError(
                "The circuit must have output_probabilities = False"
                " for Alpha3 model."
            )

        quantum_node = qml.QNode(
            self.circuit.build_circuit,
            device=self.circuit.device,
            interface="torch",
        )
        weight_shapes = {"params": self.circuit.parameters_shape}

        self.quantum_circuit_layer = qml.qnn.TorchLayer(
            quantum_node, weight_shapes, circuit_params_initialisation
        )
        self.post_layer = torch.nn.Linear(
            self.n_outputs_quantum_circuit, self.n_classes
        )
        if mlp_params_initialisation != None:
            mlp_params_initialisation(self.post_layer.weight)
        self.softmax_layer = torch.nn.Softmax()

    def forward(self, sentence_tensors: torch.tensor) -> torch.tensor:
        """
        Compute the output of the model for a tensor containing a number
        of sentence vectors equal to the batch size.

        Parameters
        ----------
        sentence_tensors : torch.tensor
            Sentence tensors to be input to the model.

        Returns
        -------
        post_layer_output: torch.tensor
            The output of the model for the given input tensor.
        """
        circuit_output = self.quantum_circuit_layer(sentence_tensors).float()
        post_layer_output = self.post_layer(circuit_output)
        return post_layer_output

    def compute_probs(self, sentence_tensors: torch.tensor) -> torch.tensor:
        """
        Compute the output probabilities of the model for a tensor
        containing a number of sentence vectors equal to the batch size.

        Parameters
        ----------
        sentence_tensors : torch.tensor
            Sentence tensors to be input to the model.

        Returns
        -------
        model_probabilities: torch.tensor
            The probabilities of the model for the given input tensor.
        """
        softmax = torch.nn.Softmax(dim=1)
        model_output = self.forward(sentence_tensors)
        model_probabilities = softmax(model_output)
        return model_probabilities


class Alpha4(NewAlphaModel):
    """
    A class for the Alpha4 model, a classifier based on the output
    probabilities of a quantum circuit.
    """

    def __init__(
        self,
        sentence_vectors: list[list],
        labels: list[list[int]],
        n_classes: int,
        circuit: Circuit,
        optimiser: torch.optim.Optimizer,
        epochs: int,
        batch_size: int,
        loss_function: Callable = torch.nn.CrossEntropyLoss,
        optimiser_args: dict = {},
        device: str = "cpu",
        seed: int = 1906,
        circuit_params_initialisation: torch.nn.init = None,
    ) -> None:
        """
        Initialises the Alpha4 class with the attributes specified in
        the abstrac class, as well as:

        circuit_params_initialisation : torch.nn.init
            Function to be used to initialise circuit optimisable
            parameters (default: None).
        """
        super().__init__(
            sentence_vectors,
            labels,
            n_classes,
            circuit,
            optimiser,
            epochs,
            batch_size,
            loss_function,
            optimiser_args,
            device,
            seed,
            circuit_params_initialisation,
        )
        if self.circuit.output_probabilities == False:
            raise ValueError(
                "The circuit must have output_probabilities = True"
                " for Alpha4 model."
            )
        if 2 ** len(self.circuit.observables.keys()) <= self.n_classes:
            raise ValueError(
                "The base 2 log of the number of qubits measured must"
                " be greater or equal to the number of classes in our dataset."
            )
        quantum_node = qml.QNode(
            self.circuit.build_circuit,
            device=self.circuit.device,
            interface="torch",
        )
        weight_shapes = {"params": self.circuit.parameters_shape}
        self.quantum_circuit_layer = qml.qnn.TorchLayer(
            quantum_node, weight_shapes, circuit_params_initialisation
        )

    def forward(self, sentence_tensors: torch.tensor) -> torch.tensor:
        """
        Compute the output of the network for a a tensor containing a
        number of sentence vectors equal to the batch size.

        Our circuits will output a probability vector of size 2 **
        n_measured_qubits. If this size is equal to the number of
        classes, we will take these vectors as outputs. If our number of
        classes is different from 2 ** n_measured_qubits, (which again
        is the size of the output probabilities tensor of our circuit),
        then we only take values up to n_classes.

        Parameters
        ----------
        sentence_tensors : torch.tensor
            Sentence tensors to be input to the model.

        Returns
        -------
        circuit_output_trimmed: torch.tensor
            The output of the model, which is trimmed to ensure its
            dimension is equal to the number of classes.
        """
        circuit_output = self.quantum_circuit_layer(sentence_tensors).float()
        circuit_output_trimmed = circuit_output[:, : self.n_classes]
        return circuit_output_trimmed

    def compute_probs(self, sentence_tensors: torch.tensor) -> torch.tensor:
        """
        Compute the output probabilities of the model for a tensor
        containing a number of sentence vectors equal to the batch size.
        If the n_classes it's equal to 2 ** (number of measured qubits)
        then the model output can be taken, as the output represents
        probability values which are already normalised. When n_classes
        it's not equal to the aforementioned quantity, then a softmax
        layer needs to be applied to normalise the outputs of the model
        (as we are just taking probabilities output from the circuit up
        to the value self.n_classes).

        Parameters
        ----------
        sentence_tensors : torch.tensor
            Sentence tensors to be input to the model.

        Returns
        -------
        model_output : torch.tensor
            The (normalised) probabilities of the model for the given
            input tensor.
        """
        softmax = torch.nn.Softmax(dim=1)
        output_probs = self.forward(sentence_tensors)
        num_measured_qubits = len(self.circuit.observables.keys())
        model_output = output_probs
        if self.n_classes != 2**num_measured_qubits:
            model_output = softmax(output_probs)
        return model_output
