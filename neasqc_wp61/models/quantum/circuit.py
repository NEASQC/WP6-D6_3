"""
Circuit
=======
Module containing the base class for the variational circuits of Alpha3 model.

"""
from abc import ABC, abstractmethod
from typing import Callable, Dict

import pennylane as qml
import torch


class Circuit(ABC):
    """
    Base class for circuits.
    """
    def __init__(
        self, n_qubits : int,  n_layers : int,
        axis_embedding : str, observables : Dict[int, qml.operation.Operator],
        device_name : str = "default.qubit",
        output_probabilities : bool = True,
        data_rescaling : Callable = None, **kwargs
    )-> None:
        """
        Initialise the circuit class.

        Parameters 
        ----------
        n_qubits : int
            Number of qubits of the circuit.
        n_layers : int
            Number of times the ansatz is applied to the circuit.
        axis_embedding : str 
            Rotation gate to use for the angle encoding of the inputs.
            Must be one of ['X','Y','Z'].
        observables : Dict[int, qml.operation.Operator]
            Dictionary with pennylane quantum operators. The keys of
            of the dictionary indicate on which qubit the operators act.
        device_name : str, optional
            Pennylane simulator to use. The available devices can be found in
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html.
            Optional parameter. The default value is default.qubit.
        output_probabilities : bool, optional
            If True, the circuit will output a list of probabilities, 
            where each of the items are the probabilities of measuring 
            the vectors of the basis generated by the operators acting on
            each qubit. If False, it will output the expected values of the 
            operators acting on each qubit. The default value is True.
        data_rescaling : Callable, optional
            Function to apply to rescale the inputs that will be encoded.
            The default value is None.
        ** kwargs 
            Keyword arguments to be introduced in the pennylane device.
            More info can be found in 
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html.
        
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.axis_embedding = axis_embedding
        self.observables = observables
        if output_probabilities == True:
            measured_qubit_list = list(self.observables.keys())
            operators_list = list(self.observables.values())
            for i, (qubit, operator) in enumerate(zip(
                measured_qubit_list,
                operators_list
            )):
                if i == 0:
                    self.operators_tensor_product = operator(qubit)
                else:
                    self.operators_tensor_product @= operator(qubit)
        self.device = qml.device(
            device_name, wires=self.n_qubits, **kwargs)
        self.output_probabilities = output_probabilities
        self.data_rescaling = data_rescaling
        if self.n_qubits < 2:
            raise ValueError(
                'The number of qubits must be greater or equal 2.'
            )
        if not all(x < self.n_qubits for x in self.observables.keys()): 
            raise ValueError('Qubit index out of range.')

    def rescale_circuit_inputs(self, input)-> None:
        """
        Apply the data rescaling function to the inputs that will be embbeded.

        Parameters
        ----------
        input : torch.tensor
            Input features introduced. They will be encoded in the circuit 
            using angle encoding techniques.
        """
        if self.data_rescaling != None:
            return self.data_rescaling(input)
        else:
            return input
        
    def embedd_inputs_in_circuit(self, input):
        """
        Embedd input into the quantum circuit using angle embedding.

        Parameters
        ----------
        input : torch.tensor
            Input features introduced. They will be encoded in the circuit 
            using angle encoding techniques.
        """
        return (
            qml.AngleEmbedding( 
            features=input, wires = range(self.n_qubits),
            rotation = self.axis_embedding
            )
        )

    @abstractmethod
    def build_circuit(
        self, input :torch.tensor, params : torch.tensor
    )-> Callable:
        """
        Build the circuit function that can be run using pennylane simulators.

        Parameters
        ----------
        input : torch.tensor
            Input features introduced. They will be encoded in the circuit 
            using angle encoding techniques.
        params : torch.tensor
            Variational paramaters of the ansatz. They will  be optimised
            within the model.

        Returns
        -------
        Callable
            Function represesting the quantum circuit.
        """

    def run_and_measure_circuit(
        self, circuit_function : Callable
    )-> qml.QNode:
        """
        Build a quantum node containing a circuit function (meant to be output
        of build_circuit_function) and device to
        be run on. More info can be found in 
        https://docs.pennylane.ai/en/stable/code/api/pennylane.qnode.html.

        Parameters
        ----------
        circuit_function : Callable
            Circuit function.
        
        Returns
        -------
        qnode : qml.QNode
            Pennylane Quantum node. When called, it outputs the results of 
            the measurements in our circuit function.
        """
        qnode = qml.QNode(circuit_function, self.device, interface = "torch")
        return qnode
    

class Sim14(Circuit):
    """
    Class containing ansatz 14 of
    https://arxiv.org/pdf/1905.10876.pdf
    with the controlled rotation gates placed in opposite
    orientation. Given a set of (control, target) indexes for
    the controlled rotations gates: 
    {(3,0), (2,3) , (1,2), (0,1)}
    our class will implement controlled rotation gates with indexes:
    {(0,3), (1,0), (2,1), 3,2)}.
    """
    def __init__(
        self, n_qubits : int,  n_layers : int,
        axis_embedding : str, observables : Dict[int, qml.operation.Operator],
        device_name : str = "default.qubit",
        output_probabilities : bool = True,
        data_rescaling : Callable = None, **kwargs
    )-> None:
        """
        Initialise the Sim14 class.

        Parameters 
        ----------
        n_qubits : int
            Number of qubits of the circuit.
        n_layers : int
            Number of times the ansatz is applied to the circuit.
        axis_embedding : str 
            Rotation gate to use for the angle encoding of the inputs.
            Must be one of ['X','Y','Z'].
        observables : Dict[int, qml.operation.Operator]
            Dictionary with pennylane quantum operators. The keys of
            of the dictionary indicate on which qubit the operators act.
        device_name : str, optional
            Pennylane simulator to use. The available devices can be found in
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html.
            Optional parameter. The default value is default.qubit.
        output_probabilities : bool, optional
            If True, the circuit will output a list of probabilities, 
            where each of the items are the probabilities of measuring 
            the vectors of the basis generated by the operators acting on
            each qubit. If False, it will output the expected values of the 
            operators acting on each qubit. The default value is True.
        data_rescaling : Callable, optional
            Function to apply to rescale the inputs that will be encoded.
            The default value is None.
        ** kwargs 
            Keyword arguments to be introduced in the pennylane device.
            More info can be found in 
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html.
        
        """
        super().__init__(
            n_qubits, n_layers, axis_embedding, observables,
            device_name, output_probabilities, data_rescaling, **kwargs
        )
        self.parameters_shape = (n_layers, 4  * n_qubits)

    def build_circuit(
        self, inputs : torch.Tensor,
        params : torch.Tensor
    )-> Callable:
        """
        Build the circuit function for Sim14 class.

        Parameters
        ----------
        input : torch.tensor
            Input features introduced. They will be encoded in the circuit 
            using angle encoding techniques.
        params : torch.tensor
            Variational paramaters of the ansatz. They will be optimised
            within the model.

        Returns
        -------
        Callable
            Function represesting the quantum circuit.
        """
        rescaled_circuit_inputs = self.rescale_circuit_inputs(inputs)
        self.embedd_inputs_in_circuit(rescaled_circuit_inputs)
        for i in range(self.n_layers):
            idx = 0
            for j in range(self.n_qubits):
                qml.RY(params[i, j + idx], wires = j)
            idx += self.n_qubits
            for j in range(self.n_qubits):
                ctrl = j
                target = (j - 1) % self.n_qubits
                qml.CRX(phi=params[i, j + idx], wires=[ctrl,target])
            idx += self.n_qubits
            for j in range(self.n_qubits):
                qml.RY(params[i, j + idx], wires = j)
            idx += self.n_qubits
            for j in range(self.n_qubits, 0, -1):
                ctrl = j % self.n_qubits
                target = (j + 1) % self.n_qubits
                qml.CRX(params[i, j + idx -1], wires= [ctrl, target])

        if self.output_probabilities:
            return [qml.probs(op = self.operators_tensor_product)]
        else:
            return [qml.expval(
                self.observables[k](k)) for k in self.observables.keys()
            ]
        
class Sim15(Circuit):
    """
    Class containing ansatz 15 of
    https://arxiv.org/pdf/1905.10876.pdf
    with the controlled rotation gates placed in opposite
    orientation. Given a set of (control, target) indexes for
    the controlled rotations gates: 
    {(3,0), (2,3) , (1,2), (0,1)}
    our class will implement controlled rotation gates with indexes:
    {(0,3), (1,0), (2,1), 3,2)}.
    """
    def __init__(
        self, n_qubits : int,  n_layers : int,
        axis_embedding : str, observables : Dict[int, qml.operation.Operator],
        device_name : str = "default.qubit",
        output_probabilities : bool = True,
        data_rescaling : Callable = None, **kwargs
    )-> None:
        """`
        Initialise the Sim15 class.

        Parameters 
        ----------
        n_qubits : int
            Number of qubits of the circuit.
        n_layers : int
            Number of times the ansatz is applied to the circuit.
        axis_embedding : str 
            Rotation gate to use for the angle encoding of the inputs.
            Must be one of ['X','Y','Z'].
        observables : Dict[int, qml.operation.Operator]
            Dictionary with pennylane quantum operators. The keys of
            of the dictionary indicate on which qubit the operators act.
        device_name : str, optional
            Pennylane simulator to use. The available devices can be found in
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html.
            Optional parameter. The default value is default.qubit.
        output_probabilities : bool, optional
            If True, the circuit will output a list of probabilities, 
            where each of the items are the probabilities of measuring 
            the vectors of the basis generated by the operators acting on
            each qubit. If False, it will output the expected values of the 
            operators acting on each qubit. The default value is True.
        data_rescaling : Callable, optional
            Function to apply to rescale the inputs that will be encoded.
            The default value is None.
        ** kwargs 
            Keyword arguments to be introduced in the pennylane device.
            More info can be found in 
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html.
        
        """
        super().__init__(
            n_qubits, n_layers, axis_embedding, observables,
            device_name, output_probabilities, data_rescaling, **kwargs
        )
        self.parameters_shape = (n_layers, 2  * n_qubits)
    
    def build_circuit(
        self, inputs : torch.Tensor,
        params : torch.Tensor
    )-> Callable:
        """
        Build the circuit function for Sim15 class.

        Parameters
        ----------
        input : torch.tensor
            Input features introduced. They will be encoded in the circuit 
            using angle encoding techniques.
        params : torch.tensor
            Variational paramaters of the ansatz. They will be optimised
            within the model.

        Returns
        -------
        Callable
            Function represesting the quantum circuit.
        """
        rescaled_circuit_inputs = self.rescale_circuit_inputs(inputs)
        self.embedd_inputs_in_circuit(rescaled_circuit_inputs)
        for i in range(self.n_layers):
            idx = 0
            for j in range(self.n_qubits):
                qml.RY(params[i, j], wires = j)
            idx += self.n_qubits
            for j in range(self.n_qubits):
                ctrl = j
                target = (j - 1) % self.n_qubits
                qml.CNOT(wires=[ctrl,target])
            for j in range(self.n_qubits):
                qml.RY(params[i, j + idx], wires = j)
            for j in range(self.n_qubits, 0, -1):
                ctrl = j % self.n_qubits
                target = (j + 1) % self.n_qubits
                qml.CNOT(wires= [ctrl, target])

        if self.output_probabilities:
            return [qml.probs(op = self.operators_tensor_product)]
        else:
            return [qml.expval(
                self.observables[k](k)) for k in self.observables.keys()
            ]

        
class StronglyEntangling(Circuit):
    """
    Class containing StronglyEntanglingAnstaz of
    https://docs.pennylane.ai/en/stable/code/api/pennylane.StronglyEntanglingLayers.html.
    """
    def __init__(
        self, n_qubits : int,  n_layers : int,
        axis_embedding : str, observables : Dict[int, qml.operation.Operator],
        device_name : str = "default.qubit",
        output_probabilities : bool = True,
        data_rescaling : Callable = None, **kwargs
    )-> None:
        """
        Initialise the StronglyEntangling class.

        Parameters 
        ----------
        n_qubits : int
            Number of qubits of the circuit.
        n_layers : int
            Number of times the ansatz is applied to the circuit.
        axis_embedding : str 
            Rotation gate to use for the angle encoding of the inputs.
            Must be one of ['X','Y','Z'].
        observables : Dict[int, qml.operation.Operator]
            Dictionary with pennylane quantum operators. The keys of
            of the dictionary indicate on which qubit the operators act.
        device_name : str, optional
            Pennylane simulator to use. The available devices can be found in
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html.
            Optional parameter. The default value is default.qubit.
        output_probabilities : bool, optional
            If True, the circuit will output a list of probabilities, 
            where each of the items are the probabilities of measuring 
            the vectors of the basis generated by the operators acting on
            each qubit. If False, it will output the expected values of the 
            operators acting on each qubit. The default value is True.
        data_rescaling : Callable, optional
            Function to apply to rescale the inputs that will be encoded.
            The default value is None.
        ** kwargs 
            Keyword arguments to be introduced in the pennylane device.
            More info can be found in 
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html.
        
        """
        
        super().__init__(
            n_qubits, n_layers, axis_embedding, observables,
            device_name, output_probabilities, data_rescaling, **kwargs
        )
        self.parameters_shape = (n_layers, 3  * n_qubits)

    def build_circuit(
        self, inputs : torch.Tensor,
        params : torch.Tensor
    )-> Callable:
        """
        Build the circuit function for StronglyEntangling class.

        Parameters
        ----------
        input : torch.tensor
            Input features introduced. They will be encoded in the circuit 
            using angle encoding techniques.
        params : torch.tensor
            Variational paramaters of the ansatz. They will be optimised
            within the model.

        Returns
        -------
        Callable
            Function represesting the quantum circuit.
        """
        rescaled_circuit_inputs = self.rescale_circuit_inputs(inputs)
        self.embedd_inputs_in_circuit(rescaled_circuit_inputs)
        for i in range(self.n_layers):
            idx = 0
            for j in range(self.n_qubits):
                qml.RZ(params[i, j + idx], wires = j)
                qml.RY(params[i, j + idx + 1], wires = j)
                qml.RZ(params[i, j + idx + 2], wires = j)
                idx += 2
            for j in range(self.n_qubits - 1):
                ctrl = j 
                target = (j + 1) 
                qml.CNOT(wires= [ctrl, target])
            qml.CNOT(wires = [self.n_qubits - 1, 0])

        if self.output_probabilities:
            return [qml.probs(op = self.operators_tensor_product)]
        else:
            return [qml.expval(
                self.observables[k](k)) for k in self.observables.keys()
            ]
            
            