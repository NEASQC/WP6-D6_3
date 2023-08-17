from torch import nn
from lambeq import PennyLaneModel
import torch

# inherit from PennyLaneModel to use the PennyLane circuit evaluation
class Alpha_pennylane_lambeq_model(PennyLaneModel):
    def __init__(self, version_original: bool, reduced_word_embedding_dimension: int, **kwargs):
        PennyLaneModel.__init__(self, **kwargs)

        self.version_original = version_original

        self.reduced_word_embedding_dimension = reduced_word_embedding_dimension

        self.pre_qc_output_size = None

        self.pre_qc = None

        self.param_indexes_dict = {}
        
        self.post_qc = nn.Sequential(nn.Linear(2, 1),
                                nn.Sigmoid())

    
    # To be called after the model is initialized
    ##############################################################################################################
    def update_pre_qc_output_size(self):
        """
        Find the output size of the pre_qc neural network + Create the pre_qc neural network.
        """
        self.pre_qc_output_size = self.find_pre_qc_output_size()

        if self.version_original:
            pre_qc_input_size = self.reduced_word_embedding_dimension
        else:
            # BERT embedding size = 768
            pre_qc_input_size = 768

        self.pre_qc = nn.Sequential(nn.Linear(pre_qc_input_size, self.pre_qc_output_size),
                                    nn.LeakyReLU(0.01))

    def find_pre_qc_output_size(self):
        """
        Find the output size of the pre_qc neural network.
        It is the maximum number of parameters in all the circuits.
        """
        max_number_of_params = 0
        for circuit_pennylane in list(self.circuit_map.values()):
            if circuit_pennylane._symbols.__len__() > max_number_of_params:
                max_number_of_params = circuit_pennylane._symbols.__len__()

        return max_number_of_params
    
    def find_param_indexes_from_diagram_dict(self, diagrams):
        """
        Given a diagram, find the indexes of the parameters in the model.
        Do it once and for all in the init function
        Return a dictionary with the diagram as key and the list of indexes as value.
        """
        for diagram in diagrams:
            diagram_symbols = self.circuit_map[diagram]._symbols
            model_symbols = self.symbols

            param_indexes = []

            for symbol in diagram_symbols:
                try:
                    param_indexes.append(model_symbols.index(symbol))
                except ValueError:
                    print(f"Symbol {symbol} not found in model symbols")
                    return None
                
            self.param_indexes_dict[diagram] = param_indexes
    ##############################################################################################################    
    

    def update_weights_from_embedding(self, param_indexes, param_values):
        """
        Given a list of indexes and a list of values, update the weights of the model
        """
        for index, value in zip(param_indexes, param_values):
            self.weights[index] = torch.nn.Parameter(torch.tensor(value))


    def forward(self, diagrams, bert_embeddings):
        ## Here we go through lists of diagrams and bert embeddings because the Pytorch dataloader returns lists (uses batch)
        # pass the embedding through a simple neural network

        #TODO: Maybe can accelerate this by flattening to a 2D vector the list of embeddings and then passing it through the pre_qc
        #     and then reshaping it to a 3D vector
        if self.version_original:
            # Here we have a list of embeddings for each word in the sentence
            embedding_out = []
            for embedding in bert_embeddings:
                # Here 'embedding' is a list of embeddings for each word in the sentence
                post_qc_output = self.pre_qc(embedding)

                post_qc_output = torch.sum(post_qc_output, dim=0)
                embedding_out.append(post_qc_output)

        else:
            # Here we have a list of embeddings for each sentence
            embedding_out = self.pre_qc(bert_embeddings)

        # update the corresponding weights of the model
        param_indexes_list = [self.param_indexes_dict[diagram] for diagram in diagrams]

        # update the weights of the model
        [self.update_weights_from_embedding(param_indexes, embedding) for param_indexes, embedding in zip(param_indexes_list, embedding_out)]

        # evaluate the circuits
        qc_output = self.get_diagram_output(diagrams)
        
        # pass the concatenated results through a simple neural network
        return self.post_qc(qc_output)