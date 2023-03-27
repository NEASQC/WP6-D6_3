#from module.dataset_wrapper import *
#from module.parameterised_quantum_circuit import *
from module.alpha_model import *


import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

import time

class alpha_trainer(nn.Module):
    """Trains a neural network that takes sentence BERT embeddings as input and maps them to the parameters in a parameterised quantum circuit.

    ........

    Attributes
    ----------
    
    seed: int
        Random seed setting
    wrapper: dataset_wrapper
        Generates dataset_wrapper object using filename
    sentences: list
        list of sentences in the dataset
    sentence_types: list
        list of sentence types(grammatical structures)
    sentence_labels: list
        list of sentence classification labels
    bert_embeddings: list[list[floats]]
        The BERT embeddings for each word in each sentence.
    BertDim: int
        Dimension of the BERT embedding vectors
    pre_net: nn.Linear
        Network that takes Bert Embedding input and maps it to an intermediate dimension layer.
    pre_net_max_params: nn.Linear
        Network that maps intermediate diimension to a layer with dimension equal to an estimated maximum number of parameters required for the paramterised quantum circuits.
    cascade = nn.ParameterList(nn.Linear)
        Network that maps the maximum parameters kayer to a layer with dimension equal to the number of parameters required for a particular quantum circuit.

    """
    
    def __init__(self,filename:str, seed: int):
        """Initialises alpha_trainer.

        Defines the attributes in alpha_trainer, inculding the neural network.
        

        Parameters
        ----------
        filename : str
            Input data json file
        seed: int
            Random seed setting for reproducibility
            

        """
        super().__init__()
        #Call alpha model class
        self.model = alpha_model(filename, seed)
        
        ###Define the noprmalisation factor for normalised cross entropy loss
        #p = (sum(np.array(self.sentence_labels)==[1,0])/len(self.sentence_labels))[0]
        #p = (sum(np.array(self.sentence_labels[0:10])==[1,0])/len(self.sentence_labels[0:10]))[0]
        #self.normalisation_factor = len(self.sentence_labels[0:10])*(p*np.log(p) +(1-p)*np.log(1-p))
        #self.normalisation_factor = len(self.model.sentences)*100
        self.normalisation_factor = len(self.model.sentences)
        
        

        
    def train(self, number_of_epochs):
        """Trains model.

        Uses a binary cross entropy loss criterion and stochastic gradient descent optimiser to train the model of a number of epochs.:

        Parameters
        ----------
        number_of_epochs: int
            the number of epochs over which you loop over the data and train the model
        
        Returns
        -------
        Dataset: np.array()
            array of loss for each epoch
            
        """
        ###Training the model
        
        #criterion = nn.CrossEntropyLoss()
        
        def CE_loss(input,target):
            return torch.exp(target[0]*torch.log(input[0]+0.00001) + (1.0-target[0])*torch.log(1.00001-input[0]))
        #criterion = nn.BCELoss()
        criterion = CE_loss
        #optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # generation loop
        loss_array = []
        accuracy_array = []
        for epoch in range(number_of_epochs):
            tic = time.perf_counter()
            # sentence loop
            number_of_sentences = len(self.model.sentences)
            running_loss = 0
            accuracy=0
            for i,specific_sentence in enumerate(self.model.sentences):
                print(f"Epoch: {epoch+1}/{number_of_epochs}    Sentence: {i+1}/{number_of_sentences}", end='\r')
                sentence_index = self.model.sentences.index(specific_sentence)
                sentence_label = self.model.sentence_labels[sentence_index]
                       
                # 1. forward step: takes sentence embeddings and outputs input parameters to pqc --> parameters, run_circuit(parameters) --> output=[x,1-x]
                output = self.model.forward(specific_sentence)
                
                # 3. compute loss(compare output to sentence label)
                loss = criterion(output, target=torch.Tensor(sentence_label))
                
                
                accuracy += np.sqrt((output.detach().numpy()[0]-sentence_label[0])**2)
                
                    
                # 4. backward step --> updated network
                optimizer.zero_grad()
                loss.backward()
                
   
                #print("list(self.parameters()) = ",list(self.parameters()), "\n \n \n")
                optimizer.step()
                
                #print stats
                running_loss += loss.item()
            
            toc = time.perf_counter()
            print("\n")
            print(f"Epoch {epoch} time taken: {(toc - tic)/60.0:0.4f} minutes")
            print(f"Estimated remaining time: {((number_of_epochs-epoch-1)*((toc - tic)/60.0)):0.4f} minutes")
            print("Loss = ", running_loss/self.normalisation_factor, "\n")
            loss_array.append(running_loss/self.normalisation_factor)
            accuracy_array.append(accuracy/len(self.model.sentences))
            print("Loss Array = ", loss_array, "\n")
            print("Accuracy Array = ", accuracy_array, "\n")
                
        return np.array(loss_array), np.array(accuracy_array)
                
        