import pandas as pd
import itertools

model_types = ["alpha3"] #Let's not forget Alpha4 later on
embedding_type = ["BERT", "ember"]
dimensionality_reduction_technique = ["PCA", "ICA", "TSVD", "UMAP", "TSNE"] #add the MLP later on?
ansatz = ["Sim14", "Sim15", "StronglyEntangling"]
number_of_layers = [i for i in range(1, 3+1)]
number_of_qubits = [i for i in range(2, 16+1, 2)]
qubit_initialisation_technique = ["norm", "rescaled_unif"]
optimizer = ["Adadelta", "Adagrad", "Adam", "AdamW", "SPSA", "SGD", "RMSProp", "Cobyla", "Nelder-Mead", ""]
optimizer_lr = [0.0001, 0.001, 0.01, 0.1] #1
mlp_init = ["unif", "norm"]

combinations = list(itertools.product(model_types, embedding_type, dimensionality_reduction_technique, ansatz, number_of_layers, number_of_qubits, qubit_initialisation_technique, optimizer, optimizer_lr, mlp_init))

df = pd.DataFrame(combinations, columns=['models', 'embeddings', 'dimensionality_reduction_technique', 'ansatz', 'nb_layers', 'nb_qubits', 'qubit_init_tecchnique', 'optimizer', 'optimizer_lr', 'mlp_init' ])

print(df)
