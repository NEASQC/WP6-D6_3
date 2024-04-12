import pandas as pd
import itertools

model_types = ["alpha3"]
embedding_type = ["BERT", "ember"]
dimensionality_reduction_technique = ["ICA", "UMAP", "TSNE"]
ansatz = ["Sim14", "Sim15", "StronglyEntangling"]
number_of_layers = [2]
number_of_qubits = [8]
qubit_initialisation_technique = ["norm", "rescaled_unif"]
optimizer = ["Adam", "RMSProp"] 
optimizer_lr = [0.001] 
mlp_init = ["unif", "norm"]

combinations = list(itertools.product(model_types, embedding_type, dimensionality_reduction_technique, ansatz, number_of_layers, number_of_qubits, qubit_initialisation_technique, optimizer, optimizer_lr, mlp_init))

unique_identifiers = [f"{i+1:05d}" for i in range(len(combinations))]

df = pd.DataFrame(combinations, columns=['models', 'embeddings', 'dimensionality_reduction_technique', 'ansatz', 'nb_layers', 'nb_qubits', 'qubit_init_tecchnique', 'optimizer', 'optimizer_lr', 'mlp_init' ])

df.insert(0, 'unique_id', unique_identifiers)

df.to_csv('exp_hyperparameters_PRELIM.tsv', sep='\t', index=False)

#print(df)
