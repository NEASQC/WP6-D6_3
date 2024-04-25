import pandas as pd
import itertools

model_types = ["alpha3"] #Let's not forget Alpha4 later on
embedding_type = ["BERT", "ember"]
dimensionality_reduction_technique = ["PCA", "ICA", "TSVD", "UMAP", "TSNE"] #add the MLP later on?
ansatz = ["Sim14", "Sim15", "StronglyEntangling"]
number_of_layers = [1,2,3]
number_of_qubits = [i for i in range(2, 17, 2)]
qubit_initialisation_technique = ["norm", "rescaled_unif"]
optimizer = ["Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "ASGD", "LBFGS", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD"] #None of those available... "SPSA", "Cobyla", "Nelder-Mead"
optimizer_lr = [0.0001, 0.001, 0.01, 0.1] #1
mlp_init = ["unif", "norm"]

combinations = list(itertools.product(model_types, embedding_type, dimensionality_reduction_technique, ansatz, number_of_layers, number_of_qubits, qubit_initialisation_technique, optimizer, optimizer_lr, mlp_init))

df = pd.DataFrame(combinations, columns=['models', 'embeddings', 'dimensionality_reduction_technique', 'ansatz', 'nb_layers', 'nb_qubits', 'qubit_init_tecchnique', 'optimizer', 'optimizer_lr', 'mlp_init' ])

df.to_csv('exp_hyperparameters_FULLG.tsv', sep='\t', index=False)

print(df)
