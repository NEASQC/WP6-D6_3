import pandas as pd

data = {
    'model_type': ["alpha3"], #Let's not forget Alpha4 later on
    'embedding_type': ["BERT"], #which ones???
    'dimensionality_reduction_technique': ["PCA", "ICA", "TSVD", "UMAP", "TSNE"],
    'ansatz': ["Sim14", "Sim15", "StronglyEntangling"],
    'number_of_layers': [i for i in range(1, 3+1)],
    'number_of_qubits': [i for i in range(2, 16+1, 2)],
    'qubit_initialisation_technique': ["constant", "normal_centered_pi", "uniform_between_0_2pi"], #the names must be checked to match actual names
    'optimizer': ["Adam", "SPSA", "SGD", "RMSProp", "Cobyla", "Nelder-Mead", "Adagrad"],
    'optimizer_lr': [0.0001 * (10 ** i) for i in range(5)],
    'optimizer_parameters': [None], #Not sure what we want here
}

number_of_epochs = 20 #Fixed for grid search

df = pd.DataFrame(data)

for model in df['model_type']:
    for embedding in df['embedding_type']:
        for dim_reduc in df['dimensionality_reduction_technique']:
            for ansatz in df['ansatz']:
                for nb_layers in df['number_of_layers']:
                    for nb_qubits in df['number_of_layers']:
                        for qubit_init in df['qubit_initialisation_technique']:
                            for optim in df['optimizer']:
                                for lr in df['optimizer_lr']:
                                    for nb_epochs in df['optimizer_number_of_epochs']:
                                        for opti_params in df['optimizer_parameters']:
                                            print(f"Model: {model} \t embedding : {embedding} \t dimensionality reduction : {dim_reduc} \t ansatz : {ansatz}")
