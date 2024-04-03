import pandas as pd

data = {
    'model_type': [None, None],
    'embedding_type': [None, None],
    'dimensionality_reduction_technique': [None, None],
    'ansatz': [None, None],
    'number_of_layers': [None, None],
    'number_of_qubits': [None, None],
    'qubit_initialisation_technique': [None, None],
    'dimensionality_reduction_technique': [None, None],
    'optimizer': [None, None],
    'optimizer_lr': [None, None],
    'optimizer_number_of_epochs': [None, None],
    'optimizer_parameters': [None, None],
}

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
                                            #call model
