import argparse
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd

from typing import Dict

import use_alpha_3_4

exp_config_table_path = (
    "./exp_hyperparameters_PRELIM.tsv"  # Insert path when generated
)


class PipelineArgClass:
    def __init__(
        self,
        model: str,
        dataset: str,
        split_id: int,
        seed: int,
        epochs: int,
        optimiser: str,
        batch_size: int,
        learning_rate: float,
        embedder: str,
        dim_reduction: str,
        circuit: str,
        n_qubits: int,
        n_layers: int,
        data_rescaling: str,
        circuit_init: str,
        mlp_init: str,
    ):
        self.model = model
        self.dataset = dataset
        self.split_id = split_id
        self.seed = seed
        self.epochs = epochs
        self.optimiser = optimiser
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedder = embedder
        self.dim_reduction = dim_reduction
        self.circuit = circuit
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.data_rescaling = data_rescaling
        self.circuit_init = circuit_init
        self.mlp_init = mlp_init


def create_table(
    DB, TABLE, col_definitions
) -> None:  # NOTE: actual code to be kept
    """Create a table TABLE in database DB if it doesn't already exist.
    Column definitions are assumed to be of the form
    """

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    col_definitions = [" ".join(col_def) for col_def in col_definitions]

    create_table_sql = f"""CREATE TABLE IF NOT EXISTS {TABLE} 
                        ({', '.join(col_definitions)})"""
    cursor.execute(create_table_sql)

    conn.commit()
    conn.close()


def store_data(DB, TABLE, data: Dict) -> None:  # NOTE: actual code to be kept
    """Store data in the table TABLE of database DB.
    Data is assumed to be in the form of a dictionary where each key is the name
     of a column in the table and each value is to be stored.
    """
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    # data = data.to_dict()

    columns_string = ", ".join(data.keys())
    data_values_placeholders = ", ".join(["?" for _ in range(len(data))])
    insert_sql = f"""INSERT INTO {TABLE} ({columns_string}) 
                VALUES ({data_values_placeholders})"""

    cursor.execute(insert_sql, tuple(data.values()))

    conn.commit()
    conn.close()


def serialise_results(unserialised_results: Dict) -> Dict:
    """Takes a dictionary where keys are DB column names and values are numpy
    arrays and returns the same dictionary structure but where the values have
    been converted to json.
    """
    # TODO: this but for every serial value we get from the results
    # TODO: make sure we feed a dict with only serial values
    serialised_train_labels = json.dumps(
        process_numpy_floats(unserialised_results["train_labels"])
    )
    serialised_train_loss = json.dumps(
        process_numpy_floats(unserialised_results["train_loss"])
    )
    serialised_train_preds = json.dumps(
        process_numpy_floats(unserialised_results["train_preds"])
    )

    serialised_train_probs = json.dumps(
        process_numpy_floats(unserialised_results["train_probs"])
    )

    serialised_val_labels = json.dumps(
        process_numpy_floats(unserialised_results["val_labels"])
    )
    serialised_val_loss = json.dumps(
        process_numpy_floats(unserialised_results["val_loss"])
    )
    serialised_val_preds = json.dumps(
        process_numpy_floats(unserialised_results["val_preds"])
    )

    serialised_val_probs = json.dumps(
        process_numpy_floats(unserialised_results["val_probs"])
    )

    results = {
        "train_labels": serialised_train_labels,
        "train_loss": serialised_train_loss,
        "train_preds": serialised_train_preds,
        "train_probs": serialised_train_probs,
        "val_labels": serialised_val_labels,
        "val_loss": serialised_val_loss,
        "val_preds": serialised_val_preds,
        "val_probs": serialised_val_probs,
    }
    return results


def reset_table(DB, TABLE):  # NOTE: get rid of it after development phase
    """Reset the data in the table if it exists, otherwise does nothing.
    DANGEROUS, use with care.
    Note: Again, this will not be used in our propoer code.
    """

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    # BEWARE!!!
    # the "DROP TABLE" command is extremely dangerous, it deletes ALL the
    # data contained in that table unretrievably.
    cursor.execute(f"DROP TABLE {TABLE}")

    conn.commit()
    conn.close()


def access_value(
    DB, TABLE, col_name, row_nb
):  # NOTE: get rid of it after development phase
    """NOTE: This is only an example of how to access something in the table, does
    not need to be tested because will not be used in the code per se. We'll
    use SQL requests instead (from command line)."""

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    select_sql = f"SELECT {col_name} FROM {TABLE} WHERE rowid = ?"

    cursor.execute(select_sql, (row_nb,))

    result = cursor.fetchone()

    conn.close()  # No commit because we haven't changed anything in the table

    return result[0] if result else None


def process_numpy_floats(
    list_to_process: list,
) -> list:
    array_np_float = np.array(list_to_process, dtype=np.float32)
    array_py_float = array_np_float.astype(float)
    list_py_float = array_py_float.tolist()

    return list_py_float


# NOTE: get rid of it after development phase
def mock_pipeline(hyperparameters) -> Dict:  # TODO : replace by real stuff
    mock_results = {"loss": np.ones(5) * 42.2, "accuracy": np.ones(5) * 19.84}
    return mock_results


# NOTE: get rid of it after development phase
def mock_extract_hyperparameters_from_dataframe(
    row,
) -> Dict:  # TODO : replace by real stuff
    mock_hyperparameters = {
        "nb_qbits": 5,
        "optimizer": "adam",
        "optimizer_lr": 0.01,
        "ansatz": "Sim13",
    }
    return mock_hyperparameters


def main():
    # Here there will need to be 1 DB for preliminary, 1 DB for full grid search
    # and 1 DB for final. They need to be different DB, so total of 3 DB per dataset
    DATABASE = "neasqc_experiments.db"

    # TODO: one table per experiment-section couple
    TABLE = "bert_testing_1"
    reset_table(DATABASE, TABLE)
    # TODO: change once ALL testing is done
    # TODO: make sure that we have all the columns that we want and that their
    # names are exactly what we want
    # NOTE : once set, the strcuture of the table (columns) should NOT be
    # changed along the way, so we need to make sure we have all the columns
    # we need and want for the whole experiment

    column_defs = [
        ("id", "INTEGER PRIMARY KEY"),
        # NOTE: this is crucial, always needs to be there (and first)
        # Do we want our Hyperparam REF here?, or make it the primary key?
        ("model", "TEXT"),
        ("split_id", "INT"),  # The id for the validation split
        ("seed", "INT"),
        ("dataset", "TEXT"),  # The dataset used (want int or str good?)
        ("epochs", "INT"),
        ("optimiser", "TEXT"),
        ("batch_size", "INT"),
        ("learning_rate", "REAL"),
        ("embedder", "TEXT"),
        ("dim_reduction", "TEXT"),
        ("ansatz", "TEXT"),
        ("num_qubits", "INT"),
        ("num_layers", "INT"),
        ("data_rescaling", "TEXT"),
        ("circuit_params_init", "TEXT"),
        ("mlp_params_init", "TEXT"),
        ("train_labels", "JSON"),
        ("train_loss", "JSON"),
        ("train_preds", "JSON"),
        ("train_probs", "JSON"),
        ("val_labels", "JSON"),
        ("val_loss", "JSON"),
        ("val_preds", "JSON"),
        ("val_probs", "JSON"),
        ("train_runtime", "REAL"),
    ]
    create_table(
        DATABASE, TABLE, column_defs
    )  # TODO : this action might need to be performed somewhere else on the way

    random_seeds = [42, 1984, 23]

    exp_config_df = pd.read_csv(exp_config_table_path, delimiter="\t", nrows=5)

    for row in range(exp_config_df.shape[0]):
        # This is what we need to get from the big pandas dataframe
        hyperparameters = exp_config_df.iloc[row]

        for split_id in range(3):
            hyperparameters["split_id"] = split_id

            for seed in random_seeds:
                print(type(hyperparameters["optimizer"]))

                pipeline_args = PipelineArgClass(
                    model=(
                        "alpha_3"
                        if str(hyperparameters["models"]) == "alpha3"
                        else "alpha4"
                    ),
                    dataset="bert_test",
                    split_id=split_id,
                    seed=seed,
                    epochs=10,
                    optimiser=hyperparameters["optimizer"],
                    batch_size=2,
                    learning_rate=int(hyperparameters["optimizer_lr"]),
                    embedder=(
                        "Bert"
                        if str(hyperparameters["embeddings"]) == "BERT"
                        else "none"
                    ),
                    dim_reduction=str(
                        hyperparameters["dimensionality_reduction_technique"]
                    ),
                    circuit=str(hyperparameters["ansatz"]),
                    n_qubits=int(hyperparameters["nb_qubits"]),
                    n_layers=int(hyperparameters["nb_layers"]),
                    circuit_init="none",
                    mlp_init="none",
                    data_rescaling="none",
                )

                xp_results = use_alpha_3_4.main(pipeline_args)
                """for key, value in xp_results.items():
                    print(f"Key: {key}, Val: {value}")
                    if key == "val_loss":
                        # print(f"Key: {key}, Item1: {type(value[0][0][0])}")
                        print(isinstance(value[0], np.float32))
"""
                serialised_xp_results = serialise_results(xp_results)
                xp_results.update(serialised_xp_results)

                store_data(DB=DATABASE, TABLE=TABLE, data=xp_results)

    # TODO: remove, these are just examples
    """
    unique_id = access_value(DATABASE, TABLE, "id", 5)
    some_loss = access_value(DATABASE, TABLE, "loss", 3)
    print(unique_id)
    print(json.loads(some_loss))
    """


if __name__ == "__main__":
    main()
