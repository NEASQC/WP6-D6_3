import sqlite3
from typing import Dict
import numpy as np
import json

def create_table(DB, TABLE, col_definitions):
    """ Create a table TABLE in database DB."""

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    col_definitions = [' '.join(col_def) for col_def in col_definitions]

    create_table_sql = f"""CREATE TABLE IF NOT EXISTS {TABLE} 
                        ({', '.join(col_definitions)})"""
    cursor.execute(create_table_sql)

    conn.commit()
    conn.close()



def reset_table(DB, TABLE):
    """ Reset the data in the table if it exists, otherwise does nothing.
    DANGEROUS, use with care
    """

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    # BEWARE!!!
    # the "DROP TABLE" command is extremely dangerous, it deletes ALL the
    # data contained in that table unretrievably.
    cursor.execute(f"DROP TABLE {TABLE}")

    conn.commit()
    conn.close()


def access_value(DB, TABLE, col_name, row_nb):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    select_sql = f"SELECT {col_name} FROM {TABLE} WHERE rowid = ?"
    
    cursor.execute(select_sql, (row_nb,))

    result = cursor.fetchone()

    conn.close() # No commit because we haven't changed anything in the table

    return result[0] if result else None


def store_data(DB, TABLE, data):
        conn = sqlite3.connect(DB)
        cursor = conn.cursor()

        columns_string = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in range(len(data))])
        insert_sql = f"INSERT INTO {TABLE} ({columns_string}) VALUES ({placeholders})"

        cursor.execute(insert_sql, tuple(data.values()))

        conn.commit()
        conn.close()


def mock_pipeline(hyperparameters) -> Dict:
    mock_results = {
        'loss':         np.ones(5)*42.2,
        'accuracy':     np.ones(5)*19.84
    }
    return mock_results


def mock_extract_hyperparameters_from_dataframe(row) -> Dict:
    mock_hyperparameters = {
        'nb_qbits':         5,
        'optimizer':        'adam',
        'optimizer_lr':     0.01,
        'ansatz':           'Sim13'
    }
    return mock_hyperparameters


def serialise_results(unserialised_results) -> Dict:
    # TODO: this but for every serial value we get from the results
    # TODO: make sure we feed a dict with only serial values
    serialised_loss = json.dumps(unserialised_results['loss'].tolist())
    serialised_accuracy = json.dumps(unserialised_results['accuracy'].tolist())
    results = {
        'loss':         serialised_loss,
        'accuracy':     serialised_accuracy
    }
    return results




def main():
    nb_xps = 1 # the number of experiments (aka rows in the DF)
    nb_folds = np.arange(2) # the number of fold for our k cross-validation
    random_seeds = [42, 1984, 23]

    # Here there will need to be 1 DB for preliminary, 1 DB for full grid search 
    # and 1 DB for final. They need to be different DB, so total of 3 DB per dataset
    DATABASE = 'neasqc_experiments.db'

    #TODO: one table per experiment-section couple
    TABLE = 'mock_testing'  
    reset_table(DB=DATABASE, TABLE=TABLE)
    #TODO: make sure that we have all the columns that we want and that their 
    # names are exactly what we want
    column_defs = [
    ('id', 'INTEGER PRIMARY KEY'), #TODO: move this to table creation so it's ensured to always be there
    ('loss', 'JSON'), #Note: important that it's JSON and not text for the extra functionalities
    ('accuracy', 'JSON'),
    ('nb_qbits', 'INT'),
    ('optimizer', 'TEXT'),
    ('optimizer_lr', 'REAL'),
    ('ansatz', 'TEXT'),
    ('idx', 'INT'),
    ('seed', 'INT')
    ]
    create_table(DATABASE, TABLE, column_defs)

    for row in range(nb_xps):
        # This is what we need to get from the big pandas dataframe
        hyperparameters = mock_extract_hyperparameters_from_dataframe(row)

        for fold_index in nb_folds:
            hyperparameters['idx'] = fold_index

            for seed in random_seeds:
                hyperparameters['seed'] = seed
                xp_results = mock_pipeline(hyperparameters)
                serialised_xp_results = serialise_results(xp_results)
                hyperparameters.update(serialised_xp_results)
                store_data(DB=DATABASE, TABLE=TABLE, data=hyperparameters)
             
    unique_id = access_value(DATABASE, TABLE, 'id', 5)
    some_loss = access_value(DATABASE, TABLE, 'loss', 3)
    print(unique_id)
    print(json.loads(some_loss))




if __name__ == "__main__":
    main()




