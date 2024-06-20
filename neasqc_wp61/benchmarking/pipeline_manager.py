import sqlite3
from typing import Dict
import numpy as np
import json

def create_table(DB, TABLE, col_definitions)-> None: #NOTE: actual code to be kept
    """ Create a table TABLE in database DB if it doesn't already exist.
    Column definitions are assumed to be of the form
    """

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    col_definitions = [' '.join(col_def) for col_def in col_definitions]

    create_table_sql = f"""CREATE TABLE IF NOT EXISTS {TABLE} 
                        ({', '.join(col_definitions)});"""
    cursor.execute(create_table_sql)

    conn.commit()
    conn.close()


def store_data(DB, TABLE, data:Dict) -> None: #NOTE: actual code to be kept
    """ Store data in the table TABLE of database DB.
    Data is assumed to be in the form of a dictionary where each key is the name
     of a column in the table and each value is to be stored.
    """
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    columns_string = ', '.join(data.keys())
    data_values_placeholders = ', '.join(['?' for _ in range(len(data))])
    insert_sql = f"""INSERT INTO {TABLE} ({columns_string}) 
                VALUES ({data_values_placeholders});"""

    cursor.execute(insert_sql, tuple(data.values()))

    conn.commit()
    conn.close()


def serialise_results(unserialised_results:Dict) -> Dict:
    """ Takes a dictionary where keys are DB column names and values are numpy 
    arrays and returns the same dictionary structure but where the values have 
    been converted to json.
    """
    # TODO: should be tested by being fed a dictionary and checking that it does serialise it
    # TODO: this but for every serial value we get from the results
    # TODO: make sure we feed a dict with only serial values
    serialised_loss = json.dumps(unserialised_results['loss'].tolist())
    serialised_accuracy = json.dumps(unserialised_results['accuracy'].tolist())
    results = {
        'loss':         serialised_loss,
        'accuracy':     serialised_accuracy
    }
    return results



def reset_table(DB, TABLE): #NOTE: get rid of it after development phase
    """ Reset the data in the table if it exists, otherwise does nothing.
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


def access_value(DB, TABLE, col_name, row_nb): #NOTE: get rid of it after development phase
    """ NOTE: This is only an example of how to access something in the table, does 
    not need to be tested because will not be used in the code per se. We'll 
    use SQL requests instead (from command line)."""

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    select_sql = f"SELECT {col_name} FROM {TABLE} WHERE rowid = ?"
    
    cursor.execute(select_sql, (row_nb,))

    result = cursor.fetchone()

    conn.close() # No commit because we haven't changed anything in the table

    return result[0] if result else None




#NOTE: get rid of it after development phase
def mock_pipeline(hyperparameters) -> Dict: #TODO : replace by real stuff
    mock_results = {
        'loss':         np.ones(5)*42.2,
        'accuracy':     np.ones(5)*19.84
    }
    return mock_results

#NOTE: get rid of it after development phase
def mock_extract_hyperparameters_from_dataframe(row) -> Dict: #TODO : replace by real stuff
    mock_hyperparameters = {
        'nb_qbits':         5,
        'optimizer':        'adam',
        'optimizer_lr':     0.01,
        'ansatz':           'Sim13'
    }
    return mock_hyperparameters







def main():
    nb_xps = 1 # the number of experiments (aka rows in the DF)
    nb_folds = np.arange(2) # the number of fold for our k cross-validation
    random_seeds = [42, 1984, 23]

    # Here there will need to be 1 DB for preliminary, 1 DB for full grid search 
    # and 1 DB for final. They need to be different DB, so total of 3 DB per dataset
    DATABASE = 'neasqc_experiments.db'

    #TODO: one table per experiment-section couple
    TABLE = 'mock_testing'  #TODO: change once ALL testing is done
    #reset_table(DB=DATABASE, TABLE=TABLE) # TODO: remove, this will not be part of our proper code
    #TODO: make sure that we have all the columns that we want and that their 
    # names are exactly what we want
    # NOTE : once set, the strcuture of the table (columns) should NOT be 
    # changed along the way, so we need to make sure we have all the columns 
    # we need and want for the whole experiment
    column_defs = [
    ('id', 'INTEGER PRIMARY KEY'), # NOTE: this is crucial, always needs to be there (and first)
    ('loss', 'JSON'), #Note: important that it's JSON and not text for the extra functionalities
    ('accuracy', 'JSON'),
    ('nb_qbits', 'INT'),
    ('optimizer', 'TEXT'),
    ('optimizer_lr', 'REAL'),
    ('ansatz', 'TEXT'),
    ('kfold_idx', 'INT'),
    ('seed', 'INT')
    ]
    create_table(DATABASE, TABLE, column_defs) #TODO : this action might need to be performed somewhere else on the way

    for row in range(nb_xps):
        # This is what we need to get from the big pandas dataframe
        hyperparameters = mock_extract_hyperparameters_from_dataframe(row)

        for fold_index in nb_folds:
            hyperparameters['kfold_idx'] = fold_index

            for seed in random_seeds:
                hyperparameters['seed'] = seed
                xp_results = mock_pipeline(hyperparameters) #TODO: replace by actual call to real pipeline
                serialised_xp_results = serialise_results(xp_results)
                hyperparameters.update(serialised_xp_results)
                store_data(DB=DATABASE, TABLE=TABLE, data=hyperparameters)

    # TODO: remove, these are just examples
    unique_id = access_value(DATABASE, TABLE, 'id', 5)
    some_loss = access_value(DATABASE, TABLE, 'loss', 3)
    print(unique_id)
    print(json.loads(some_loss))




if __name__ == "__main__":
    main()




