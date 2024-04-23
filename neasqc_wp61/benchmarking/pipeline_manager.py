import sqlite3
from typing import Dict
import numpy as np

"""
Let's not write complicated tests for these. Just very basic things like:
1) After I create a table, it's there (and it wasn't there before)
2) When I retrieve data from the table, it's the correct row/col (hardcode smth 
and see whether it fetches it correctly)
3) If I ask for a row/col that's not there, I do get None
4) Something that I insert is there aftern I'v inserted it (and it wasn't there before)
"""

def create_table(DB, TABLE, col_names):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    create_table_sql = f"""CREATE TABLE IF NOT EXISTS {TABLE} 
                        ({', '.join(col_names)})"""
    cursor.execute(create_table_sql)

    conn.commit()
    conn.close()


def reset_table(DB, TABLE):
    """ Resets the data in the table if it exists, otherwise does nothing. """

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
        'loss':         0.05,
        'accuracy':     0.2
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



def main():
    nb_xps = 1 # the number of experiments (aka rows in the DF)
    nb_folds = np.arange(2) # the number of fold for our k cross-validation
    random_seeds = [42, 1984, 23]

    # Here there will need to be 1 DB for preliminary, 1 DB for full grid search 
    # and 1 DB for final. They need to be different DB, so total of 3 DB per dataset
    DATABASE = 'neasqc_experiments.db'
    TABLE = 'mock_testing'
    #reset_table(DB=DATABASE, TABLE=TABLE)
    column_names = ['loss', 'accuracy', 'nb_qbits', 'optimizer', 'optimizer_lr',
                     'ansatz', 'idx', 'seed']
    create_table(DATABASE, TABLE, column_names)

    for row in range(nb_xps):
        # This is what we need to get from the big pandas dataframe
        hyperparameters = mock_extract_hyperparameters_from_dataframe(row)

        for fold_index in nb_folds:
            hyperparameters['idx'] = fold_index

            for seed in random_seeds:
                hyperparameters['seed'] = seed
                xp_results = mock_pipeline(hyperparameters)
                # It's easier to feed one dict into the next function and cheap 
                # so we merge them
                hyperparameters.update(xp_results)
                store_data(DB=DATABASE, TABLE=TABLE, data=hyperparameters)
             
    #val = access_value(DATABASE, TABLE, 'loss', 2)
    #print(val)



if __name__ == "__main__":
    main()




