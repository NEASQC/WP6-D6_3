import sqlite3

"""
Let's not write complicated tests for these. Just very basic things like:
1) After I create a table, it's there (and it wasn't there before)
2) When I retrieve data from the table, it's the correct row/col (hardcode smth 
and see whether it fetches it correctly)
3) If I ask for a row/col that's not there, I do get None
4) Something that I insert is there aftern I'v inserted it (and it wasn't there before)
"""

def create_table(name):
    """ Connects to the database, or creates it if it doesn't already exist."""

    # Open access to DB
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    # Design SQL query
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS quantum_circuits (
        id INTEGER PRIMARY KEY,
        nb_qbits INTEGER,
        nb_layers INTEGER,
        ansatz TEXT
    )
    """
    # Execute SQL query
    cursor.execute(create_table_sql)

    # Close access to DB
    conn.commit()
    conn.close()



def get_value(column_number, row_number, name):
    """ Returns the value located at a specific column / row (or None). """

    # Open access to DB
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    # Design SQL query
    select_sql = f"""
    SELECT *
    FROM quantum_circuits
    ORDER BY id
    LIMIT 1 OFFSET {row_number - 1}
    """

    # Execute SQL query
    cursor.execute(select_sql)
    result = cursor.fetchone()

    # Close access to DB (no commit because we have not changed it)
    conn.close()

    return result[column_number] if result else None # Always returns smth



def insert_data(nb_qbits, nb_layers, ansatz, name):
    """ Inserts experimental data results into the database."""

    # Open access to DB
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    # Design SQL query
    insert_sql = """
    INSERT INTO quantum_circuits (nb_qbits, nb_layers, ansatz)
    VALUES (?, ?, ?)
    """

    # Execute SQL query
    cursor.execute(insert_sql, (nb_qbits, nb_layers, ansatz))

    # Close access to DB
    conn.commit()
    conn.close()


def reset_table(name):
    """ Resets the data in the table if it exists, otherwise does nothing. """

    # Open access to DB
    conn = sqlite3.connect(name)
    cursor = conn.cursor()

    # Check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quantum_circuits'")
    table_exists = cursor.fetchone()

    if table_exists:
        # Design SQL query
        delete_sql = """
        DELETE FROM quantum_circuits
        """

        # Execute SQL query
        cursor.execute(delete_sql)

        # Close access to DB
        conn.commit()
        conn.close()

        print("Table successfully reset!")
    else:
        print("Table does not exist. No action taken.")




def main():
    """
    Here, since we're not exactly which order data will be stored in, we need to
     make ABSOLUTELY sure that each row will be unique and recognisable, so for 
     instance having columns fopr dataset_split AND for random_seed AND for 
     xp_stage (like preliminary, grd search, final) is a good idea!

     The idea would be to
        1) Create the table
        2) Launch the experiments, each run (one set of HP + one split of the 
        dataset + 1 seed) will be 1 row in the DB.
        3) The pipeline spits out the results loss/accuracy/etc
        4) We have some function that acts as glue code: transforms list of HP 
        + split of dataset + seed + results of pipeline into something like the 
        DATA here.
        5) Write one row of DATA (corresponding to that one run) to the table
    """

    #STEP 1 here
    table_name = 'experimental_results'

    reset_table(name=table_name) # If you want to start off fresh, uncomment this line

    create_table(name=table_name) # Only created if it doesn't exist, otherwise appends

    # STEP 2 here

    # STEP 3 here
    data = [
        (4, 3, 'Sim14'),
        (5, 2, 'Sim13'),
        (3, 4, 'StronglyEntangling'),
        (6, 2, 'Sim13'),
    ]

    # STEP 4 here 

    # STEP 5 here
    for d in data:
        insert_data(*d, name=table_name)

    # Example :
    column_number = 3  # Example: 1 for nb_qbits, 2 for nb_layers, 3 for ansatz
    row_number = 2  # Example: 2 for the second row
    value = get_value(column_number, row_number, name=table_name)
    print(f"The value at column {column_number} and row {row_number} is: {value}")

if __name__ == "__main__":
    main()




