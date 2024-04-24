import sqlite3
import unittest
from pipeline_manager import create_table

# This is an ugly global variable, yes, I know...
DATABASE = 'neasqc_experiments.db'

class TestDatabaseQueries(unittest.TestCase):

    def tear_down(self, TABLE):
        """ Delete the table from the DB."""
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        query1 = f"DROP TABLE IF EXISTS {TABLE};"
        query2 = f"DROP TABLE IF EXISTS basic;"
        cursor.execute(query1)
        cursor.execute(query2)
        conn.commit()
        conn.close()
        

    def setup(self):
        """ Create the table in the DB. """
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        query = f"""CREATE TABLE IF NOT EXISTS basic (
                FRUIT_NAME TEXT,
                QUANTITY INTEGER,
                PRICE_PER_KG REAL,
                GROWTH_HISTORY JSON
                );"""
        cursor.execute(query)
        conn.commit()
        conn.close()




    def test_table_creation_is_succesful(self):
        some_table = 'test_table'
        some_cols = [('animal', 'TEXT'), ('population', 'INT')]
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Query which tests the existence of said table in the DB
        query =   f"""    SELECT 1 FROM sqlite_master 
                            WHERE type='table' AND name={some_table};
                    """

        # The table doesn't exist before I create it
        with self.assertRaises(sqlite3.OperationalError):
            cursor.execute(query)
        
        # I create the table
        create_table(DB=DATABASE, TABLE=some_table, col_definitions=some_cols)

        # The table now exists
        response_post_creation = cursor.execute(query)
        assert response_post_creation == 1

        tear_down(some_table)
    

    def test_table_creation_has_all_wanted_rows(self):
        TABLE = 'test_table'
        # I create a table with a certain set of rows
        # It contains all of those rows
        # It doesn't contain any other rows
        tear_down(TABLE)
        pass

    def test_table_creation_has_correct_types_for_each_row(self):
        TABLE = 'test_table'
        # I create a table with a certain set of rows
        # Each row has the type I specified when creating
        tear_down(TABLE)
        pass

    def test_inserting_a_full_row_is_succesfull_and_correct(self):
        TABLE = 'test_table'
        # I insert data to populate a row
        # All the specified rows are populated
        # The values in each of them is correct
        tear_down(TABLE)
        pass



# import sqlite3

# # Connect to the SQLite database
# conn = sqlite3.connect('your_database.db')

# # Create a cursor object
# cursor = conn.cursor()

# # Execute the SQL query
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='my_table'")

# # Fetch the result (if any)
# result = cursor.fetchone()

# # Check if the result is not None (i.e., table exists)
# if result:
#     print("Table 'my_table' exists")
# else:
#     print("Table 'my_table' does not exist")

# # Close the cursor and connection
# cursor.close()
# conn.close()


# What happens if I want to add stuff to a column that doesnt exist?

if __name__ == '__main__':
    unittest.main()