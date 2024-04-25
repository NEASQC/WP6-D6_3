import sqlite3
import unittest
from pipeline_manager import create_table, store_data
from json import dumps
from numpy import ones

# This is an ugly global variable, yes, I know...
DATABASE = 'neasqc_experiments.db'
TABLE = 'test_table'

class TestDatabaseQueries(unittest.TestCase):

    def tearDown(self):
        """ Delete the table from the DB."""
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        query1 = f"DROP TABLE IF EXISTS {TABLE};"
        query2 = f"DROP TABLE IF EXISTS basic;"
        cursor.execute(query1)
        cursor.execute(query2)
        conn.commit()
        conn.close()
        

    def setUp(self):
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


    def test_table_creation_is_successful(self):
        some_cols = [('animal', 'TEXT'), ('population', 'INT')]

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Query to test for absence/presence
        query = f"""SELECT 1 FROM sqlite_master 
                            WHERE type='table' AND name='{TABLE}';
                        """

        # There should be no table before I create it
        cursor.execute(query)
        response_pre_creation = cursor.fetchone()
        self.assertIsNone(response_pre_creation)

        # I create the table
        create_table(DB=DATABASE, TABLE=TABLE, col_definitions=some_cols)

        # There should be a table after I create it
        cursor.execute(query)
        response_post_creation = cursor.fetchone()
        self.assertEqual(response_post_creation, (1,))

        conn.close()


    
    def test_table_creation_has_all_wanted_rows(self):
        some_cols = [('animal', 'TEXT'), 
                    ('population', 'INT'), 
                    ('avg_weight', 'REAL')
                    ]

        # Create the table
        create_table(DB=DATABASE, TABLE=TABLE, col_definitions=some_cols)

        # Get all the columns from the created table
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        query_columns = f"PRAGMA table_info('{TABLE}')"
        cursor.execute(query_columns)
        query_result = cursor.fetchall()

        # Check that I have all the columns I asked for and their type is ok
        for i in range(3):
            db_row = query_result[i]
            self.assertEqual(db_row[1], some_cols[i][0])
            self.assertEqual(db_row[2], some_cols[i][1])

        # And there are no extra columns that have popped up
        self.assertEqual(len(query_result), len(some_cols))

        conn.close()

        

    def test_inserting_a_full_row_is_successful_and_correct(self):
        values = ('apple', 1984, 4.2, dumps(ones(5).tolist()))
        data = {'FRUIT_NAME': values[0],
                'QUANTITY': values[1],
                'PRICE_PER_KG': values[2],
                'GROWTH_HISTORY': values[3]
                }

        store_data(DB=DATABASE, TABLE='basic', data=data)

        #Check that the new data has been added and is correct
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        query = f"SELECT * FROM basic LIMIT 1 OFFSET 0"
        cursor.execute(query)
        query_result = cursor.fetchall()

        # Check that the values used to populate table are indeed there
        for i in range(len(values)):
            self.assertEqual(query_result[0][i], values[i])
        


if __name__ == '__main__':
    unittest.main()
