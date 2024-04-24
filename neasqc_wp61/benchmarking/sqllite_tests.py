import sqlite3
import unittest
from pipeline_manager import create_table

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
        # I create a table with a certain set of rows
        # It contains all of those rows
        # It doesn't contain any other rows
        pass

    def test_table_creation_has_correct_types_for_each_row(self):
        # I create a table with a certain set of rows
        # Each row has the type I specified when creating
        pass

    def test_inserting_a_full_row_is_successful_and_correct(self):
        # I insert data to populate a row
        # All the specified rows are populated
        # The values in each of them is correct
        pass


if __name__ == '__main__':
    unittest.main()
