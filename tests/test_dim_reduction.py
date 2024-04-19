import argparse
import unittest

import numpy as np
import os
import pandas as pd
import sys

# The two lines below will be removed when converting the library to a package.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/")
import dim_reduction as dr


class TestDimReduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        Generate a random dataset for our tests.
        """
        number_of_sentences = args.number_of_sentences
        cls.dim_out = args.reduced_dimension
        labels = [np.random.randint(2) for _ in range(number_of_sentences * 2)]
        sentence_embeddings = [
            np.random.rand(args.original_dimension).astype(np.float32)
            for _ in range(number_of_sentences * 2)
        ]

        data = {
            "label": labels[:number_of_sentences],
            "sentence_vectorised": sentence_embeddings[:number_of_sentences],
        }
        cls.df = pd.DataFrame(data)

        data_2 = {
            "label": labels[number_of_sentences:],
            "sentence_vectorised": sentence_embeddings[number_of_sentences:],
        }
        cls.df2 = pd.DataFrame(data_2)
        cls.dim_reduction_funcs = [dr.PCA, dr.ICA, dr.TSVD, dr.UMAP]
        cls.kwargs_dim_reduction_funcs = [
            {"svd_solver": "full", "tol": 9.0},
            {"fun": "cube"},
            {"algorithm": "arpack"},
            {"n_neighbors": 2},
        ]
        cls.dim_reduction_object_list = []
        cls.dim_reduction_object_list_2 = []
        for dim_reduction_func, kwargs in zip(
            cls.dim_reduction_funcs, cls.kwargs_dim_reduction_funcs
        ):
            dim_reduction_object = dim_reduction_func(
                dim_out=cls.dim_out, **kwargs
            )
            dim_reduction_object.fit(data_to_fit=cls.df)
            cls.dim_reduction_object_list.append(dim_reduction_object)
            # second set of objects fitted with a list of dataframes instead of a single dataframe
            dim_reduction_object_2 = dim_reduction_func(cls.dim_out, **kwargs)
            dim_reduction_object_2.fit(data_to_fit=[cls.df, cls.df2])
            cls.dim_reduction_object_list_2.append(dim_reduction_object_2)

        cls.dim_reduction_object_list.extend(cls.dim_reduction_object_list_2)

        cls.reduced_dataset_list = []
        for object in cls.dim_reduction_object_list:
            reduced_dataset = object.reduce_dimension(data_to_reduce=cls.df)
            cls.reduced_dataset_list.append(reduced_dataset)

        # NEXT: MODIFY ALL BELOW TESTS SO THAT THEY USE THIS LIST INSTEAD

    def test_dim_reduction_produces_the_desired_output_dimension(self):
        """
        Test that the available dim reduction techniques output a
        vector with the desired dimension.
        """
        for reduced_dataset in self.reduced_dataset_list:
            with self.subTest(reduced_dataset=reduced_dataset):
                for value in reduced_dataset["reduced_sentence_vector"]:
                    self.assertEqual(value.shape[0], self.dim_out)

    def test_save_and_load_dataset_preserves_desired_dimension(self):
        """
        Test that save dataset function preserves the desired reduced
        dimension after being saved and loaded.
        """

        for dim_reduction_object, reduced_dataset in zip(
            self.dim_reduction_object_list, self.reduced_dataset_list
        ):
            with self.subTest(
                dim_reduction_object=dim_reduction_object,
                reduced_dataset=reduced_dataset,
            ):
                name_dataset = "test_dataset"
                dim_reduction_object.save_dataset(
                    reduced_dataset=reduced_dataset,
                    filename=name_dataset,
                    path="./",
                )
                name_dataset += ".pkl"
                df = pd.read_pickle(name_dataset)
                for value in df["reduced_sentence_vector"]:
                    self.assertEqual(value.shape[0], self.dim_out)

    def test_reduced_embedding_is_populated_by_float_values(self):
        """
        Test that the reduced embedding is populated by float values.
        """
        for reduced_dataset in self.reduced_dataset_list:
            with self.subTest(reduced_dataset=reduced_dataset):
                for sentence_vector in reduced_dataset[
                    "reduced_sentence_vector"
                ]:
                    self.assertTrue(
                        np.all(
                            np.issubdtype(sentence_vector.dtype, np.float32)
                        )
                    )

    def test_not_all_elements_are_equal_in_reduced_embedding(self):
        """
        Test that at least two of the values of the reduced embeddings
        are different.
        """
        for reduced_dataset in self.reduced_dataset_list:
            with self.subTest(reduced_dataset=reduced_dataset):
                for sentence_vector in reduced_dataset[
                    "reduced_sentence_vector"
                ]:
                    self.assertNotEqual(len(set(sentence_vector.tolist())), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ns",
        "--number_of_sentences",
        type=int,
        help="Number of random sentences to generate for testing.",
        default=10,
    )
    parser.add_argument(
        "-od",
        "--original_dimension",
        type=int,
        help="Dimension of the vectors to be reduced.",
        default=768,
    )
    parser.add_argument(
        "-rd",
        "--reduced_dimension",
        type=int,
        help="Desired output reduced dimension.",
        default=3,
    )
    args, remaining = parser.parse_known_args()
    remaining.insert(0, sys.argv[0])
    unittest.main(argv=remaining)
