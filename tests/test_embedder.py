import ast
import os
import random
import sys

import numpy as np
import pandas as pd
import unittest

from itertools import product, combinations
from typing import Union

current_path = os.path.dirname(os.path.abspath(__file__))
embedder_path = current_path + "/../neasqc_wp61/models/quantum/"

sys.path.append(embedder_path)

import embedder as emb

dataset_path = (
    current_path
    + "/../neasqc_wp61/data/datasets/reduced_amazonreview_train.tsv"
)


# Define helper functions


def random_dataset_sample(dataset: pd.DataFrame, nrows: int) -> pd.DataFrame:
    """
    Returns a randomly selected subset of a dataset.

    Parameters
    ----------
    dataset: pd.DataFrame
        The dataset containing the natural language data we wish to
        sample from in order to perform our tests.
    nrows: int
        The number of rows that we want to randomly select. This will be
        the number of rows of the output DataFrame.

    Returns
    -------
    random_subset:
        A Dataframe consisting of a randomly selected subset of rows
        from the input dataset.
    """
    shuffled_df = dataset.sample(frac=1)
    random_subset = shuffled_df.head(nrows)

    return random_subset


def check_dimension(
    vectors: list, is_sentence_embedding: bool, dim: int
) -> bool:
    """
    Checks that the dimensions of a sentence or word embedding are
    correct.

    Parameters
    ----------
    vectors: list
        A list of NumPy arrays representing vectors. The arrays may be
        nested or not.
    is_sentence_embedding: bool
        Specifies whether we are checking for the dimension of a
        sentence embedding vector or a vector of word embeddings.
    dim: int
        The dimension we wish to check matches the size of our vectors.

    Returns
    -------
    bool
        True if all vectors have the correct dimension, False otherwise.
    """
    if is_sentence_embedding:
        for vector in vectors:
            if isinstance(vector, np.ndarray):
                if vector.shape[0] != dim:
                    return False
            else:
                return False
    else:
        for vector in vectors:
            if isinstance(vector, list):
                for element in vector:
                    if element.shape[0] != dim:
                        return False
            else:
                return False
    return True


def check_sentence_embedding_type(x: list) -> bool:
    """
    Checks if a list consists of elements of type
    np.ndarray[np.float32], the expected type for a sentence embedding.

    Parameters
    ----------
    x: list
        The list to check.

    Returns
    -------
    bool
        True if x is of type list[np.ndarray[np.float32]], False
        otherwise.
    """
    for embedding in x:
        if isinstance(embedding, np.ndarray):
            return np.all(np.issubdtype(embedding.dtype, np.float32))

    return False


def check_word_embedding_type(x: list) -> bool:
    """
    Checks if a list consists of elements of type
    list[np.ndarray[np.float32]], the expected type for a word
    embedding.

    Parameters
    ----------
    x: list
        The list to check.

    Returns
    -------
    bool
        True if x is of type list[list[np.ndarray[np.float32]]], False
        otherwise.
    """
    for vector in x:
        if isinstance(vector, list):
            return all(
                np.all(np.issubdtype(embedding.dtype, np.float32))
                for embedding in vector
            )

    return False


# Define unit tests


class TestEmbedder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Generates a test dataset and a list of all different embedder
        objects in order to perform unit tests on them.
        """
        full_dataset = pd.read_csv(dataset_path, delimiter="\t")

        # Generate a random subset of the full dataset, from which we
        # will generate further random subsets for our tests
        cls.dataset = random_dataset_sample(full_dataset, nrows=100)
        embedding_type_params = [True, False]
        casing_params = [True, False]
        embedder_params = [
            (x, y) for x, y in product(embedding_type_params, casing_params)
        ]
        # Bert embedder objects to be tested
        cls.bert_object_list = [
            emb.Bert(
                dataset=random_dataset_sample(cls.dataset, nrows=5),
                is_sentence_embedding=params[0],
                cased=params[1],
            )
            for params in embedder_params
        ]
        # FastText embedder objects to be tested
        cls.fasttext_object_list = [
            emb.FastText(
                dataset=random_dataset_sample(cls.dataset, nrows=5),
                is_sentence_embedding=params[0],
                cased=params[1],
            )
            for params in embedder_params
        ]
        # FastText embedder objects with reduced dimensions to be tested
        cls.fasttext_dim_object_list = [
            emb.FastText(
                dataset=random_dataset_sample(cls.dataset, nrows=5),
                is_sentence_embedding=params[0],
                cased=params[1],
                dim=random.randint(1, 299),
            )
            for params in embedder_params
        ]
        # Ember embedder objects to be tested
        cls.ember_object_list = [
            emb.Ember(
                dataset=random_dataset_sample(cls.dataset, nrows=5),
                is_sentence_embedding=param,
            )
            for param in embedding_type_params
        ]

        # Group all different embedders into a list for testing
        cls.embedders_list = (
            cls.bert_object_list
            + cls.fasttext_object_list
            + cls.ember_object_list
        )

        # Create a list of embedder objects only for error raising tests
        cls.error_embedder_list = [
            emb.Bert(dataset=random_dataset_sample(cls.dataset, nrows=5)),
            emb.FastText(dataset=random_dataset_sample(cls.dataset, nrows=5)),
            emb.Ember(dataset=random_dataset_sample(cls.dataset, nrows=5)),
        ]

    def testDimensionOfEmbeddingsIsCorrect(self):
        """
        Test that the dimension of the embeddings is correct.

        Raises
        ------
        ValueError
            If the embedder object type (i.e. Bert, FastText, etc.) is
            not recognised.
        """
        dim_bert = 768
        dim_ft = 300
        dim_ember = 1024

        for embedder in self.embedders_list:
            with self.subTest(embedder=embedder):
                embedder.vectorise_dataset()
                vectorised_df = embedder.dataset
                vectors = vectorised_df["sentence_vectorised"]

                if embedder in self.bert_object_list:
                    dim = dim_bert
                elif embedder in self.fasttext_object_list:
                    dim = dim_ft
                elif embedder in self.ember_object_list:
                    dim = dim_ember
                else:
                    raise ValueError(
                        "Error: embedder type not recognised."
                        "Make sure this embedder is added to its appropriate embedder type list in setUpClass!"
                    )

                self.assertTrue(
                    check_dimension(
                        vectors, embedder.is_sentence_embedding, dim
                    )
                )

    def testFastTextReducesDimensionOfEmbeddingsCorrectly(self):
        """
        Test that FastText's inbuilt dimensionality reduction tool
        outputs embeddings of the desired target dimension dimension.
        """
        for embedder in self.fasttext_dim_object_list:
            with self.subTest(embedder=embedder):
                embedder.vectorise_dataset()
                vectorised_df = embedder.dataset
                vectors = vectorised_df["sentence_vectorised"]
                self.assertTrue(
                    check_dimension(
                        vectors, embedder.is_sentence_embedding, embedder.dim
                    )
                )

    def testSavedEmbeddingsCanBeReadCorrectly(self):
        """
        Tests that our vectorised sentences are saved correctly on the
        TSV files by ensuring that what we are reading from the file is
        indeed a sentence or word embedding vector.
        """
        for embedder in self.embedders_list:
            with self.subTest(embedder=embedder):
                filename = "test_file"
                file_path = os.path.join(current_path, filename)
                embedder.save_embedding_dataset(
                    path=current_path, filename=filename
                )
                file_path += ".pkl"
                saved_df = pd.read_pickle(file_path)

                if os.path.exists(file_path):
                    os.remove(file_path)

                vectors = saved_df["sentence_vectorised"].to_list()

                if embedder.is_sentence_embedding:
                    self.assertTrue(check_sentence_embedding_type(vectors))
                else:
                    self.assertTrue(check_word_embedding_type(vectors))

    def testSentenceEmbeddingsForAGivenSentenceAreDifferentForDifferentEmbeddingTypes(
        self,
    ):
        """
        Checks that different vectorisation mechanisms yield different
        embedding vectors.
        """
        for embedder1, embedder2 in combinations(self.embedders_list, 2):
            if (
                embedder1.is_sentence_embedding
                == embedder2.is_sentence_embedding
            ):
                with self.subTest(embedder1=embedder1, embedder2=embedder2):
                    embedding_list_1 = embedder1.dataset[
                        "sentence_vectorised"
                    ].to_numpy()
                    embedding_list_2 = embedder2.dataset[
                        "sentence_vectorised"
                    ].to_numpy()
                    self.assertFalse(
                        np.array_equal(embedding_list_1, embedding_list_2)
                    )

    def checkAddingEmbeddingsToDatasetRaisesRuntimeErrorIfEmbeddingsNotComputed(
        self,
    ):
        """
        Checks that if one calls the add_embeddings_to_dataset method
        without previously computing the embeddings by calling
        compute_embeddings, an attribute error is raised.
        """
        for embedder in self.error_embedder_list:
            with self.subTest(embedder=embedder):
                self.assertRaises(
                    RuntimeError,
                    embedder._add_embedding_to_dataset(embeddings=[]),
                )

    def checkAddingEmbeddingsToDatasetRaisesTypeErrorIfEmbeddingsNotList(
        self,
    ):
        """
        Checks that if one calls the add_embeddings_to_dataset method
        with an embeddings parameter that is not a list, it raises a
        TypeError.
        """
        invalid_types = [
            123,
            "string",
            (1, 2, 3),
            {"a": 1, "b": 2},
            True,
            None,
        ]
        for embedder in self.error_embedder_list:
            with self.subTest(embedder=embedder, invalid_type=invalid_type):
                for invalid_type in invalid_types:
                    self.assertRaises(
                        TypeError,
                        embedder._add_embedding_to_dataset(
                            embeddings=invalid_type
                        ),
                    )

    def checkSavingEmbeddingsDatasetRaisesAttributeErrorIfEmbeddingsNotAddedToDataset(
        self,
    ):
        """
        Checks that if one calls the save_embeddings_dataset method
        without previously adding the embeddings to the dataset by
        calling add_embeddings_to_dataset, an attribute error is raised.
        """

        for embedder in self.error_embedder_list:
            with self.subTest(embedder=embedder):
                self.assertRaises(TypeError, embedder.save_embedding_dataset())


if __name__ == "__main__":
    unittest.main()
