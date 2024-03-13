"""
Embedder
============
Module containing the base class for transforming sentences in a dataset into embeddings.

"""

from abc import ABC, abstractmethod

import fasttext as ft
import fasttext.util as ftu
import numpy as np
import os
import pandas as pd
import torch

from pandas.core.api import DataFrame as DataFrame
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from typing import Union


class Embedder(ABC):
    """
    Base class for the generation of embeddings for the sentences
    comprising a dataset.

    Attributes
    ----------
    dataset: DataFrame
        The natural language dataset we wish to compute the vector
        embeddings for.
    is_sentence_embedding: bool
        Indicates whether we want to compute sentence embeddings (True)
        or word embedding (False) vectors (default: True).
    embeddings_computed: bool
        Tracks whether the user has computed the embeddings or not
        (default: False).

    Methods
    -------
    compute_embeddings:
        Computes embedding vectors for the sentences in the dataset.
    add_embeddings_to_dataset:
        Adds the vector embeddings to a new column 'sentence_vectorised'
        in the dataset.
    save_embedding_dataset:
        Writes the dataset containing the embeddings to a TSV file.
    """

    def __init__(
        self, dataset: DataFrame, is_sentence_embedding: bool = True
    ) -> None:
        """
        Initialises the embedder class.

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataset. Each row of the dataset correspods to a
            sentence.
        is_sentence_embedding : bool
            A bool controllig whether we want to produce a sentence
            embedding or word embedding vector for each sentence in the
            dataset. True is for sentence embedding, False is for word
            embedding.
        """
        self.dataset = dataset
        self.is_sentence_embedding = is_sentence_embedding
        self.embeddings_computed = False

    @abstractmethod
    def compute_embeddings(
        self,
    ) -> Union[list[list[float]], list[list[list[float]]]]:
        """
        Computes vector embeddings for sentences in a dataset.

        Returns
        -------
        Union[list[list[float]], list[list[list[float]]]]
            A list containing the vectorised representation of the
            sentences in the dataset. A list of floats in the case of a
            sentence embedding, and a list of float lists in the case of
            word embeddings.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement compute_embeddings method."
        )

    @abstractmethod
    def add_embeddings_to_dataset(
        self, embeddings: Union[list[list[float]], list[list[list[float]]]]
    ) -> None:
        """
        Adds the calculated sentence vectors to a new column in our
        dataset.

        Parameters
        ----------
        embeddings: Union[list[list[float]], list[list[list[float]]]]
            The embeddings to be added to the dataset.

        Raises
        ------
        RuntimeError
            If the embeddings have not been previously computed by
            calling compute_embeddings.
        TypeError
            Something other than a list is passed to the embeddings
            parameter.
        NotImplementedError
            If the method has not been implemented by subclasses.
        """
        if not self.embeddings_computed:
            raise RuntimeError(
                "compute_embeddings must be called before add_embeddings_to_dataset"
            )
        if not isinstance(embeddings, list):
            raise TypeError("embeddings must be of type list")
        raise NotImplementedError(
            "Subclasses must implement add_embedding_to_dataset method."
        )

    @abstractmethod
    def save_embedding_dataset(self, path: str, filename: str) -> None:
        """
        Saves the dataset containing the embeddings as a TSV file.

        Parameters
        ----------
        path : str
            The path to the location where the user wishes to save the
            generated dataset containing the embeddings (it should not
            include a / at the end).
        filename: str
            The name with which the generated dataset containing the
            embeddings will be saved (it should not include .csv or any
            other file extension).

        Raises
        ------
        ValueError
            If the dataset has not been vectorised. You must call
            compute_embeddings before attempting to save the dataset.
        NotImplementedError
            If the method has not been implemented by subclasses.
        """
        try:
            _ = self.dataset["sentence_vectorised"].tolist()
        except KeyError:
            raise ValueError(
                "This dataset has not been vectorised. "
                "You must call compute_embeddings to create the embeddings and then add them "
                "to the dataset with add_embedding_dataset before attempting to save the dataset."
            )
        raise NotImplementedError(
            "Subclasses must implement save_embedding_dataset method."
        )


class Bert(Embedder):
    """
    Class for generating BERT embeddings.
    """

    def __init__(
        self,
        dataset: DataFrame,
        is_sentence_embedding: bool = True,
        cased: bool = False,
        **kwargs
    ) -> None:
        """
        Initialises the BERT embedder class

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataset. Each row of the dataset correspods to a
            sentence.
        is_sentence_embedding : bool
            States whether we want to produce a sentence embedding or
            word embedding vector for each sentence in the dataset. True
            is for sentence embedding, False is for word embedding.
        cased: bool
            States whether we want to work with BERT's cased or uncased
            pretrained base model. True is for cased, False is for
            uncased.
        **kwargs
            Additional arguments to be passed to the BertTokenizer
            object. These can be found in
            https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.
        """
        super().__init__(dataset, is_sentence_embedding)
        self.cased = cased
        self.kwargs = kwargs

    def compute_embeddings(
        self, **kwargs
    ) -> Union[list[list[float]], list[list[list[float]]]]:
        """
        Creates BERT embeddings for sentences in a dataset and adds them
        to a new column in the dataset.

        Parameters
        ----------
        **kwargs
            Additional arguments to be passed to the BertTokenizer
            object. These can be found in
            https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.

        Returns
        -------
        vectorised_sentence_list: list
            A list containing the vector embeddings corresponding to the
            sentences in the dataset.
        """
        embeddings_df = self.dataset.copy()
        embeddings_df.columns = ["class", "sentence", "sentence_structure"]

        model = "bert-base-cased" if self.cased else "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model, **kwargs)
        bert_model = BertModel.from_pretrained(model)

        for param in bert_model.parameters():
            param.requires_grad = False
        bert_model.eval()

        if self.is_sentence_embedding:
            vectorised_sentence_list = []
            for sentence in embeddings_df.sentence.values:
                inputs = tokenizer.encode_plus(sentence).input_ids
                inputs = torch.LongTensor(inputs)

                with torch.no_grad():
                    sentence_embedding = (
                        bert_model(inputs.unsqueeze(0))[1]
                        .squeeze(0)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                vectorised_sentence_list.append(
                    [tensor.item() for tensor in sentence_embedding]
                )

        else:
            vectorised_sentence_list = []
            for sentence in embeddings_df.sentence.values:
                # List storing the word embeddings of each word in a sentence
                sentence_word_embeddings = []
                for word in sentence.split():
                    inputs = tokenizer.encode_plus(word).input_ids
                    inputs = torch.LongTensor(inputs)

                    with torch.no_grad():
                        word_embedding = (
                            bert_model(inputs.unsqueeze(0))[1]
                            .squeeze(0)
                            .cpu()
                            .detach()
                            .numpy()
                        )

                    sentence_word_embeddings.append(word_embedding.tolist())
                vectorised_sentence_list.append(sentence_word_embeddings)

        self.embeddings_computed = True

        return vectorised_sentence_list

    def add_embeddings_to_dataset(
        self, embeddings: Union[list[list[float]], list[list[list[float]]]]
    ) -> None:
        """
        Adds the calculated BERT embeddings to a new column in our
        dataset.

        Parameters
        ----------
        embeddings: list
            A list of the embeddings corresponding to the sentences in
            the dataset.
        """
        self.dataset["sentence_vectorised"] = embeddings

    def save_embedding_dataset(self, path: str, filename: str) -> None:
        """
        Saves the dataset containing the BERT embeddings as a TSV file.

        Parameters
        ----------
        path : str
            The path to the location where the user wishes to save the
            generated dataset containing the embeddings.
        filename: str
            The name with which the generated dataset containing the
            embeddings will be saved (it should not include .csv or any
            other file extension).
        """
        self.dataset.to_csv(
            os.path.join(path, filename) + ".tsv", index=False, sep="\t"
        )


class FastText(Embedder):
    """
    Class for generating FastText embeddings.
    """

    def __init__(
        self,
        dataset: DataFrame,
        is_sentence_embedding: bool = True,
        dim: int = 300,
        cased: bool = True,
    ) -> None:
        """
        Initialises the FastText embedder class

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataset. Each row of the dataset correspods to a
            sentence.
        is_sentence_embedding : bool
            States whether we want to produce a sentence embedding or
            word embedding vector for each sentence in the dataset. True
            is for sentence embedding, False is for word embedding.
        dim : int
            The output dimension of FastText's word/sentence vectors.
            Default value is 300, but can be set to any value below
            that.
        cased: bool
            Controls whether we casefold our input sentences before
            vectorising them using FastText. FastText's pretrained model
            is case sensitive so casefolding will produce different
            embeddings.

        Raises
        ------
        ValueError
            If the dim attribute is initialised to an integer value
            larger than 300.
        """
        super().__init__(dataset, is_sentence_embedding)
        if dim <= 300:
            self.dim = dim
        else:
            raise ValueError("Error: dimension must be <= 300")
        self.cased = cased

    def compute_embeddings(
        self,
    ) -> Union[list[list[float]], list[list[list[float]]]]:
        """
        Creates FastText embeddings for sentences in a dataset and adds
        them to a new column in the dataset.

        Returns
        -------
        vectorised_sentence_list: list
            A list containing the vector embeddings corresponding to the
            sentences in the dataset.
        """
        ftu.download_model("en", if_exists="ignore")
        model = ft.load_model("cc.en.300.bin")
        if self.dim < 300:
            ftu.reduce_model(model, self.dim)
        embeddings_df = self.dataset.copy()
        embeddings_df.columns = ["class", "sentence", "sentence_structure"]

        if self.is_sentence_embedding:
            vectorised_sentence_list = []
            for sentence in embeddings_df.sentence.values:
                if not self.cased:
                    sentence = sentence.casefold()
                sentence_embedding = model.get_sentence_vector(
                    sentence
                ).tolist()
                vectorised_sentence_list.append(sentence_embedding)

        else:
            vectorised_sentence_list = []
            for sentence in embeddings_df.sentence.values:
                word_embeddings_list = []
                for word in sentence.split():
                    word_embedding = model.get_word_vector(word).tolist()
                    word_embeddings_list.append(word_embedding)
                vectorised_sentence_list.append(word_embeddings_list)

        self.embeddings_computed = True

        return vectorised_sentence_list

    def add_embeddings_to_dataset(
        self, embeddings: Union[list[list[float]], list[list[list[float]]]]
    ) -> None:
        """
        Adds the calculated FastText embeddings to a new column in our
        dataset.

        Parameters
        ----------
        embeddings: list
            A list of the embeddings corresponding to the sentences in
            the dataset.
        """
        self.dataset["sentence_vectorised"] = embeddings

    def save_embedding_dataset(self, path: str, filename: str) -> None:
        """
        Saves the dataset containing the FastText embeddings as a TSV
        file.

        Parameters
        ----------
        path : str
            The path to the location where the user wishes to save the
            generated dataset containing the embeddings.
         filename: str
            The name with which the generated dataset containing the
            embeddings will be saved (it should not include .csv or any
            other file extension).
        """
        self.dataset.to_csv(
            os.path.join(path, filename) + ".tsv", index=False, sep="\t"
        )


class Ember(Embedder):
    """
    Class for generating llmrails/ember-v1 embeddings.
    """

    def __init__(
        self,
        dataset: DataFrame,
        is_sentence_embedding: bool = True,
    ) -> None:
        """
        Initialises the Ember embedder class

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataset. Each row of the dataset correspods to a
            sentence.
        is_sentence_embedding : bool
            States whether we want to produce a sentence embedding or
            word embedding vector for each sentence in the dataset. True
            is for sentence embedding, False is for word embedding.
        """
        super().__init__(dataset, is_sentence_embedding)

    def compute_embeddings(
        self,
    ) -> Union[list[list[float]], list[list[list[float]]]]:
        """
        Creates ember-v1 embeddings for sentences in a dataset and adds them
        as a new column in the dataset.

        Returns
        -------
        vectorised_sentence_list: list
            A list containing the vector embeddings corresponding to the
            sentences in the dataset.
        """
        embeddings_df = self.dataset.copy()
        embeddings_df.columns = ["class", "sentence", "sentence_structure"]

        model = SentenceTransformer("llmrails/ember-v1")
        sentences = embeddings_df.sentence.values.tolist()

        if self.is_sentence_embedding:
            vectorised_sentence_list = model.encode(sentences).tolist()

        else:
            vectorised_sentence_list = []
            for sentence in sentences:
                word_embeddings_list = []
                for word in sentence.split():
                    word_embedding = model.encode(word).tolist()
                    word_embeddings_list.append(word_embedding)
                vectorised_sentence_list.append(word_embeddings_list)

        self.embeddings_computed = True

        return vectorised_sentence_list

    def add_embeddings_to_dataset(self, embeddings) -> None:
        """
        Adds the calculated ember-v1 embeddings to a new column in our
        dataset.

        Parameters
        ----------
        embeddings: list
            A list of the embeddings corresponding to the sentences in
            the dataset.
        """
        self.dataset["sentence_vectorised"] = embeddings

    def save_embedding_dataset(self, path: str, filename: str) -> None:
        """
        Saves the dataset containing the ember-v1 embeddings as a TSV
        file.

        Parameters
        ----------
        path : str
            The path to the location where the user wishes to save the
            generated dataset containing the embeddings.
         filename: str
            The name with which the generated dataset containing the
            embeddings will be saved (it should not include .csv or any
            other file extension).
        """
        self.dataset.to_csv(
            os.path.join(path, filename) + ".tsv", index=False, sep="\t"
        )
