"""
DimReduction
============
Module containing the base class for performing dimensionality reduction.

"""

import os

import numpy as np
import pandas as pd
import sklearn.decomposition as skd
import sklearn.manifold as skm
import umap

from abc import ABC, abstractmethod
from typing import Union


class DimReduction(ABC):
    """
    Base class for dimensionality reduction of
    vectors representing sentences.
    """

    def __init__(self, dim_out: int) -> None:
        """
        Initialise the dimensionality reduction class.

        Parameters
        ----------
        dim_out : int
            Desired output dimension of the vectors.
        """
        self.dim_out = dim_out

    def _pre_fit(
        self, data_to_fit: Union[pd.DataFrame, list[pd.DataFrame]]
    ) -> np.ndarray:
        """
        Pre-processes data to prepare it to be fed into the fit method.

        Parameters
        ----------
        data_to_fit: Union[pd.DataFrame, list[pd.DataFrame]]
            The dataset(s) containing the embeddings we wish to fit our
            dimensionality reduction object with.

        Returns
        -------
        np.ndarray
            A stacked array containing the embeddings to be used to fit
            the dimensionality reduction object.
        """
        if isinstance(data_to_fit, pd.DataFrame):
            try:
                embeddings_to_fit = data_to_fit[
                    "sentence_vectorised"
                ].to_numpy()
            except KeyError:
                raise KeyError("Sentence vectors not present in the fit data.")
        elif isinstance(data_to_fit, list):
            embeddings_to_fit = []
            for df in data_to_fit:
                try:
                    df_embeddings = df["sentence_vectorised"].to_list()
                except KeyError:
                    raise KeyError(
                        "Sentence vectors not present in the fit data."
                    )
                embeddings_to_fit.extend(df_embeddings)
            embeddings_to_fit = np.array(embeddings_to_fit)
        else:
            raise ValueError(
                "Invalid input type. Expecting pd.DataFrame or list[pd.DataFrame]."
            )
        embeddings_to_fit = np.stack(embeddings_to_fit)

        return embeddings_to_fit

    @abstractmethod
    def fit(
        self, data_to_fit: Union[pd.DataFrame, list[pd.DataFrame]]
    ) -> None:
        """
        Fits the dimensionality reduction mechanism to the data
        contained in the dataset and any additional data.

        Parameters
        ----------
        data_to_fit: Union[pd.DataFrame, list[pd.DataFrame]]
            Dataframe(s) containg the embedding vectors with which to wish
            to fit our dimensionality reduction object.
        """
        raise NotImplementedError("Subclasses must implement the fit method.")

    def _pre_reduce_dimension(
        self, data_to_reduce: pd.DataFrame
    ) -> np.ndarray:
        """
        Pre-processes data to prepare it to be fed into the
        reduce_dimension method.

        Parameters
        ----------
        data_to_reduce: pd.DataFrame
            The dataset containing the embeddings we wish to reduce.

        Returns
        -------
        np.ndarray:
            A stacked array containing the embeddings we wish to reduce.
        """
        try:
            embeddings_to_reduce = data_to_reduce[
                "sentence_vectorised"
            ].to_numpy()
        except KeyError:
            raise KeyError(
                "Sentence vectors not present in the data to be reduced."
            )
        embeddings_to_reduce = np.stack(embeddings_to_reduce)

        return embeddings_to_reduce

    @abstractmethod
    def reduce_dimension(self, data_to_reduce: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces embedding vectors in a given dataset to the desired
        output dimension.

        Parameters
        ----------
        data_to_reduce: pd.DataFrame
            Dataframe containing the embedding vectors that we wish to
            perform dimensionality reduction on.

        Returns
        -------
        pd.DataFrame
            A dataframe identical to the input dataframe but with an
            additional column contatining the reduced embedding vectors.
        """
        raise NotImplementedError(
            "Suclasses must implement the reduce_dimension method."
        )

    def _post_reduce_dimension(
        self,
        sentence_vectors_reduced: np.ndarray,
        data_to_reduce: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Takes the reduced embeddings and stores them into a new column
        in the dataset.

        Parameters
        ----------
        sentence_vectors_reduced: np.ndarray
            An array containing the reduced embeddings for our dataset.

        data_to_reduce: pd.DataFrame
            The dataset from which the embeddings where extracted, and
            to which we wish to add the reduced embeddings.

        Returns
        -------
        pd.DataFrame
            A dataframe consisting of the original data plus a new
            column with the corresponding reduced embeddings.

        """
        reduced_vectors_list = sentence_vectors_reduced.tolist()
        reduced_vectors_list = [
            np.array(vec, dtype=np.float32) for vec in reduced_vectors_list
        ]
        reduced_data = data_to_reduce
        reduced_data["reduced_sentence_vector"] = reduced_vectors_list

        return reduced_data

    def save_dataset(
        self, reduced_dataset: pd.DataFrame, path: str, filename: str
    ) -> None:
        """
        Save the reduced dataset in a given path as a pickle file.

        Parameters
        ----------
        reduced_dataset : pd.DataFrame
            The dataframe with the reduced embedding vectors that we
            wish to save as a file.
        path : str
            Path where to store the dataset.
        filename : str
            Name of the file to save to (should not contain extensions).
        """
        reduced_dataset.to_pickle(os.path.join(path, filename) + ".pkl")


class PCA(DimReduction):
    """
    Class for principal component analysis implementation.
    """

    def __init__(self, dim_out: int, **kwargs) -> None:
        """
        Initialise the PCA dimensionality reduction class.

        Parameters
        ----------
        dim_out : int
            Desired output dimension of the vectors.
        **kwargs
            Arguments passed to the sklearn.decomposition.PCA object
            that can be found in
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html.
        """
        super().__init__(dim_out=dim_out)
        self.pca_sk = skd.PCA(n_components=self.dim_out, **kwargs)

    def fit(
        self, data_to_fit: Union[pd.DataFrame, list[pd.DataFrame]]
    ) -> None:
        """
        Fits the PCA object on a given set(s) of input embeddings.
        """
        embeddings_to_fit = super()._pre_fit(data_to_fit)
        self.pca_sk.fit(embeddings_to_fit)

    def reduce_dimension(self, data_to_reduce: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces the embeddings in the input dataframe with the fitted
        PCA.
        """
        embeddings_to_reduce = super()._pre_reduce_dimension(data_to_reduce)
        sentence_vectors_reduced = self.pca_sk.transform(embeddings_to_reduce)
        reduced_data = super()._post_reduce_dimension(
            sentence_vectors_reduced, data_to_reduce
        )

        return reduced_data


class ICA(DimReduction):
    """
    Class for the independent component analysis implementation.
    """

    def __init__(self, dim_out: int, **kwargs) -> None:
        """
        Initialise the ICA dimensionality reduction class.

        Parameters
        ----------
        dim_out : int
            Desired output dimension of the vectors.
        **kwargs
            Arguments passed to the sklearn.decomposition.FastICA object.
            They can be found in
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html.
        """
        super().__init__(dim_out=dim_out)
        self.ica_sk = skd.FastICA(n_components=self.dim_out, **kwargs)

    def fit(
        self, data_to_fit: Union[pd.DataFrame, list[pd.DataFrame]]
    ) -> None:
        """
        Fits the ICA object on a given set(s) of input embeddings.
        """
        embeddings_to_fit = super()._pre_fit(data_to_fit)
        self.ica_sk.fit(embeddings_to_fit)

    def reduce_dimension(self, data_to_reduce: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces the embeddings in the input dataframe with the fitted
        ICA.
        """
        embeddings_to_reduce = super()._pre_reduce_dimension(data_to_reduce)
        sentence_vectors_reduced = self.ica_sk.transform(embeddings_to_reduce)
        reduced_data = super()._post_reduce_dimension(
            sentence_vectors_reduced, data_to_reduce
        )

        return reduced_data


class TSVD(DimReduction):
    """
    Class for truncated SVD dimensionality reduction.
    """

    def __init__(self, dim_out: int, **kwargs) -> None:
        """
        Initialise the TSVD dimensionality reduction class.

        Parameters
        ----------
        dim_out : int
            Desired output dimension of the vectors.
        **kwargs
            Arguments passed to the sklearn.decomposition.TruncatedSVD object.
            They can be found in
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html.
        """
        super().__init__(dim_out=dim_out)
        self.tsvd_sk = skd.TruncatedSVD(n_components=self.dim_out, **kwargs)

    def fit(
        self, data_to_fit: Union[pd.DataFrame, list[pd.DataFrame]]
    ) -> None:
        """
        Fits the TSVD object on a given set(s) of input embeddings.
        """
        embeddings_to_fit = super()._pre_fit(data_to_fit)
        self.tsvd_sk.fit(embeddings_to_fit)

    def reduce_dimension(self, data_to_reduce: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces the embeddings in the input dataframe with the fitted
        TSVD.
        """
        embeddings_to_reduce = super()._pre_reduce_dimension(data_to_reduce)
        sentence_vectors_reduced = self.tsvd_sk.transform(embeddings_to_reduce)
        reduced_data = super()._post_reduce_dimension(
            sentence_vectors_reduced, data_to_reduce
        )

        return reduced_data


class UMAP(DimReduction):
    """
    Class for UMAP dimensionality reduction.
    """

    def __init__(self, dim_out: int, **kwargs) -> None:
        """
        Initialise the UMAP dimensionality reduction class.

        Parameters
        ----------
        dim_out : int
            Desired output dimension of the vectors.
        **kwargs
            Arguments passed to the sklearn.decomposition.UMAP object.
            They can be found in
            https://umap-learn.readthedocs.io/en/latest/parameters.html.
        """
        super().__init__(dim_out=dim_out)
        self.umap_sk = umap.UMAP(n_components=self.dim_out, **kwargs)

    def fit(
        self, data_to_fit: Union[pd.DataFrame, list[pd.DataFrame]]
    ) -> None:
        """
        Fits the UMAP object on a given set(s) of input embeddings.
        """
        embeddings_to_fit = super()._pre_fit(data_to_fit)
        self.umap_sk.fit(embeddings_to_fit)

    def reduce_dimension(self, data_to_reduce: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces the embeddings in the input dataframe with the fitted
        ICA.
        """
        embeddings_to_reduce = super()._pre_reduce_dimension(data_to_reduce)
        sentence_vectors_reduced = self.umap_sk.transform(embeddings_to_reduce)
        reduced_data = super()._post_reduce_dimension(
            sentence_vectors_reduced, data_to_reduce
        )

        return reduced_data


# Dropping TSNE for now as it does not have a transform method
'''
class TSNE(DimReduction):
    """
    Class for truncated TSNE dimensionality reduction.
    """

    def __init__(self, dim_out: int, **kwargs) -> None:
        """
        Initialise the TSNE dimensionality reduction class.

        Parameters
        ----------
        dim_out : int
            Desired output dimension of the vectors.
        **kwargs
            Arguments passed to the sklearn.decomposition.TSNE object.
            They can be found in
            https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html.
        """
        super().__init__(dim_out=dim_out)
        self.tsne_sk = skm.TSNE(n_components=self.dim_out, **kwargs)

    def fit(
        self, data_to_fit: Union[pd.DataFrame, list[pd.DataFrame]]
    ) -> None:
        """
        Fits the TSNE object on a given set(s) of input embeddings.
        """
        if isinstance(data_to_fit, pd.DataFrame):
            embeddings_to_fit = data_to_fit["sentence_vectorised"].to_numpy()
        elif isinstance(data_to_fit, list):
            embeddings_to_fit = []
            for df in data_to_fit:
                df_embeddings = df["sentence_vectorised"].to_list()
                embeddings_to_fit.extend(df_embeddings)
            embeddings_to_fit = np.array(embeddings_to_fit)
        else:
            raise ValueError(
                "Invalid input type. Expecting pd.DataFrame or list[pd.DataFrame]."
            )
        embeddings_to_fit = np.stack(embeddings_to_fit)
        self.tsne_sk.fit(embeddings_to_fit)

    def reduce_dimension(self, data_to_reduce: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces the embeddings in the input dataframe with the fitted
        TSNE.
        """
        reduced_data = data_to_reduce
        embeddings_to_reduce = data_to_reduce["sentence_vectorised"].to_numpy()
        embeddings_to_reduce = np.stack(embeddings_to_reduce)
        sentence_vectors_reduced = self.tsne_sk.transform(embeddings_to_reduce)
        reduced_data["reduced_sentence_vector"] = (
            sentence_vectors_reduced.tolist()
        )
        return reduced_data
'''
