"""
DimReduction
============
Module containing the base class for performing dimensionality reduction.

"""

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
        for df in data_to_fit:
            try:
                self.embeddings_to_fit = df["sentence_vectorised"].to_list()
            except KeyError:
                raise KeyError("Sentence vectors not present in the fit data.")

        raise NotImplementedError("Suclasses must implement the fit method.")

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
        try:
            self.embeddings_to_reduce = data_to_reduce[
                "sentence_vectorised"
            ].to_list()
        except KeyError:
            raise KeyError(
                "Sentence vectors not present in the data to be reduced."
            )
        raise NotImplementedError(
            "Suclasses must implement the reduce_dimension method."
        )

    def save_dataset(self, filename: str, dataset_path: str) -> None:
        """
        Save the reduced dataset in a given path.

        Parameters
        ----------
        filename : str
            Name of the file to save to.
        dataset_path : str
            Path where to store the dataset.
        """
        filepath = f"{dataset_path}{filename}.tsv"
        self.dataset.to_csv(filepath, sep="\t", index=False)


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
            Arguments passed to the sklearn.decomposition.PCA object.
            They can be found in
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
        self.pca_sk.fit(embeddings_to_fit)

    def reduce_dimension(self, data_to_reduce: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces the embeddings in the input dataframe with the fitted
        PCA.
        """
        reduced_data = data_to_reduce.copy()
        embeddings_to_reduce = data_to_reduce["sentence_vectorised"].to_numpy()
        sentence_vectors_reduced = self.pca_sk.transform(embeddings_to_reduce)
        reduced_data["reduced_sentence_vector"] = sentence_vectors_reduced
        return reduced_data


class ICA(DimReduction):
    """
    Class for the independent component analysis implementation.
    """

    def __init__(self, dataset: pd.DataFrame, dim_out: int, **kwargs) -> None:
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
        self.ica_sk.fit(embeddings_to_fit)

    def reduce_dimension(self, data_to_reduce: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces the embeddings in the input dataframe with the fitted
        ICA.
        """
        reduced_data = data_to_reduce.copy()
        embeddings_to_reduce = data_to_reduce["sentence_vectorised"].to_numpy()
        sentence_vectors_reduced = self.ica_sk.transform(embeddings_to_reduce)
        reduced_data["reduced_sentence_vector"] = sentence_vectors_reduced
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
        self.tsvd_sk.fit(embeddings_to_fit)

    def reduce_dimension(self, data_to_reduce: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces the embeddings in the input dataframe with the fitted
        ICA.
        """
        reduced_data = data_to_reduce.copy()
        embeddings_to_reduce = data_to_reduce["sentence_vectorised"].to_numpy()
        sentence_vectors_reduced = self.tsvd_sk.transform(embeddings_to_reduce)
        reduced_data["reduced_sentence_vector"] = sentence_vectors_reduced
        return reduced_data


class UMAP(DimReduction):
    """
    Class for UMAP dimensionality reduction.
    """

    def __init__(self, dataset: pd.DataFrame, dim_out: int, **kwargs) -> None:
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
        self.umap_sk.fit(embeddings_to_fit)

    def reduce_dimension(self, data_to_reduce: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces the embeddings in the input dataframe with the fitted
        ICA.
        """
        reduced_data = data_to_reduce.copy()
        embeddings_to_reduce = data_to_reduce["sentence_vectorised"].to_numpy()
        sentence_vectors_reduced = self.umap_sk.transform(embeddings_to_reduce)
        reduced_data["reduced_sentence_vector"] = sentence_vectors_reduced
        return reduced_data


class TSNE(DimReduction):
    """
    Class for truncated TSNE dimensionality reduction.
    """

    def __init__(self, dataset: pd.DataFrame, dim_out: int, **kwargs) -> None:
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
        super().__init__(dataset=dataset, dim_out=dim_out)
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
        self.tsne_sk.fit(embeddings_to_fit)

    def reduce_dimension(self, data_to_reduce: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces the embeddings in the input dataframe with the fitted
        TSNE.
        """
        reduced_data = data_to_reduce.copy()
        embeddings_to_reduce = data_to_reduce["sentence_vectorised"].to_numpy()
        sentence_vectors_reduced = self.tsne_sk.transform(embeddings_to_reduce)
        reduced_data["reduced_sentence_vector"] = sentence_vectors_reduced
        return reduced_data
