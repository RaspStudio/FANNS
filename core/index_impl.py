import random
from typing import Iterable

import faiss
import numpy as np

from core.interface import (AbstractData, AbstractDataSet,
                            AbstractEmbeddingModel, AbstractVectorStorage,
                            vector)

# ---------------------# Index Implementation #---------------------#


class FlatIndex[D: AbstractData](AbstractVectorStorage[D]):
    '''A storage for image embeddings using FAISS.'''

    def __init__(self, data: list[tuple[D, list[float]]]):
        '''Initialize the image vector storage from the image embeddings.'''
        embeddings, data_items_ = [], []
        for data_item, embedding in data:
            embeddings.append(embedding)
            data_items_.append(data_item)

        self.embeddings_ = np.array(embeddings).astype('float32')
        self.data_items_ = data_items_

        self.restore_embeddings = []
        self.restore_data_items = []
        if len(self.embeddings_) != len(self.data_items_):
            raise ValueError(
                'The number of embeddings and data items are different.')

        self.index_ = faiss.IndexFlatL2(self.embeddings_.shape[1])
        self.index_.add(self.embeddings_)

    def search(self, query: vector, k: int, with_vector: bool = False) -> list[tuple[float, D]] | list[tuple[float, D, vector]]:
        query_embedding = np.array([query]).astype(
            'float32')  # Convert to 2D numpy array
        distances, indices = self.index_.search(query_embedding, k)

        # Retrieve the nearest embeddings and their corresponding data items
        results: list[tuple[float, D]] = []
        for i in range(k):
            idx: int = indices[0][i]
            distance: float = distances[0][i]
            data_item = self.data_items_[idx]
            if with_vector:
                results.append((distance, data_item, self.embeddings_[idx]))
            else:
                results.append((distance, data_item))

        return results

    def get_vectors(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            idx = self.data_items_.index(item)
            ret.append(self.embeddings_[idx])
        return ret

    def restore(self, sended_data: Iterable[tuple[D, vector]]):
        for data, vector in sended_data:
            self.restore_data_items.append(data)
            self.restore_embeddings.append(vector)

    def get_from_restore(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            idx = self.restore_data_items.index(item)
            ret.append(self.restore_embeddings[idx])
        return ret

    @staticmethod
    def distance(query: vector, target: vector) -> float:
        '''Return the L2 distance between the query and target vectors.'''
        qarr = np.array([query]).astype('float32')
        tarr = np.array([target]).astype('float32')
        qptr = faiss.swig_ptr(qarr)
        tptr = faiss.swig_ptr(tarr)
        return faiss.fvec_L2sqr(qptr, tptr, len(query))

    @staticmethod
    def build_from_dataset(dataset: AbstractDataSet[D], model: AbstractEmbeddingModel[D], echo: bool = False) -> 'FlatIndex[D]':
        import os
        import pickle

        # Prepare the data pairs
        data_pairs: list[tuple[D, vector]] = []
        for i, data in enumerate(dataset):
            embpkl_path = data.label() + f".{model.__class__.__name__}.emb"
            if os.path.exists(embpkl_path):
                data_emb = pickle.load(open(embpkl_path, "rb"))
            else:
                data_emb = model.embed([data])[0]
                pickle.dump(data_emb, open(embpkl_path, "wb"))
            data_pairs.append((data, data_emb))
            if echo:
                print(f"Embedded {i+1}/{len(dataset)}...", end="\r")
        # Create the index
        index = FlatIndex(data_pairs)
        if echo:
            print(f"Finished Building FlatIndex with {len(dataset)} Objects")
        return index


class HNSWIndex[D: AbstractData](AbstractVectorStorage[D]):
    '''A storage for image embeddings using FAISS.'''

    def __init__(self, data: list[tuple[D, list[float]]]):
        '''Initialize the image vector storage from the image embeddings.'''
        embeddings, data_items_ = [], []
        for data_item, embedding in data:
            embeddings.append(embedding)
            data_items_.append(data_item)

        self.embeddings_ = np.array(embeddings).astype('float32')
        self.data_items_ = data_items_

        if len(self.embeddings_) != len(self.data_items_):
            raise ValueError(
                'The number of embeddings and data items are different.')
        print("start create HNSWIndex")
        self.index_ = faiss.IndexHNSWFlat(self.embeddings_.shape[1], 32)
        self.index_.add(self.embeddings_)
        self.index_.hnsw.efSearch = 64
        print("finish create")

    def search(self, query: vector, k: int, with_vector: bool = False) -> list[tuple[float, D]]:
        query_embedding = np.array([query]).astype(
            'float32')  # Convert to 2D numpy array
        distances, indices = self.index_.search(query_embedding, k)

        # Retrieve the nearest embeddings and their corresponding data items
        results: list[tuple[float, D]] = []
        for i in range(k):
            idx: int = indices[0][i]
            distance: float = distances[0][i]
            data_item = self.data_items_[idx]
            results.append((distance, data_item))

        return results

    def restore(self, sended_data: Iterable[tuple[D, vector]]):
        return

    def get_from_restore(self, items: Iterable[D]) -> list[vector]:
        return []

    def get_vectors(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            idx = self.data_items_.index(item)
            ret.append(self.embeddings_[idx])
        return ret

    @staticmethod
    def distance(query: vector, target: vector) -> float:
        '''Return the L2 distance between the query and target vectors.'''
        qarr = np.array([query]).astype('float32')
        tarr = np.array([target]).astype('float32')
        qptr = faiss.swig_ptr(qarr)
        tptr = faiss.swig_ptr(tarr)
        return faiss.fvec_L2sqr(qptr, tptr, len(query))

    @staticmethod
    def build_from_dataset(dataset: AbstractDataSet[D], model: AbstractEmbeddingModel[D], echo: bool = False) -> 'HNSWIndex[D]':
        import os
        import pickle

        # Prepare the data pairs
        data_pairs: list[tuple[D, vector]] = []
        for i, data in enumerate(dataset):
            embpkl_path = data.label() + f".{model.__class__.__name__}.emb"
            if os.path.exists(embpkl_path):
                data_emb = pickle.load(open(embpkl_path, "rb"))
            else:
                data_emb = model.embed([data])[0]
                pickle.dump(data_emb, open(embpkl_path, "wb"))
            data_pairs.append((data, data_emb))
            if echo:
                print(f"Embedded {i+1}/{len(dataset)}...", end="\r")
        # Create the index
        index = HNSWIndex(data_pairs)
        if echo:
            print(f"Finished Building HNSWIndex with {len(dataset)} Objects")
        return index


class IVFPQIndex[D: AbstractData](AbstractVectorStorage[D]):
    '''A storage for image embeddings using FAISS.'''

    def __init__(self, data: list[tuple[D, list[float]]]):
        '''Initialize the image vector storage from the image embeddings.'''
        embeddings, data_items_ = [], []
        for data_item, embedding in data:
            embeddings.append(embedding)
            data_items_.append(data_item)

        data_pairs = list(zip(data_items_, embeddings))
        random.shuffle(data_pairs)
        data_items_, embeddings = zip(*data_pairs)
        self.embeddings_ = np.array(embeddings).astype('float32')
        self.data_items_ = data_items_

        if len(self.embeddings_) != len(self.data_items_):
            raise ValueError(
                'The number of embeddings and data items are different.')
        print("start create IVFPQIndex")

        nlist = int(4 * len(self.embeddings_) ** 0.5)
        m = 96
        nbits = 8
        quantizer = faiss.IndexFlatL2(self.embeddings_.shape[1])
        self.index_ = faiss.IndexIVFPQ(
            quantizer, self.embeddings_.shape[1], nlist, m, nbits)

        self.index_.train(self.embeddings_[:int(
            len(self.embeddings_) ** 0.5)*256])
        self.index_.nprobe = 1500
        self.index_.add(self.embeddings_)
        print("finish create")

    def search(self, query: vector, k: int, with_vector: bool = False) -> list[tuple[float, D]]:
        query_embedding = np.array([query]).astype(
            'float32')  # Convert to 2D numpy array
        distances, indices = self.index_.search(query_embedding, k)

        # Retrieve the nearest embeddings and their corresponding data items
        results: list[tuple[float, D]] = []
        for i in range(k):
            idx: int = indices[0][i]
            distance: float = distances[0][i]
            data_item = self.data_items_[idx]
            results.append((distance, data_item))

        return results

    def restore(self, sended_data: Iterable[tuple[D, vector]]):
        return

    def get_from_restore(self, items: Iterable[D]) -> list[vector]:
        return []

    def get_vectors(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            idx = self.data_items_.index(item)
            ret.append(self.embeddings_[idx])
        return ret

    @staticmethod
    def distance(query: vector, target: vector) -> float:
        '''Return the L2 distance between the query and target vectors.'''
        qarr = np.array([query]).astype('float32')
        tarr = np.array([target]).astype('float32')
        qptr = faiss.swig_ptr(qarr)
        tptr = faiss.swig_ptr(tarr)
        return faiss.fvec_L2sqr(qptr, tptr, len(query))

    @staticmethod
    def build_from_dataset(dataset: AbstractDataSet[D], model: AbstractEmbeddingModel[D], echo: bool = False) -> 'IVFPQIndex[D]':
        import os
        import pickle

        # Prepare the data pairs
        data_pairs: list[tuple[D, vector]] = []
        for i, data in enumerate(dataset):
            embpkl_path = data.label() + f".{model.__class__.__name__}.emb"
            if os.path.exists(embpkl_path):
                data_emb = pickle.load(open(embpkl_path, "rb"))
            else:
                data_emb = model.embed([data])[0]
                pickle.dump(data_emb, open(embpkl_path, "wb"))
            data_pairs.append((data, data_emb))
            if echo:
                print(f"Embedded {i+1}/{len(dataset)}...", end="\r")
        # Create the index
        index = IVFPQIndex(data_pairs)
        if echo:
            print(f"Finished Building IVFPQIndex with {len(dataset)} Objects")
        return index


class IVFFlatIndex[D: AbstractData](AbstractVectorStorage[D]):
    '''A storage for image embeddings using FAISS.'''

    def __init__(self, data: list[tuple[D, list[float]]]):
        '''Initialize the image vector storage from the image embeddings.'''
        embeddings, data_items_ = [], []
        for data_item, embedding in data:
            embeddings.append(embedding)
            data_items_.append(data_item)

        data_pairs = list(zip(data_items_, embeddings))
        random.shuffle(data_pairs)
        data_items_, embeddings = zip(*data_pairs)
        self.embeddings_ = np.array(embeddings).astype('float32')
        self.data_items_ = data_items_

        if len(self.embeddings_) != len(self.data_items_):
            raise ValueError(
                'The number of embeddings and data items are different.')
        print("start create IVFIndex")

        nlist = int(4 * len(self.embeddings_) ** 0.5)
        quantizer = faiss.IndexFlatL2(self.embeddings_.shape[1])

        self.index_ = faiss.IndexIVFFlat(
            quantizer, self.embeddings_.shape[1], nlist, faiss.METRIC_L2)

        self.index_.train(self.embeddings_[:int(
            len(self.embeddings_) ** 0.5)*256])
        self.index_.nprobe = 128
        self.index_.add(self.embeddings_)
        print("finish create")

    def search(self, query: vector, k: int, with_vector: bool = False) -> list[tuple[float, D]]:
        query_embedding = np.array([query]).astype(
            'float32')  # Convert to 2D numpy array
        distances, indices = self.index_.search(query_embedding, k)

        # Retrieve the nearest embeddings and their corresponding data items
        results: list[tuple[float, D]] = []
        for i in range(k):
            idx: int = indices[0][i]
            distance: float = distances[0][i]
            data_item = self.data_items_[idx]
            results.append((distance, data_item))

        return results

    def restore(self, sended_data: Iterable[tuple[D, vector]]):
        return

    def get_from_restore(self, items: Iterable[D]) -> list[vector]:
        return []

    def get_vectors(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            idx = self.data_items_.index(item)
            ret.append(self.embeddings_[idx])
        return ret

    @staticmethod
    def distance(query: vector, target: vector) -> float:
        '''Return the L2 distance between the query and target vectors.'''
        qarr = np.array([query]).astype('float32')
        tarr = np.array([target]).astype('float32')
        qptr = faiss.swig_ptr(qarr)
        tptr = faiss.swig_ptr(tarr)
        return faiss.fvec_L2sqr(qptr, tptr, len(query))

    @staticmethod
    def build_from_dataset(dataset: AbstractDataSet[D], model: AbstractEmbeddingModel[D], echo: bool = False) -> 'IVFFlatIndex[D]':
        import os
        import pickle

        # Prepare the data pairs
        data_pairs: list[tuple[D, vector]] = []
        for i, data in enumerate(dataset):
            embpkl_path = data.label() + f".{model.__class__.__name__}.emb"
            if os.path.exists(embpkl_path):
                data_emb = pickle.load(open(embpkl_path, "rb"))
            else:
                data_emb = model.embed([data])[0]
                pickle.dump(data_emb, open(embpkl_path, "wb"))
            data_pairs.append((data, data_emb))
            if echo:
                print(f"Embedded {i+1}/{len(dataset)}...", end="\r")
        # Create the index
        index = IVFFlatIndex(data_pairs)
        if echo:
            print(
                f"Finished Building IVFFlatIndex with {len(dataset)} Objects")
        return index
