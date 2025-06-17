# ---------------------# Types #---------------------#

# Vector
from typing import Iterable, Any
from abc import ABC, abstractmethod
vector = list[float]


# ---------------------# Interfaces #---------------------#


class AbstractData(ABC):
    '''An abstract class for data items, containing raw data and labels.'''

    @staticmethod
    @abstractmethod
    def from_bytes(data: bytes) -> 'AbstractData':
        '''Return a new data item from the serialized data.'''
        ...

    @abstractmethod
    def to_bytes(self) -> bytes:
        '''Return the serialized data item.'''
        ...

    @abstractmethod
    def label(self) -> str:
        '''Return the label of the data item.'''
        ...

    @abstractmethod
    def data(self) -> Any:
        '''Return the raw data of the data item.'''
        ...

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, AbstractData):
            return False
        return self.label() == __value.label()

    def __str__(self) -> str:
        return self.label()

    def __repr__(self) -> str:
        return self.label()


class AbstractDataSet[D: AbstractData](ABC):
    '''An abstract class for datasets.'''

    @abstractmethod
    def __getitem__(self, idx: int) -> D:
        '''Return the data item at the given index.'''
        ...

    @abstractmethod
    def __len__(self) -> int:
        '''Return the number of data items in the dataset.'''
        ...

    def __iter__(self):
        '''Return an iterator over the dataset.'''
        for i in range(self.__len__()):
            yield self.__getitem__(i)


class AbstractEmbeddingModel[D: AbstractData](ABC):
    '''An abstract class for embedding models.'''

    @abstractmethod
    def __init__(self, device: str):
        '''Initialize the model with the given device.'''
        ...

    @abstractmethod
    def to_device(self):
        '''Move the model to the given device.'''
        ...

    @abstractmethod
    def embed(self, data: Iterable[D]) -> list[vector]:
        '''Return the embedding of the given data item.'''
        ...


class AbstractVectorStorage[D: AbstractData](ABC):
    '''An abstract class for vector storage'''

    @abstractmethod
    def __init__(self, data: Iterable[tuple[D, vector]]):
        '''Initialize the storage with the given data items and vectors.'''
        ...

    @abstractmethod
    def search(self, query: vector, k: int, with_vector: bool = False) -> list[tuple[float, D]] | list[tuple[float, D, vector]]:
        '''Return the (approximate) nearest data item'''
        ...

    @abstractmethod
    def restore(self, sended_data: Iterable[tuple[D, vector]]):
        '''restore the sended data item'''
        ...

    @abstractmethod
    def get_from_restore(self, items: Iterable[D]) -> list[vector]:
        '''Resturn the vector of the given data item in restored'''
        ...

    @abstractmethod
    def get_vectors(self, items: Iterable[D]) -> list[vector]:
        '''Return the vector of the given data item.'''
        ...

    @staticmethod
    @abstractmethod
    def distance(query: vector, target: vector) -> float:
        '''Return the distance between two vectors.'''
        ...

    @staticmethod
    @abstractmethod
    def build_from_dataset(dataset: AbstractDataSet[D], model: AbstractEmbeddingModel[D]) -> 'AbstractVectorStorage':
        '''Build the vector storage from the dataset.'''
        ...
