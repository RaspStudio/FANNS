import os

from core.interface import vector
from core.interface import AbstractData, AbstractDataSet, AbstractEmbeddingModel, AbstractVectorStorage

import json
from typing import List

#---------------------# Implementations #---------------------#

class MarcoData(AbstractData):
    data_: str | None

    def __init__(self, path: str = ""):
        '''Initialize the image data item from the image file path.'''
        if len(path) > 0:
            self.path_ = path
            self.data_ = None
            self.label_ = path.removesuffix('.txt')
        else:
            ... # from_bytes

    def data(self) -> str:
        if self.data_ is None:
            try:
                with open(self.path_, 'r',encoding='utf-8') as file:
                    ret = file.read()
            except UnicodeDecodeError  as e:
                print(self.path_)
                print("UnicodeDecodeError")
        else:
            ret = self.data_
        return ret

    @staticmethod
    def from_bytes(data: str) -> 'MarcoData':
        data_len = int.from_bytes(data[:4],'big')
        label_len = int.from_bytes(data[4:8],'big')
        data_bytes = data[8:8+data_len]
        label_bytes = data[8+data_len:]
        
        if len(label_bytes) != label_len:
            raise ValueError('The length of the label is incorrect.')
        
        ret = MarcoData()
        ret.data_ = data_bytes.decode()
        ret.label_ = label_bytes.decode()

        return ret
    
    def to_bytes(self) -> bytes:
        data_bytes = self.data().encode()
        data_len = len(data_bytes).to_bytes(4,'big')
        label_bytes = self.label_.encode()
        label_len = len(label_bytes).to_bytes(4,'big')
        ret = data_len + label_len + data_bytes + label_bytes
        return ret
    
    def label(self) -> str:
        return self.label_
    
    def save_as(self, path: str):
        with open(path, 'w') as file:
            file.write(self.data_)
        

#---------------------# Dataset Implementations #---------------------#

class MarcoDataSet(AbstractDataSet[MarcoData]):

    def __init__(self, path: str | list[str] = ""):
        '''Initialize the image dataset from the image directory path.'''
        self.data_ = []

        if isinstance(path, str):
            path_ = [path]
        else:
            path_ = path

        for p in path_:
            with open(p, "r", encoding="utf-8") as file:
                data = json.load(file) 
        
            for file in data['data']:
                self.data_.append(MarcoData(os.path.join(data['path'],file)))
    
    def __len__(self) -> int:
        '''Return the number of texts in the dataset.'''
        return len(self.data_)
    
    def __getitem__(self, idx: int) -> MarcoData:
        '''Return the image at the given index.'''
        return self.data_[idx]
    
    def path(self) -> str:
        return self.path_
    
    def merge(self, other: 'MarcoDataSet') -> None:
        '''Merge the dataset with another dataset.'''
        self.data_.extend(other.data_)

    @staticmethod
    def from_list(data: list[MarcoData]) -> 'MarcoDataSet':
        '''Initialize the image dataset from a list of image data items.'''
        ret = MarcoDataSet()
        ret.data_ = data
        return ret

