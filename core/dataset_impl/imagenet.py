import os

from PIL import Image
from torchvision import transforms

from core.interface import vector
from core.interface import AbstractData, AbstractDataSet, AbstractEmbeddingModel, AbstractVectorStorage

import json

#---------------------# Implementations #---------------------#

class ImageNetData(AbstractData):
    '''A data item containing an image.'''
    
    TRANSFORM = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB') if isinstance(x, Image.Image) and x.mode != 'RGB' else x),
        transforms.Resize((224, 224)),
    ])

    data_: Image.Image | None

    def __init__(self, path: str = ""):
        '''Initialize the image data item from the image file path.'''
        if len(path) > 0:
            self.path_ = path
            self.data_ = None
            self.label_ = path.removesuffix('.JPEG')
            self.protected = False

        else:
            ... # from_bytes

    def data(self) -> Image.Image:
        if self.protected:
            return self.data_
        if self.data_ is None:
            ret = self.TRANSFORM(Image.open(self.path_))
        else:
            ret = self.data_
        return ret

    def protected_data(self, protected_image: Image.Image):
        if not self.protected:
            self.protected = True
            self.data_ = protected_image

    @staticmethod
    def from_bytes(data: bytes) -> 'iNaturalistData':
        '''Initialize the image data item from the serialized data.'''
        data_len = int.from_bytes(data[:4], 'big')
        label_len = int.from_bytes(data[4:8], 'big')
        data_bytes = data[8:8+data_len]
        label_bytes = data[8+data_len:8+data_len+label_len]
        protected_bytes = data[8+data_len+label_len:]
        if len(label_bytes) != label_len:
            raise ValueError('The length of the label is incorrect.')
        
        ret = iNaturalistData()
        ret.data_ = Image.frombytes('RGB', (224, 224), data_bytes)
        ret.label_ = label_bytes.decode()
        ret.protected = bool(protected_bytes) 
        return ret
    
    def to_bytes(self) -> bytes:
        '''Return the serialized image data item.'''
        data_bytes = self.data().tobytes()
        data_len = len(data_bytes).to_bytes(4, 'big')
        label_bytes = self.label_.encode()
        label_len = len(label_bytes).to_bytes(4, 'big')
        protected_bytes = bytes([int(self.protected)])
        protected_len = len(protected_bytes).to_bytes(4, 'big')
        return data_len + label_len + data_bytes + label_bytes + protected_bytes
    
    def label(self) -> str:
        return self.label_
    
    def save_as(self, path: str):
        '''Save the image data item as an image file.'''
        self.data().save(path)

#---------------------# Dataset Implementations #---------------------#
    
class ImageNetDataSet(AbstractDataSet[ImageNetData]):
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
                self.data_.append(ImageNetData(os.path.join(data['path'],file)))
    
    def __len__(self) -> int:
        '''Return the number of images in the dataset.'''
        return len(self.data_)
    
    def __getitem__(self, idx: int) -> ImageNetData:
        '''Return the image at the given index.'''
        return self.data_[idx]
    
    def merge(self, other: 'ImageNetDataSet') -> None:
        '''Merge the dataset with another dataset.'''
        self.data_.extend(other.data_)

    @staticmethod
    def from_list(data: list[ImageNetData]) -> 'ImageNetDataSet':
        '''Initialize the image dataset from a list of image data items.'''
        ret = ImageNetDataSet()
        ret.data_ = data
        return ret