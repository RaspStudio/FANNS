from core.model_impl import *
import io, struct, pickle, torch
from torch.serialization import safe_globals
import pickle
import base64


class AbstractEmbeddingModelPacked:
    @staticmethod
    def to_bytes(model_instance: 'AbstractEmbeddingModel') -> bytes:
        device = model_instance.device_
        model_instance.device_ = "cpu"
        model_instance.to_device()
        serialized = base64.b64encode(pickle.dumps(model_instance))
        model_instance.device_ = device
        model_instance.to_device()

        return serialized

    @staticmethod
    def from_bytes(combined_bytes: bytes, device:str) -> 'AbstractEmbeddingModel':
        

        
        instance = pickle.loads(base64.b64decode(combined_bytes))
        instance.device_ = device
        instance.to_device()
        return instance