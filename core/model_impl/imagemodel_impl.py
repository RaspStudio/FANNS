
from typing import Any, Callable, Iterable

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from core.interface import AbstractData, AbstractEmbeddingModel, vector

#---------------------# Model Implementations #---------------------#

def default_pre_process(data: Any) -> torch.Tensor:
    return transforms.Compose([transforms.ToTensor()])(data)

def default_post_process(output: torch.Tensor) -> vector:
    return [o for o in output.squeeze().numpy()]

class TorchVisionEmbeddingModel[D: AbstractData](AbstractEmbeddingModel[D]):
    '''An implementation of the embedding model using PyTorch.'''

    model_: torch.nn.Module
    device_: torch.device

    def __init__(self, device: str, model: torch.nn.Module, 
                 pre_process: Callable[[Any], torch.Tensor] = default_pre_process, 
                 post_process: Callable[[torch.Tensor], vector] = default_post_process):
        self.model_ = model
        self.device_ = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_.to(self.device_)
        self.model_.eval()

        self.pre_process_ = pre_process
        self.post_process_ = post_process

    def to_device(self):
        curdev = next(self.model_.parameters()).device       
        if curdev != self.device_:
            self.model_ = self.model_.to(self.device_)

    def embed(self, data: Iterable[D]) -> list[vector]:
        embeddings = []
    
        process_data = []

        for d in data:
            processed = self.pre_process_(d.data())
            process_data.append(processed)

        data_loader = DataLoader(process_data, batch_size=32, shuffle=False)
        
        batch_outputs = []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device_)
                batch_outputs.append(self.model_(batch).cpu())
        for batch_output in batch_outputs:
            for output in batch_output:
                embedding = self.post_process_(output)
                embeddings.append(embedding)
   
        return embeddings
        

class ResNet34Model[D: AbstractData](TorchVisionEmbeddingModel[D]):
    
    def __init__(self, device: str):
        model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        super().__init__(device, model)


class ResNet101Model[D: AbstractData](TorchVisionEmbeddingModel[D]):
    
    def __init__(self, device: str):
        model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        super().__init__(device, model)


class ResNet152Model[D:AbstractData](TorchVisionEmbeddingModel[D]):

    def __init__(self, device: str):
        model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        super().__init__(device, model)


class VitModel[D: AbstractData](TorchVisionEmbeddingModel[D]):

    def __init__(self, device: str):
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = torch.nn.Identity()
        super().__init__(device, model)


class SwinV2SModel[D: AbstractData](TorchVisionEmbeddingModel[D]):

    def __init__(self, device: str):
        model = torchvision.models.swin_v2_s(weights=torchvision.models.Swin_V2_S_Weights.IMAGENET1K_V1)
        model.head = torch.nn.Identity()
        super().__init__(device, model)


class SwinV2BModel[D: AbstractData](TorchVisionEmbeddingModel[D]):

    def __init__(self, device: str):
        model = torchvision.models.swin_v2_b(weights=torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1)
        model.head = torch.nn.Identity()
        super().__init__(device, model)


class SwinV2TModel[D: AbstractData](TorchVisionEmbeddingModel[D]):

    def __init__(self, device: str):
        model = torchvision.models.swin_v2_t(weights=torchvision.models.Swin_V2_T_Weights.IMAGENET1K_V1)
        model.head = torch.nn.Identity()
        super().__init__(device, model)