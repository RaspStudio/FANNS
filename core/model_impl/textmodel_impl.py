
from core.interface import vector
from core.interface import AbstractData, AbstractEmbeddingModel
from transformers import LongformerTokenizer, LongformerModel, DebertaTokenizer, DebertaModel
from sentence_transformers import SentenceTransformer

# ---------------------# Model Implementations #---------------------#
import torch
from typing import Callable, Iterable, Any


def default_pre_process(data: Any, max_len: int) -> str:
    return data.data()[:max_len]


def default_post_process(output: torch.Tensor) -> vector:

    num, _ = output.shape
    if num != 1:
        return [o for o in output.squeeze().tolist()]
    else:
        return [[o for o in output.squeeze().tolist()]]


class TorchTextEmbeddingModel[D: AbstractData](AbstractEmbeddingModel[D]):
    '''An implementation of the text embedding model using PyTorch.'''

    model_: torch.nn.Module
    device_: torch.device

    def __init__(self, device: str, model: torch.nn.Module, tokenizer=None,
                 pre_process: Callable[[Any],
                                       torch.Tensor] = default_pre_process,
                 post_process: Callable[[torch.Tensor], vector] = default_post_process):
        self.model_ = model
        self.device_ = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.model_.to(self.device_)
        self.model_.eval()
        if tokenizer:
            self.tokenizer_ = tokenizer
        else:
            self.tokenizer_ = self.model_.tokenizer
        self.pre_process_ = pre_process
        self.post_process_ = post_process

    def to_device(self):
        curdev = next(self.model_.parameters()).device
        if curdev != self.device_:
            self.model_ = self.model_.to(self.device_)

    def embed(self, data: Iterable[D]) -> list[vector]:
        if len(data) == 0:
            return []
        with torch.no_grad():

            texts = [self.pre_process_(
                text, self.tokenizer_.model_max_length) for text in data]

            output = self.model_.encode(
                texts,
                batch_size=32,
                convert_to_tensor=False,
                device=self.device_,
                show_progress_bar=False

            )

            res = self.post_process_(output)

            return res


class MsmarcoDistilbertCosV5Model[D: AbstractData](TorchTextEmbeddingModel):
    def __init__(self, device: str):
        model = SentenceTransformer("msmarco-distilbert-cos-v5", device=device)
        super().__init__(device, model)


class allMiniLML6v2Model[D:AbstractData](TorchTextEmbeddingModel):
    def __init__(self, device: str):
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        super().__init__(device, model)


class MultiQaDistilbertCosv1Model[D:AbstractData](TorchTextEmbeddingModel):
    def __init__(self, device: str):
        model = SentenceTransformer(
            "multi-qa-distilbert-cos-v1", device=device)
        super().__init__(device, model)


class MultiQaMiniLML6cosv1Model[D:AbstractData](TorchTextEmbeddingModel):
    def __init__(self, device: str):
        model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device=device)
        super().__init__(device, model)


class MultiQaMpnetBaseCosV1Model[D:AbstractData](TorchTextEmbeddingModel):
    def __init__(self, device: str):
        model = SentenceTransformer(
            "multi-qa-mpnet-base-cos-v1", device=device)
        super().__init__(device, model)


class DebertabaseModel[D:AbstractData](TorchTextEmbeddingModel):
    def __init__(self, device: str):
        model = DebertaModel.from_pretrained('microsoft/deberta-base')
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        super().__init__(device, model, tokenizer)

    def embed(self, data: Iterable[D]) -> list[vector]:
        batch_size = 32
        texts = [self.pre_process_(
            text, self.tokenizer_.model_max_length) for text in data]

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer_(batch, return_tensors="pt", padding=True,
                                     truncation=True, max_length=self.tokenizer_.model_max_length)
            inputs = {k: v.to(self.device_) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model_(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]
                embedding_numpy = embedding.cpu().numpy()

                for vector in embedding_numpy:
                    embeddings.append(vector.tolist())
        return embeddings


class LongformerBaseModel[D:AbstractData](TorchTextEmbeddingModel):
    def __init__(self, device: str):
        model = LongformerModel.from_pretrained(
            '/app/model/longformer_model/models--allenai--longformer-base-4096/snapshots/301e6a42cb0d9976a6d6a26a079fef81c18aa895')
        tokenizer = LongformerTokenizer.from_pretrained(
            '/app/model/longformer_model/models--allenai--longformer-base-4096/snapshots/301e6a42cb0d9976a6d6a26a079fef81c18aa895')
        super().__init__(device, model, tokenizer)

    def embed(self, data: Iterable[D]) -> list[vector]:
        batch_size = 32
        texts = [self.pre_process_(
            text, self.tokenizer_.model_max_length) for text in data]

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer_(batch, return_tensors="pt", padding=True,
                                     truncation=True, max_length=self.tokenizer_.model_max_length)
            inputs = {k: v.to(self.device_) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model_(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]
                embedding_numpy = embedding.cpu().numpy()

                for vector in embedding_numpy:
                    embeddings.append(vector.tolist())

        return embeddings
