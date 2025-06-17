import torch
import faiss
import concurrent.futures.process
import os
import pickle
from typing import Type

from core.algorithm import handler
from core.index_impl import HNSWIndex
from core.interface import (AbstractData, AbstractDataSet,
                            AbstractEmbeddingModel, vector)
from core.owner import OwnerContext
from rpc import basenn_wrapper as fed_basenn
from rpc import federation as fed


def server_thread[D: AbstractData](path: str, port: str, model_cls: Type[AbstractEmbeddingModel[D]], dataset_cls: Type[AbstractDataSet[D]], data_cls: Type[D], device: str):
    # Initialize Server
    model = model_cls(device)

    # Offline Indexing
    pkl_path = f"{path}_{str(model.__class__.__name__)}_index.pkl"
    if os.path.exists(pkl_path):
        # Load Index From Pickle
        print(f"[{port}] Loading Index From Pickle: {pkl_path}")
        vdb = pickle.load(open(pkl_path, "rb"))
    else:
        # Create Index
        print(f"[{port}] Creating Index To: {pkl_path}")
        dataset = dataset_cls(path)
        # Load or Generate Embeddings
        vdb = HNSWIndex.build_from_dataset(dataset, model, echo=True)
        pickle.dump(vdb, open(pkl_path, "wb"))

    svc = fed.ServiceManager(port, [fed_basenn.get_servicer(
        handler, model, OwnerContext(vdb), data_cls)])
    svc.serve()


if __name__ == "__main__":
    import config
    torch.multiprocessing.set_start_method('spawn')
    faiss.omp_set_num_threads(config.OWNER_FAISS_THREADS)
    torch.set_num_threads(4)
    servers = []
    print(config.OWNER_MODELS)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, (port, path) in enumerate(zip(config.PORTS, config.DATA_PATHS)):
            server = executor.submit(
                server_thread,
                path,
                str(port),
                config.OWNER_MODELS[i],
                config.DATASET_CLS,
                config.DATA_CLS,
                config.TORCH_DEVICE
            )

    print("Server End")
