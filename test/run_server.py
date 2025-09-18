import os
from core.model_impl import *

os.environ["OMP_NUM_THREADS"] = "8"       
os.environ["OPENBLAS_NUM_THREADS"] = "8"  
os.environ["MKL_NUM_THREADS"] = "8"     
os.environ["NUMEXPR_NUM_THREADS"] = "8" 
import pickle
import concurrent.futures.process
from core.model_impl import *

from typing import Type

from core.interface import AbstractData, AbstractDataSet, AbstractEmbeddingModel, vector, AbstractProtector
from core.index_impl import FlatIndex,HNSWIndex,IVFPQIndex,FlatIPIndex,IVFFlatIndex
from core.owner import OwnerContext
from core.algorithm import handler
from rpc import federation as fed
from rpc import basenn_wrapper as fed_basenn
import os, json, importlib, argparse

import importlib

def load_class(fqn: str):
    module, name = fqn.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def server_thread[D: AbstractData](path: str, port: str, model_cls: Type[AbstractEmbeddingModel[D]], dataset_cls: Type[AbstractDataSet[D]], data_cls: Type[D], device: str, privacy: bool, privacy_cls: Type[AbstractProtector]):
    # Initialize Server
    print("begin", flush=True)
    model = model_cls(device)
    pkl_path = f"{path}_{str(model.__class__.__name__)}_index.pkl"
    print("success", flush=True)
    if os.path.exists(pkl_path):
        # Load Index From Pickle
        print(f"[{port}] Loading Index From Pickle: {pkl_path}", flush=True)
        vdb = pickle.load(open(pkl_path, "rb"))
        print(type(vdb))
    else:
        # Create Index
        print(f"[{port}] Creating Index To: {pkl_path}", flush=True)
        dataset = dataset_cls(path)
        # Load or Generate Embeddings
        vdb = FlatIndex.build_from_dataset(dataset, model, echo=True)
        pickle.dump(vdb, open(pkl_path, "wb"))
    if privacy:
        privacy_protector = privacy_cls()
    else:
        privacy_protector = None
  
    svc = fed.ServiceManager(port, [fed_basenn.get_servicer(handler, model, OwnerContext(vdb), data_cls, privacy, privacy_protector, device)])
    svc.serve()


import torch
import faiss


if __name__ == "__main__":
    torch.set_num_threads(4)


    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config_path",
                        help="Path to config file (json).")
    args = parser.parse_args()
    print(args.config_path,flush=True)
    

    with open(args.config_path, "r") as f:
        cfg = json.load(f)

    print(torch.cuda.device_count(),flush=True)


    model_classes = load_class(cfg["OWNER_MODEL"])
    dataset_cls         = load_class(cfg["DATASET_CLS"])
    data_cls            = load_class(cfg["DATA_CLS"])
    if cfg['PRIVACY']:
        privacy_cls       = load_class(cfg["PRIVACY_CLS"])
    else:
        privacy_cls = None
    print(torch.cuda.is_available())
    server_thread(cfg['DATA_PATH'],
            str(cfg['PORT']),
            model_classes,
            dataset_cls,
            data_cls,
            cfg['TORCH_DEVICE'],
            cfg['PRIVACY'],
            privacy_cls)

 