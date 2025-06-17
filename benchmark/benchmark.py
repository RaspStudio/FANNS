
import os
import pickle
from concurrent.futures import Future, ProcessPoolExecutor
from itertools import product as itertools_product
from time import perf_counter_ns as bench_timer
from time import strftime
from typing import Any, Callable, Iterable, Sequence

from core.index_impl import FlatIndex
from core.interface import AbstractData, AbstractEmbeddingModel, vector
from rpc.federation import FederationManager

def _run_a_query[D: AbstractData](algo: Callable[..., Sequence[tuple[float, D]]], fedmgr_args, query: D, k: int, 
                    query_model: AbstractEmbeddingModel[D], query_model_device: str, args: tuple) -> tuple[int, int, list[str]]:
    
    # Instiantiate the fedmgr
    fedmgr = FederationManager(*fedmgr_args)
    # Run the algorithm for a single query
    t_start = bench_timer()
    res = algo(fedmgr, query, k, query_model, query_model_device, *args)
    t_end = bench_timer()
    res_labels = [o.label() for _, o in res]
    return t_end - t_start, fedmgr.get_total_comm(), res_labels


class FANNSBenchmark[D: AbstractData]:
    
    def __init__(self, algo: Callable[..., Sequence[tuple[float, D]]], fedmgr_args, queries: Sequence[D], k: int, query_model_cls: type[AbstractEmbeddingModel[D]], device: str, args: list[list[Any]], arg_names: list[str] | None):
        self.algo_ = algo
        self.fedmgr_args_ = fedmgr_args
        self.queries_ = queries
        self.k_ = k
        self.query_model_cls_ = query_model_cls
        self.device_ = device
        self.arg_names_ = arg_names
        self.combinations_: list[tuple[Any]] = list(itertools_product(*args))

    def run_and_evaluate(self, max_workers: int, ground_truths: list[set[str]]) -> list[tuple[float, float]]:

        print("="*20 + " " + self.algo_.__name__ + " Evaluation start at " + strftime("%Y-%m-%d %H:%M:%S") + " " + "="*20)
        argnames = self.arg_names_ if self.arg_names_ is not None else [f"Arg{i}" for i in range(len(self.combinations_[0]))]

        ress: list[tuple[float, float]] = []     
        for args in self.combinations_:
            avgt, hit, comm = self._run_a_combination(max_workers, args, ground_truths)
            for name, arg in zip(argnames, args):
                print(f"{name}: {arg}", end="; ")
            print(f"Hit: {hit}/{self.k_*len(self.queries_)} = {hit/(self.k_*len(self.queries_))}, Avg Time: {avgt/len(self.queries_)/1e9}s, Avg Communication: {comm/len(self.queries_)/1024:.2f} KB")
            ress.append((avgt, hit))

        return ress
    
    def _run_a_combination(self, max_workers: int, args: tuple[Any], ground_truths: list[set[str]]) -> tuple[float, float, float]:
        # Using given args to run the algorithm for all queries, Return the time and hit
        ret: list[tuple[int, int, int]] = []
        model = self.query_model_cls_(self.device_)
        results: list[tuple[int, int, list[str]]] = []
        for i, query in enumerate(self.queries_):
            print(f"\rRunning query {i}/{len(self.queries_)}...", end="")        
            results.append(_run_a_query(self.algo_, self.fedmgr_args_, query, self.k_, model, self.device_, args))

        print("\r", end="")

        for query, result, ground_truth in zip(self.queries_, results, ground_truths):
            t, comm, res_labels = result
            ret.append((t, comm, len(set(res_labels) & ground_truth)))

        # Return the total time and hit
        return sum(t for t, _, _ in ret), sum(hit for _, _, hit in ret), sum(c for _, c, _ in ret)
    
def _disted[D: AbstractData](query: vector, datas: list[tuple[D, vector]], k: int) -> list[tuple[float, D]]:
    ret = [(FlatIndex.distance(query, e), d) for d, e in datas]
    ret.sort(key=lambda x: x[0])
    return ret[:k]
    
class FANNSGroundTruth[D: AbstractData]:
    
    def __init__(self, testid: str, datas: Iterable[D], queries: Iterable[D], k: int, query_model_cls: type[AbstractEmbeddingModel[D]], device: str) -> None:
        self.testid_ = testid
        self.datas_ = [d for d in datas]
        self.queries_ = [q for q in queries]
        self.k_ = k
        self.query_model_cls_ = query_model_cls
        self.device_ = device

        self.data_embs_: list[vector] = []
        self.query_embs_: list[vector] = []

    def embed(self, overwrite: bool = False, echo: bool = False) -> None:
        # Prepare Ground Truth Calculation
        model = self.query_model_cls_(self.device_)
        
        self.data_embs_ = []
        print("Start Load or Embed Data") if echo else ...
        for i, data in enumerate(self.datas_):
            embpkl_path = data.label() + f".{model.__class__.__name__}.emb"
            if not overwrite and os.path.exists(embpkl_path):
                data_emb = pickle.load(open(embpkl_path, "rb"))
            else:
                data_emb = model.embed([data])[0]
                pickle.dump(data_emb, open(embpkl_path, "wb"))
            self.data_embs_.append(data_emb)
            print(f"Embedded {i+1}/{len(self.datas_)}...", end="\r") if echo else ...
        print(f"Embedded {len(self.datas_)} Data Objects")


        self.query_embs_ = []
        for i, query in enumerate(self.queries_):
            embpkl_path = query.label() + f".{model.__class__.__name__}.emb"
            if not overwrite and os.path.exists(embpkl_path):
                query_emb = pickle.load(open(embpkl_path, "rb"))
            else:
                query_emb = model.embed([query])[0]
                pickle.dump(query_emb, open(embpkl_path, "wb"))
            self.query_embs_.append(query_emb)
            print(f"Embedded {i+1}/{len(self.queries_)}...", end="\r") if echo else ...
        print(f"Embedded {len(self.queries_)} Query Objects")

    def ground_truth(self, overwrite: bool = False, use_index: bool = True) -> list[set[str]]:
        # Checkpoint
        gtpkl_path = self.testid_ + "_gt.pkl"

        if not overwrite and os.path.exists(gtpkl_path):
            print(f"Loading GT from pickle: {gtpkl_path}")
            gtsdata = pickle.load(open(gtpkl_path, "rb"))
            return [set(o.label() for _, o in gts) for gts in gtsdata]
        
        print("Creating GT")

        if use_index:
            # Using FlatIndex
            indexpkl_path = self.testid_ + f"_index{len(self.datas_)}.pkl"
            print(indexpkl_path)
            if not overwrite and os.path.exists(indexpkl_path):
                print(f"Loading GT Index from pickle: {indexpkl_path}")
                index = pickle.load(open(indexpkl_path, "rb"))
            else:
                print(f"Creating GT Index to: {indexpkl_path}")
                index = FlatIndex([(d, e) for d, e in zip(self.datas_, self.data_embs_)])
                pickle.dump(index, open(indexpkl_path, "wb"))
            gtsdata: list[list[tuple[float, D]]] = []
            for query, query_emb in zip(self.queries_, self.query_embs_):
                gtsdata.append(index.search(query_emb, self.k_))

        else:
            # Parralelize GT Calculation
            future_qs_gts: list[list[Future[list[tuple[float, D]]]]] = []
            SEGMENT_SIZE = 100
            with ProcessPoolExecutor(max_workers=16) as executor:
                for query_emb in self.query_embs_:
                    curq_gts: list[Future[list[tuple[float, D]]]] = []
                    for i in range(0, len(self.datas_), SEGMENT_SIZE):
                        curq_gts.append(executor.submit(_disted, query_emb, list(zip(self.datas_[i:i+SEGMENT_SIZE], self.data_embs_[i:i+SEGMENT_SIZE])), self.k_))
                    future_qs_gts.append(curq_gts)
            
            gtsdata: list[list[tuple[float, D]]] = []
            for future_q_gts in future_qs_gts:
                q_dists: list[tuple[float, D]] = []
                for future_gts in future_q_gts:
                    q_dists.extend(future_gts.result())
                q_dists.sort(key=lambda x: x[0])
                gtsdata.append(q_dists[:self.k_])

        pickle.dump(gtsdata, open(gtpkl_path, "wb"))

        return [set(o.label() for _, o in gts) for gts in gtsdata]
