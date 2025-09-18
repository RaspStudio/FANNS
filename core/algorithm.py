import concurrent.futures.thread as thread
import random
import time
from typing import Callable
import gc
from core.index_impl import FlatIndex
from core.interface import AbstractData, AbstractEmbeddingModel, vector, AbstractProtector
from core.owner import OwnerContext
from rpc.federation import FederationManager
from core.pack_model import AbstractEmbeddingModelPacked
import torch
import struct
# ---------------------# Server Implementations #---------------------#
NOT_USED_INT = 0
NOT_USED_BYTES = bytes()

PHASE_GREEDY = 1
PHASE_FEEDBACK = 2
PHASE_TRANSFER = 3
PHASE_GET_REEMBED = 4
def handler[D: AbstractData](id: str, data: bytes, phase: int, args: list[int], fargs:list[float], model: AbstractEmbeddingModel[D], ctx: OwnerContext[D], data_cls: type[D], privacy: bool, privacy_protector: AbstractProtector, device: str) -> list[bytes]:
    model.to_device()
    vdb = ctx.vstore()
    qctx = ctx.qstore(id)
    def pop_from_remaining(k: int, result_to_bytes: Callable[[tuple[float, D]], bytes] = lambda pair: pair[1].to_bytes(), erase: bool = True, restore: bool = False, not_bytes: bool = False) -> list[bytes]:
        if "remaining" not in qctx:
            raise RuntimeError("No Remaining Results")
        topk: list[tuple[float, D]] = qctx["remaining"]
        ret_end = min(k, len(topk))
        if erase:
            remaining = topk[ret_end:]
            qctx["remaining"] = remaining
        if restore:
            vdb.restore([(o[1],o[2]) for o in topk[:ret_end]])
        if privacy:
            for o in topk[:ret_end]:
                privacy_protector.protect([o[1]])

        if not_bytes:
            return [o[1] for o in topk[:ret_end]]
        return [result_to_bytes(o) for o in topk[:ret_end]]

    if phase == PHASE_GREEDY:
        # Greedy Phase
        if len(qctx) == 0:
            # Initialize Top-K and return
            query = data_cls.from_bytes(data)
            query_embedded = model.embed([query])[0]
            qctx["remaining"] = vdb.search(query_embedded, args[0])
            return pop_from_remaining(args[1])
        else:
            # Return Some Remaining Results
            return pop_from_remaining(args[1])

    elif phase == PHASE_FEEDBACK:
        # Feedback Phase
        if len(qctx) == 0:
            raise RuntimeError("Feedback Phase Requires Previous Greedy Phase")
        if data == NOT_USED_BYTES:
            return pop_from_remaining(args[1], restore=True)
        # Embed Feedback
        feedback = data_cls.from_bytes(data)
        feedback_embedded = vdb.get_from_restore([feedback])[0]
        # Update Top-K
        topk: list[tuple[float, D, vector]
                   ] = qctx["remaining"][:int(2*args[1])]

        # Create New Top-K Using Feedback
        new_topk: list[tuple[float, D]] = []
        mix_rate = args[0] / 100.0
        for odis, odata, ovec in topk:
            mix_dis = mix_rate * \
                FlatIndex.distance(feedback_embedded, ovec) + odis
            new_topk.append((mix_dis, odata, ovec))
        new_topk.sort(key=lambda x: x[0])
        res = [odata for _, odata, _ in new_topk[:args[1]]]
        qctx["remaining"] = [
            item for item in qctx["remaining"] if item[1] not in res]

        vdb.restore([(o[1], o[2]) for o in new_topk[:args[1]]])
        return [o[1].to_bytes() for o in new_topk[:args[1]]]
    elif phase == PHASE_TRANSFER:
        query_len = int.from_bytes(data[:4], 'big')
        query = data_cls.from_bytes(data[4:4+query_len])
        model_bytes = data[4+query_len:]

        start_time = time.perf_counter_ns()
        received_model = AbstractEmbeddingModelPacked.from_bytes(model_bytes, device)
        query_embedded_received = received_model.embed([query])[0]
    
        res = pop_from_remaining(args[1], erase = False, not_bytes=True)
        ## use the received model to re-embeded the res

        dist = []
        start_time = time.perf_counter_ns()
        for i in range(args[1]):
            re_embedding = received_model.embed([res[i]])[0]
            dist.append((FlatIndex.distance(query_embedded_received, re_embedding), res[i]))
        print(f"reembed and cal: {(time.perf_counter_ns() - start_time) / 1e9}")
        dist.sort(key=lambda x: x[0])
        qctx["re_embeded"] = dist


        del received_model.model_
        received_model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
   
        return  [struct.pack('f', dist[i][0]) for i in range(args[2])]
    elif phase == PHASE_GET_REEMBED:
        res = qctx["re_embeded"]
        qctx["re_embeded"] = qctx["re_embeded"][args[1]:]
        return [o[1].to_bytes() for o in res[:args[1]]]
    else:
        raise RuntimeError("Unimplemented phase")

# ---------------------# Client Implementations #---------------------#


def base_nn_postprocess[D: AbstractData](query: D, qvec: vector, model: AbstractEmbeddingModel[D], basenn_results: list[tuple[int, list[bytes]]], cosine_similarity: bool = False) -> list[tuple[float, D, int, vector]]:
    ret: list[tuple[float, D, int, vector]] = []
    # Unpack Results
    for fromid, res in basenn_results:
        datas = [query.from_bytes(o) for o in res]
        embeddings = model.embed(datas)
        if cosine_similarity:
            dists = [-FlatIPIndex.distance(qvec, e) for e in embeddings]
        else:
            dists = [FlatIndex.distance(qvec, e) for e in embeddings]
        ret.extend([(dist, data, fromid, emb) for dist, data, emb in zip(dists, datas, embeddings)])

    # Sort and Return
    ret.sort(key=lambda x: x[0])
    return ret


def initialize[D: AbstractData](fedmgr: FederationManager, query: D, k: int, first_takes: list[int],
                                init_overtakes: list[float], model: AbstractEmbeddingModel[D], max_workers: int | None = None, cosine_similarity: bool = False) -> tuple[str, vector, list[tuple[float, D, int, vector]]]:
    # Generate QID
    qid = "test" + str(time.perf_counter_ns())

    # Send query to all servers for init
    raw_ress: list[tuple[int, list[bytes]]] = []
    if max_workers == None:
        max_workers = fedmgr.n_sources_
    with thread.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                fedmgr.base_nn,
                i, qid, query.to_bytes(), PHASE_GREEDY, [int(
                    init_overtakes[i]*k) + 1, first_takes[i]], []
            ) for i in range(fedmgr.n_sources_)
        ]
        raw_ress = [(i, f.result()) for i, f in enumerate(futures)]

    # Embed all results and query # TODO optimize embed
    query_embedded = model.embed([query])[0]
    cands = base_nn_postprocess(
        query, query_embedded, model, raw_ress, cosine_similarity)

    return qid, query_embedded, cands


def fanns_competition[D: AbstractData](fedmgr: FederationManager, query: D, k: int, model: AbstractEmbeddingModel[D], model_device: str, overtake: float, cosine_similarity: bool = False) -> list[tuple[float, D]]:
    qid, qvec, cands = initialize(fedmgr, query, k, [1]*fedmgr.n_sources_, [overtake + 1/k]*fedmgr.n_sources_, model, cosine_similarity=cosine_similarity)
    
    # Loop Till Finish
    pool: list[tuple[float, D]] = []
    while len(pool) < int(overtake*k):
        # Pick a Candidate
        if len(cands) == 0:
            print("Not Enough Candidates")
            break

        # Pick Nearest Candidate
        cand_dist, cand_data, cand_fromid, _ = cands.pop(0)
        pool.append((cand_dist, cand_data))

        # Get More Candidates from the Same Server
        res = fedmgr.base_nn(cand_fromid, qid, NOT_USED_BYTES, PHASE_GREEDY, [NOT_USED_INT, 1],[])
        new_cands = base_nn_postprocess(query, qvec, model, [(cand_fromid, res)], cosine_similarity=cosine_similarity)
        # Merge New Candidates
        cands.extend(new_cands)
        cands.sort(key=lambda x: x[0])

    # Return Top-K
    pool.sort(key=lambda x: x[0])
    return pool[:k]



def fanns_contribution[D: AbstractData](fedmgr: FederationManager, query: D, k: int, model: AbstractEmbeddingModel[D], model_device: str,
                                        prob_base: float, overtake: float, batch_size: int,
                                        feedback: bool = False, feedback_rate: int = 10,
                                        prob_decrease: float = 1, cosine_similarity: bool = False) -> list[tuple[float, D]]:
    # Initialize
    qid, qvec, cands = initialize(fedmgr, query, k, [k//fedmgr.n_sources_]*fedmgr.n_sources_, [
                                  overtake + 1/k]*fedmgr.n_sources_, model, cosine_similarity=cosine_similarity)
    # Loop Till Finish

    def top_cand(fid: int) -> bytes:
        for _, odata, ofrom, _ in cands[:k]:
            if ofrom == fid:
                return odata.to_bytes()
        return NOT_USED_BYTES

    with thread.ThreadPoolExecutor(max_workers=fedmgr.n_sources_) as executor:
        while len(cands) < overtake * k:
            # Update Probabilities
            topk_freqs = [len([NOT_USED_INT for _, _, fromid, _ in cands[:min(k,len(cands)//fedmgr.n_sources_)] if fromid == i]) for i in range(fedmgr.n_sources_)]
            probs = [t + prob_base for t in topk_freqs]
            prob_base *= prob_decrease
            # Sample
            samples = random.choices(range(fedmgr.n_sources_), weights=probs, k=batch_size)
            need_takes = [len([NOT_USED_INT for o in samples if o == i]) for i in range(fedmgr.n_sources_)]
            futures = [
                    (i,
                    executor.submit(
                        fedmgr.base_nn, 
                        i, qid, 
                        NOT_USED_BYTES if not feedback else top_cand(i),
                        PHASE_GREEDY if not feedback else PHASE_FEEDBACK, 
                        [NOT_USED_INT if not feedback else feedback_rate, need_takes[i]],
                        []
                    )) for i in range(fedmgr.n_sources_) if need_takes[i] > 0
            ]
            
            for i, f in futures:
                    res = f.result()
                    new_cands = base_nn_postprocess(
                        query, qvec, model, [(i, res)], cosine_similarity=cosine_similarity)
                    cands.extend(new_cands)
            # Sort and Update Top-K Takes
            cands.sort(key=lambda x: x[0])

    res = [(dist, data) for dist, data, _, _ in cands[:k]]
    # Return Top-K
    return res


def epsilon_greedy[D:AbstractData](fedmgr: FederationManager, query: D, k: int, model: AbstractEmbeddingModel[D], model_device: str, overtake: float, epsilon: float, cosine_similarity: bool = False) -> list[tuple[float, D]]:
    qid, qvec, cands = initialize(fedmgr, query, k, [1]*fedmgr.n_sources_, [overtake + 1/k]*fedmgr.n_sources_, model, cosine_similarity=cosine_similarity)
    sample_num = [1]*fedmgr.n_sources_

    while len(cands) < overtake * k:
        topk_freqs = [len([NOT_USED_INT for _, _, fromid, _ in cands[:min(k, len(cands))] if fromid == i]) for i in range(fedmgr.n_sources_)]

        values = [topk_freqs[i] / sample_num[i] for i in range(fedmgr.n_sources_)]
        max_val = max(values)
        max_indices = [i for i, v in enumerate(values) if v == max_val]
        max_index = random.choice(max_indices)

        probs = [epsilon / max(fedmgr.n_sources_ - 1, 1)] * fedmgr.n_sources_
        probs[max_index] = 1 - epsilon

        sample = random.choices(range(fedmgr.n_sources_), weights=probs, k=1)[0]
        res = fedmgr.base_nn(sample, qid, NOT_USED_BYTES, PHASE_GREEDY, [NOT_USED_INT, 1], [])
    
        new_cands = base_nn_postprocess(query, qvec, model, [(sample, res)], cosine_similarity=cosine_similarity)
     

        cands.extend(new_cands)
        cands.sort(key=lambda x: x[0])

        sample_num[sample] += 1
    return [(dist, data) for dist, data, _, _ in cands[:k]]


def UCB[D:AbstractData](fedmgr: FederationManager, query: D, k: int, model: AbstractEmbeddingModel[D], model_device: str, overtake: float, c: float, cosine_similarity: bool = False) -> list[tuple[float, D]]:
    qid, qvec, cands = initialize(fedmgr, query, k, [1]*fedmgr.n_sources_, [overtake + 1/k]*fedmgr.n_sources_, model, cosine_similarity=cosine_similarity)
    sample_num = [1]*fedmgr.n_sources_

    while len(cands) < overtake * k:
        topk_freqs = [len([NOT_USED_INT for _, _, fromid, _ in cands[:min(k, len(cands))] if fromid == i]) for i in range(fedmgr.n_sources_)]

        values = [topk_freqs[i] / sample_num[i]  + c * math.sqrt(math.log(sum(sample_num)) / sample_num[i]) for i in range(fedmgr.n_sources_)]
        max_val = max(values)
        max_indices = [i for i, v in enumerate(values) if v == max_val]
        sample = random.choice(max_indices)
        res = fedmgr.base_nn(sample, qid, NOT_USED_BYTES, PHASE_GREEDY, [NOT_USED_INT, 1], [])
        new_cands = base_nn_postprocess(query, qvec, model, [(sample, res)], cosine_similarity=cosine_similarity)

        cands.extend(new_cands)
        cands.sort(key=lambda x: x[0])

        sample_num[sample] += 1

    return [(dist, data) for dist, data, _, _ in cands[:k]]

def transfer_Model[D:AbstractData](fedmgr: FederationManager, query: D, k: int, model: AbstractEmbeddingModel[D], model_device: str, overtake: float) -> list[tuple[float, D]]:
    
    qid, qvec, cands = initialize(fedmgr, query, k, [0]*fedmgr.n_sources_, [overtake + 1/k]*fedmgr.n_sources_, model)
    # transer model into bytes
    model_bytes = AbstractEmbeddingModelPacked.to_bytes(model)
    query_bytes = query.to_bytes()
    combined_bytes = len(query_bytes).to_bytes(4, 'big') + query_bytes + model_bytes
    with thread.ThreadPoolExecutor(max_workers=fedmgr.n_sources_) as executor:
        futures = [
                    (i,
                    executor.submit(
                        fedmgr.base_nn, 
                        i, qid, 
                        combined_bytes,
                        PHASE_TRANSFER, 
                        [NOT_USED_INT, overtake * k, k],
                        []
                    )) for i in range(fedmgr.n_sources_) 
            ]
    
    dist = []
    for i, f in futures:
        dist.extend([(i,struct.unpack('f',d)) for d in f.result()])

    dist.sort(key=lambda x: x[1])
    res = []
    for item in dist[:k]:
        d = fedmgr.base_nn(item[0], qid, NOT_USED_BYTES, PHASE_GET_REEMBED
        , [NOT_USED_INT, 1], [])[0]
        res.append((item[1], query.from_bytes(d)))

    return res

