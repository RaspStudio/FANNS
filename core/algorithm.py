import concurrent.futures.thread as thread
import random
import time
from typing import Callable

from core.index_impl import FlatIndex
from core.interface import AbstractData, AbstractEmbeddingModel, vector
from core.owner import OwnerContext
from rpc.federation import FederationManager

# ---------------------# Server Implementations #---------------------#
NOT_USED_INT = 0
NOT_USED_BYTES = bytes()

PHASE_GREEDY = 1
PHASE_FEEDBACK = 2


def handler[D: AbstractData](id: str, data: bytes, phase: int, args: list[int], fargs: list[float], model: AbstractEmbeddingModel[D], ctx: OwnerContext[D], data_cls: type[D]) -> list[bytes]:
    model.to_device()
    vdb = ctx.vstore()
    qctx = ctx.qstore(id)

    def pop_from_remaining(k: int, result_to_bytes: Callable[[tuple[float, D]], bytes] = lambda pair: pair[1].to_bytes(), erase: bool = True, restore: bool = False) -> list[bytes]:
        if "remaining" not in qctx:
            raise RuntimeError("No Remaining Results")
        topk: list[tuple[float, D]] = qctx["remaining"]
        ret_end = min(k, len(topk))
        if erase:
            remaining = topk[ret_end:]
            qctx["remaining"] = remaining
        if restore:
            vdb.restore([(o[1], o[2]) for o in topk[:ret_end]])
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
    else:
        raise RuntimeError("Unimplemented phase")

# ---------------------# Client Implementations #---------------------#


def base_nn_postprocess[D: AbstractData](query: D, qvec: vector, model: AbstractEmbeddingModel[D], basenn_results: list[tuple[int, list[bytes]]], cosine_similarity: bool = False) -> list[tuple[float, D, int, vector]]:
    ret: list[tuple[float, D, int, vector]] = []
    # Unpack Results
    for fromid, res in basenn_results:
        datas = [query.from_bytes(o) for o in res]
        embeddings = model.embed(datas)
        dists = [FlatIndex.distance(qvec, e) for e in embeddings]
        ret.extend([(dist, data, fromid, emb)
                   for dist, data, emb in zip(dists, datas, embeddings)])

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


def fanns_competition[D: AbstractData](fedmgr: FederationManager, query: D, k: int, model: AbstractEmbeddingModel[D], model_device: str, overtake: float) -> list[tuple[float, D]]:
    # Initialize
    qid, qvec, cands = initialize(fedmgr, query, k, [
                                  1]*fedmgr.n_sources_, [overtake + 1/k]*fedmgr.n_sources_, model)

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
        res = fedmgr.base_nn(cand_fromid, qid, NOT_USED_BYTES,
                             PHASE_GREEDY, [NOT_USED_INT, 1], [])
        new_cands = base_nn_postprocess(
            query, qvec, model, [(cand_fromid, res)])

        # Merge New Candidates
        cands.extend(new_cands)
        cands.sort(key=lambda x: x[0])

    # Return Top-K
    return pool


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
            topk_freqs = [len([NOT_USED_INT for _, _, fromid, _ in cands[:min(k, len(
                cands)//fedmgr.n_sources_)] if fromid == i]) for i in range(fedmgr.n_sources_)]
            probs = [t + prob_base for t in topk_freqs]
            prob_base *= prob_decrease
            # Sample
            samples = random.choices(
                range(fedmgr.n_sources_), weights=probs, k=batch_size)
            need_takes = [len([NOT_USED_INT for o in samples if o == i])
                          for i in range(fedmgr.n_sources_)]
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
    return res
