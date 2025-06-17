from benchmark.benchmark import FANNSBenchmark, FANNSGroundTruth
from rpc import basenn_wrapper as fed_basenn

import time
import os

if __name__ == "__main__":

    # Set the timezone to Asia/Shanghai
    os.environ['TZ'] = 'Asia/Shanghai'
    time.tzset()

    # Experiement setting
    import config

    # Get the ground truth
    ds = config.DATASET_CLS(config.DATA_PATHS)
    qds = config.DATASET_CLS(config.QUERY_PATH)
    dlist = [d for d in ds]
    qlist = [q for q in qds][:config.MAX_QUERIES]
    k = config.QUERY_K

    print("="*20 + " " + config.QUERY_METHOD.__name__ +
          " K:" + str(k) + " " + "="*20)
    bench_gt = FANNSGroundTruth(
        config.SETTING+"K_"+str(k), dlist, qlist, k, config.QUERY_MODEL, config.TORCH_DEVICE)
    bench_gt.embed()
    gts = bench_gt.ground_truth()

    # Prepare the benchmark
    fedmgr_args = (len(config.PORTS), config.SOCKETS, -
                   1, fed_basenn.basenn_method_map)
    bench = FANNSBenchmark(config.QUERY_METHOD, fedmgr_args, qlist, k, config.QUERY_MODEL,
                           config.TORCH_DEVICE, config.QUERY_PARAMS, config.QUERY_PARAM_NAMES)

    bench.run_and_evaluate(4, gts)
    print("Done")
