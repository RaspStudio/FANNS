from benchmark.benchmark import FANNSBenchmark, FANNSGroundTruth
from rpc import basenn_wrapper as fed_basenn
import os, json, importlib, argparse
from core.algorithm import *
import time

def load_class(fqn: str):
    module, name = fqn.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)

if __name__ == "__main__":
 
    # Set the timezone to Asia/Shanghai
    os.environ['TZ'] = 'Asia/Shanghai'
    time.tzset()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config_path",
                        help="Path to config file (json).")
    args = parser.parse_args()
    print(args.config_path)
    

    with open(args.config_path, "r") as f:
        cfg = json.load(f)

    # Experiement setting
    model_class = load_class(cfg["QUERY_MODEL"])
    dataset_cls         = load_class(cfg["DATASET_CLS"])
    data_cls            = load_class(cfg["DATA_CLS"])
    
    method_map = {
        "fanns_competition": fanns_competition,
        "fanns_contribution": fanns_contribution,
        "transfer_Model": transfer_Model,
        "epsilon_greedy": epsilon_greedy,
        "UCB": UCB
    }


    # Get the ground truth
    ds = dataset_cls(cfg['DATA_PATHS'])
    qds = dataset_cls(cfg['QUERY_PATH'])
    dlist = [d for d in ds]
    qlist = [q for q in qds]
   
    

    for k in cfg['QUERY_K']: 
        print("="*20 + " " + method_map[cfg['QUERY_METHOD']].__name__ +" K:"+ str(k) + " " + "="*20, flush=True)
        bench_gt = FANNSGroundTruth(cfg['SETTING']+"K_"+str(k), dlist, qlist, k, model_class, cfg['TORCH_DEVICE'])
        bench_gt.embed()
        gts = bench_gt.ground_truth()

        print(gts[1],flush=True)
        
        fedmgr_args = (len(cfg['SOCKETS']), cfg['SOCKETS'], -1, fed_basenn.basenn_method_map)
        bench = FANNSBenchmark(method_map[cfg['QUERY_METHOD']], fedmgr_args, qlist, k, model_class, cfg['TORCH_DEVICE'], cfg['QUERY_PARAMS'], cfg['QUERY_PARAM_NAMES'])

        bench.run_and_evaluate(4, gts)
        print("Done",flush=True)
        