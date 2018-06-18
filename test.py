import config
import models
import tensorflow as tf
import numpy as np
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CONDA_PREFIX']=''


def main():
    kb = 'RV15M' # FB15K
    model = "TransE" # DistMult
    clusters = "V2"
    
    con = config.Config()
    con.set_in_path("/u/wujieche/Projects/OpenKE/data/"+kb+"/")
    con.set_test_link_prediction(True)
    con.set_test_triple_classification(False)
    con.set_work_threads(8)
    con.set_dimension(100)
    con.set_import_files("models/{}-{}_model.vec.tf".format(model, kb))
    con.init()
    con.set_model(getattr(models, model)) # models.TransE
    con.test(clusters)
    log_path = "{}-{}_clusters_test_log.log".format(model, kb)
    os.rename("log.log", log_path)

    rank_by_relID_plot(log_path)

def read_log(path):
    import pandas as pd
    with open(path) as f:
        data = [l.split() for l in f]
    columns = ['arg1_id', 'rel_id', 'arg2_id', 'head_rank', 'tail_rank']
    df = pd.DataFrame.from_records(data, columns=columns)
    return df

def rank_by_relID_plot(log_path):
    import matplotlib.pyplot as plt
    df = read_log(log_path)
    df = df.drop(columns=["arg1_id", "arg2_id"])
    # max_r = max(df['rel_id'])
    # df['ranked_ids'] = df['rel_id'].rank(method='first')

    bins = 30
    grp = df.groupby(by = pd.qcut(df['rel_id'], bins))
    df = grp.aggregate(np.average)

    plt.plot(ids, tail_ranks, "b", label = "arg1 mean rank")
    plt.plot(ids, head_ranks, "g", label = "arg2 mean rank")
    freq_head_MR, freq_tail_MR = 169512, 237961
    darkblue, darkgreen = "#3030AA", "#40AA40"
    plt.hlines([freq_head_MR, freq_tail_MR], xmin=0, xmax = 10000, colors = [darkblue, darkgreen], label = "baseline")
    plt.legend()
    plt.savefig(log_path[:-4]+".MR_by_ID-graph.png")
    
main()
