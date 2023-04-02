import numpy as np
import networkx as nx


### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
def compute_nspdk_mmd(samples1, samples2, metric, is_hist=True, n_jobs=None):
    from sklearn.metrics.pairwise import pairwise_kernels
    from eden.graph import vectorize

    def kernel_compute(X, Y=None, is_hist=True, metric='linear', n_jobs=None):
        X = vectorize(X, complexity=4, discrete=True)
        if Y is not None:
            Y = vectorize(Y, complexity=4, discrete=True)
        return pairwise_kernels(X, Y, metric='linear', n_jobs=n_jobs)

    X = kernel_compute(samples1, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Y = kernel_compute(samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Z = kernel_compute(samples1, Y=samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)

    return np.average(X) + np.average(Y) - 2 * np.average(Z)


##### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/stats.py
def nspdk_stats(graph_ref_list, graph_pred_list):
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    # prev = datetime.now()
    mmd_dist = compute_nspdk_mmd(graph_ref_list, graph_pred_list_remove_empty, metric='nspdk', is_hist=False, n_jobs=20)
    # elapsed = datetime.now() - prev
    # if PRINT_TIME:
    #     print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def get_nspdk_eval(config):
    return nspdk_stats
