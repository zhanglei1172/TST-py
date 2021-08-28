import numpy as np

from tst.smbo import SMBO
from tst.surrogate import TwoStageSurrogate
from tst.utils.arg import cfg, load_cfg_from_args
from tst.utils.data import load_datasets
from tst.acq import EI

def experiment(cfg):
    datasets, filenames = load_datasets(cfg.file_path)
    test_idx = filenames.index(cfg.dataset)
    # new_datasets = datasets[test_idx]
    tst_surrogates = None
    tst_cached_predictions = None
    new_datasets = datasets.pop(test_idx)

    rank = np.zeros((cfg.tries, cfg.iter_num))
    acc = np.zeros((cfg.tries, cfg.iter_num))
    count = np.zeros((cfg.tries))
    for iter in range(cfg.iter_num):
        print("starting iter {:d}".format(iter))
        acq = EI()
        surrogate = None

        if tst_surrogates is None:
            surr = TwoStageSurrogate(datasets, new_datasets, cfg.bandwidth, cfg.hp_num, meta_features=False)
            tst_surrogates = surr.gps
            tst_cached_predictions = surr.cached_predictions
        else:
            surr = TwoStageSurrogate(datasets, new_datasets, cfg.bandwidth, cfg.hp_num,
                                     meta_features=False, cached_predictions=tst_cached_predictions,
                                     tst_surrogates=tst_surrogates)
        smbo = SMBO(new_datasets,acq,surr)
        for t in range(cfg.tries):
            if t > 0 and rank[t-1][iter] == 1:
                acc[t][iter] = acc[t-1][iter]
                rank[t][iter] = rank[t-1][iter]
            else:
                smbo.iterate()
                acc[t][iter] = smbo.getBestAcc()
                rank[t][iter] = smbo.getBestRank()
                count[t] += 1



def main():
    cfg_clone = cfg.clone()
    load_cfg_from_args(cfg_clone, argv=['-f', './example.yaml'])
    experiment(cfg_clone)

if __name__ == '__main__':
    main()