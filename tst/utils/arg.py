import argparse
from yacs.config import CfgNode

_C = CfgNode()

cfg = _C

_C.file_path = "data/svm/"
_C.dataset = "A9A"

_C.tries = 10
_C.surrogate = "tst-r"
_C.bandwidth = 0.1
_C.hp_num = 6
_C.hp_indicator_num = 3
_C.seed = 0
_C.iter_num = 1
_C.sparse_grid = False

def load_cfg_from_args(_cfg, argv=None, description=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest="cfg_file", required=True)
    args = parser.parse_args(argv)
    _cfg.merge_from_file(args.cfg_file)
    print("Experiment parameters setting:\n", _cfg)
    _cfg.freeze()

