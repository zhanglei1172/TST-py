import numpy as np
import copy

from tst.utils.data import MetaInsts, MetaInstDense


class SMBO():
    def __init__(self, test_data, acq, surr):
        self.insts = test_data
        self.acq = acq
        self.surr = surr
        self.history = []
        self.history_y = []
        self.h = MetaInsts([])
        self.h.numValues = 28
        self.candidates = (test_data) # TODO
        self.best_inst = None
        self.best_inst_y = None

    def iterate(self):
        x = self.acq.getNext(max(self.history_y+[-1]), self.surr, self.candidates)
        self.candidates.remove(x)
        if self.best_inst is None or x.target > self.best_inst_y:
            self.best_inst = x
            self.best_inst_y = x.target
        self.history.append(x)
        self.history_y.append(x.target)
        self.h.add(x)
        self.surr.train(self.h)

    def getBestAcc(self):
        return self.best_inst.target

    def getBestRank(self):
        rank = 1
        for inst in self.insts:
            if inst.target > self.best_inst.target:
                rank += 1
        return rank
