import glob, os
import numpy as np
from typing import List




class MetaInst():
    pass

class MetaInstDense(MetaInst):
    cached_index = {}
    def __init__(self, target, values):
        self.target = (target)
        self._values = np.array(values)
        self._length = len(values)
        if self._length not in self.cached_index:
            self.cached_index[self._length] = list(range(self._length))

    def getValues(self, index=None):
        return self._values if index is None else self._values[index]

    def getKeys(self):
        return self.cached_index[self._length]

    def setValues(self, value, index):
        self._values[index] = value



class MetaInstSparse(MetaInst):
    def __init__(self):
        pass

class MetaInsts():
    def __init__(self, lines: List[str]):
        self.numValues = 0
        # self._targets = None
        self.instances = []
        first_line = True
        sparse = False
        for line in lines:
            if first_line:
                first_line = False
                sparse = ':' in line
            if sparse: # TODO
                pass
            else:
                line_data = list(map(float, line.strip().split(' ')))
                self.instances.append(MetaInstDense(line_data[0], line_data[1:]))
                self.numValues = max(self.numValues, len(line_data))

    def getTargets(self):
        return [inst.target for inst in self.instances]

    def add(self, inst: MetaInstDense):
        max_key = inst.getKeys()[-1]
        assert max_key <= self.numValues
        self.instances.append(inst)

    def remove(self, inst: MetaInstDense):
        self.instances.remove(inst)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]


def load_datasets(file_path):
    file_lists = glob.glob(file_path+"*")
    datasets = []
    filenames = []
    for file in file_lists:
        # data = []
        filename = file.rsplit('/', maxsplit=1)[-1]
        filenames.append(filename)
        with open(file, 'r') as f:
            meta_insts = MetaInsts(f.readlines())
        datasets.append(meta_insts)
    return datasets, filenames