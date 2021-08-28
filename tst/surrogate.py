import numpy as np
import scipy
from scipy.linalg import solve_triangular
import copy

from tst.kernel import SEARDKernel
from tst.utils.data import MetaInsts, MetaInstDense


class GaussianProcessRegression():
    '''
    Kstar: kernel of train and predict
    K: kernel of train and train
    '''

    def __init__(self, kernel, epoch=100, learn_kernel_param=False):
        self.epoch = epoch
        self.lern_kernel_param = learn_kernel_param
        self.kernel = kernel
        self.insts = None

    def estimateAlpha(self):
        target = [inst.target for inst in self.insts]
        _tmp = solve_triangular(self.L, target, lower=True)
        self.alpha = solve_triangular(self.L.T, _tmp, lower=False)

    def train(self, insts):
        self.insts = insts
        # target = obs_pairs[:, 0]
        # obs_lambdas = obs_pairs[:, 1:]
        if self.lern_kernel_param:
            for iter in range(self.epoch):
                kernel = self.kernel.computeKernel(insts)
                self.L = scipy.linalg.cholesky(kernel)
                self.estimateAlpha()
                self.kernel.updateKernelParam(insts, kernel, self.alpha, iter == 0)
        kernel = self.kernel.computeKernel(self.insts)
        self.L = scipy.linalg.cholesky(kernel, lower=True)
        self.estimateAlpha()

    def predict(self, inst):
        Kstar = np.array(self.getKstar(inst))
        return np.dot(Kstar, self.alpha)

    def predictWithUncerten(self, inst):
        if self.insts is None:
            return 0, np.inf
        Kstar = np.array(self.getKstar(inst))
        L_invKstar = solve_triangular(self.L, Kstar, lower=True)
        return np.dot(Kstar, self.alpha), np.sqrt(np.dot(-L_invKstar, L_invKstar) + self.kernel.computeEntry(inst, inst))

    def getKstar(self, inst2):
        return [self.kernel.computeEntry(inst1, inst2) for inst1 in self.insts]


class TwoStageSurrogate():

    def __init__(self, train_insts_list, test_insts, bandwidth, hp_num, meta_features=False, cached_predictions=None,
                 tst_surrogates=None):
        self.model = GaussianProcessRegression(SEARDKernel(hp_num))
        self.train_insts_list = train_insts_list
        self.hp_num = hp_num
        self.bandwidth = bandwidth
        self.scaledTrain = []
        # self.scaledTrain = copy.deepcopy(train)
        self.meta_num = len(train_insts_list)
        self.meta_features = meta_features
        for i in range((self.meta_num)):
            self.scaledTrain.append(MetaInsts([]))
            # self.test_insts.append(MetaInsts([]))
            self.scaledTrain[i].numValues = self.hp_num
            targets = train_insts_list[i].getTargets()
            max_ = np.max(targets)  # sclae target to [0,1]
            min_ = np.min(targets)
            for inst in (train_insts_list[i].instances):
                scaled_target = (inst.target - min_) / (max_ - min_)
                self.scaledTrain[i].add(MetaInstDense(scaled_target, inst._values[:self.hp_num]))

        if tst_surrogates is None:
            self.gps = []
            for i in range(self.meta_num):
                gp = GaussianProcessRegression(
                    SEARDKernel(hp_num), learn_kernel_param=False
                )
                gp.train(self.scaledTrain[i])
                self.gps.append(gp)
        else:
            self.gps = tst_surrogates
        if cached_predictions is None:
            self.preComputeSimilarity(test_insts)
        else:
            self.cached_predictions = cached_predictions
        if self.meta_features:
            raise NotImplemented()
        else:
            self.kernel = self.kendallTauCorrelation
        self.similarity = []
        for i in range(self.meta_num):
            self.similarity.append(0.75)

    def preComputeSimilarity(self, test_insts):
        test_insts_ = MetaInsts([])
        test_insts_.numValues = self.hp_num
        for inst in test_insts:
            test_insts_.add(MetaInstDense(inst.target, inst._values[:self.hp_num]))
        self.cached_predictions = [{} for _ in range(self.meta_num)]
        for d in range(self.meta_num):
            for i in range(len(test_insts)):  # \hat{f}_D
                self.cached_predictions[d][test_insts[i]] = self.gps[d].predict(test_insts_[i])

    def train(self, insts):
        self.untouched_konw_test = insts
        self.data = MetaInsts([])
        self.data.numValues = self.hp_num
        for inst in insts:
            self.data.add(MetaInstDense(inst.target, inst._values[:self.hp_num]))
        self.model = GaussianProcessRegression(SEARDKernel(self.hp_num))
        self.model.train(self.data)
        for i in range(self.meta_num):
            self.similarity[i] = self.kernel(i, self.untouched_konw_test)

    def kendallTauCorrelation(self, index, untouched_konw_test):
        if untouched_konw_test is None or len(untouched_konw_test) < 2:
            return 0.75
        disordered_pairs = total_pairs = 0
        for i in range(len(untouched_konw_test)):
            for j in range(len(untouched_konw_test)):
                if (untouched_konw_test[i].target < untouched_konw_test[j].target != self.cached_predictions[index][
                    untouched_konw_test[i]] < self.cached_predictions[index][untouched_konw_test[j]]):
                    disordered_pairs += 1
                total_pairs += 1
        t = disordered_pairs / total_pairs / self.bandwidth
        return 0.75 * (1 - t * t) if t < 1 else 0

    def predict(self, inst):
        denominator = 0.75
        mu, sigma_2 = self.model.predictWithUncerten(MetaInstDense(inst.target, inst._values[:self.hp_num]))
        for d in range(self.meta_num):
            mu += self.cached_predictions[d][inst] * self.similarity[d]
            denominator += self.similarity[d]
        mu /= denominator
        if sigma_2 is np.inf:
            sigma_2 = 1000
        return mu, sigma_2