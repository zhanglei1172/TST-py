import numpy as np
import scipy



class SEARDKernel():
    def __init__(self, length):
        self.length = length
        self.initialize()

    def initialize(self):

        self.sigma_l = np.ones(self.length)
        self.sum_l = np.ones(self.length) * 0.001
        self.sum_f = 0.001
        self.sum_y = 0.001
        self.sigma_f = 1
        self.sigma_y = 0.001 # noise

    def getAATMinusKInverse(self, alpha, kernel):
        return np.dot(alpha.T, alpha)


    def computeKernel(self, insts):
        nums = len(insts.instances)
        kernel = np.zeros((nums, nums))
        for i in range(nums):
            for j in range(i, nums):
                kernel[i, j] = self.computeEntry(insts[i], insts[j])
                kernel[j, i] = kernel[i, j]
        return kernel

    def computeEntry(self, inst1, inst2):
        if (inst1 == inst2): # 对角元素
            return self.sigma_f**2 * self.computerExpPart(inst1, inst2) + self.sigma_y**2
        else:
            return self.sigma_f**2 * self.computerExpPart(inst1, inst2)

    def computerExpPart(self, inst1, inst2): # inst 对应文件一行数据
        keys1 = inst1.getKeys()
        keys2 = inst2.getKeys()
        values1 = inst1.getValues()
        values2 = inst2.getValues()
        idx1 = idx2 = 0
        z = 0
        while idx1 < len(keys1) or idx2 < len(keys2):
            if idx1 < len(keys1) and (idx2 >= len(keys2) or keys2[idx2]>keys1[idx1]):
                z += (values1[idx1])**2 / (self.sigma_l[keys1[idx1]])**2
                idx1 += 1
            elif idx2 <len(keys2) and (idx1> len(keys1) or keys1[idx1]>keys2[idx2]):
                z += (values2[idx2])**2 / (self.sigma_l[keys2[idx2]])**2
                idx2 += 1
            elif idx1 < len(keys1) and idx2 < len(keys2):
                z+= (values1[idx1] - values2[idx2])**2 / self.sigma_l[keys1[idx1]] **2
                idx1 += 1
                idx2 += 1
        return np.exp(-z / 2)


