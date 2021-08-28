import numpy as np
from scipy import stats

class EI(object):
    def __init__(self):
        self.xi = 0.1

    def getEI(self, mu, sigma, y_best):
        z = (-y_best + mu - self.xi) / sigma
        ei = (-y_best + mu -
              self.xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        return ei

    def getNext(self, y_best, surrogate, candidates):
        best_ei = -1
        best_candidate = []
        for candidate in candidates:
            y_hat = surrogate.predict(candidate)
            ei = self.getEI(y_hat[0], y_hat[1], y_best)
            if ei > best_ei:
                best_ei = ei
                best_candidate = [candidate]
            elif ei == best_ei:
                best_candidate.append(candidate)
        return np.random.choice(best_candidate)
