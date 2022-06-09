from scipy.stats import qmc
from scipy.stats.qmc import Sobol, Halton, LatinHypercube


class RQuasiRandomSequence(qmc.QMCEngine):
        
    def __init__(self, d, seed=0.5):
        super().__init__(d=d)
        self.seed = float(seed)
        phi = 1.0
        # This is the Newton's method, solving phi**(d+1) - phi - 1 = 0
        while (np.abs(eq_check) > 1e-15):
            phi -= (phi**(self.d + 1) - phi - 1) / ((self.d + 1) * phi**self.d - 1)
            eq_check = phi**(self.d + 1) - phi - 1
        self.inv_phi = [phi**(-(1 + d)) for d in range(self.d)]
        
    def random(self, n=1):
        self.num_generated += n
        return (self.seed + np.arange(self.num_generated + 1, self.num_generated + n + 1)[:, None] * self.inv_phi) % 1.

    def reset(self):
        super().__init__(d=self.d, seed=self.seed)
        return self

    def fast_forward(self, n):
        self.random(n)
        return self

    
def get_qmc_engine(engine, **kwargs):
    
    return {'sobol': Sobol, 'halton': Halton, 'lhs': LatinHypercube, 'rqrs': RQuasiRandomSequence}.get(engine, engine)