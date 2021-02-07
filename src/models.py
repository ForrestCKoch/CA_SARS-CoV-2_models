import math

import numpy as np
import scipy as sc
import scipy.integrate as scint 
import scipy.stats as scstats

class StratifiedSEIR(object):

    def __init__(self):
        pass

    def _check_params(self):
        pass

    def odes(self, t, prev):
        """
        Implement the ODEs in a function that can be used by scipy.integrate.solve_ivp

        :param t: the timepoint (not used)
        :param prev: the previous state, [S, E, I, A, R, V]
        """


        return None

    def solve(self):
        """
        Return the solution provided by scipy.integrate.solve_ivp
        """
        return scint.solve_ivp(fun=self.odes, t_span=self.t_span,
            y0=np.array([self.S0, self.E0, self.I0, self.A0, self.R0, self.V0]),
            t_eval=self.t_eval)

    def copy(self):
        return None

    def to_dict(self):
        return {}

    def __repr__(self):
        return self.to_dict().__repr__()

    def __str__(self):
        return self.to_dict().__str__()
