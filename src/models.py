import math
import itertools as it

import numpy as np
import scipy as sc
import scipy.integrate as scint 
import scipy.stats as scstats

def get_transmission_matrix(beta,X,Y):
    Z = np.array(list(it.product(X.flatten(),Y.flatten()))).prod(axis=1)
    return beta * Z.reshape(np.repeat(len(X)*len(Y),2))

class StratifiedSEIR(object):

    def __init__(self, trans_matrix, group_ratios, t_eval, alpha, beta, gamma, eta, 
            S0=(1-1e-4), E0=1e-4, I0=0, R0=0):

        self.trans_matrix = trans_matrix
        self.group_ratios = group_ratios
        self.t_eval = t_eval
        self.t_span = (np.min(t_eval),np.max(t_eval))
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0

        self.n_comps = 4
        self.n_groups = trans_matrix.shape[0]

    def _check_params(self):
        pass

    def odes(self, t, V):
        """
        Implement the ODEs in a function that can be used by scipy.integrate.solve_ivp

        :param t: the timepoint (not used)
        :param V: The current state, with shape (compartments, groups)
        """
        N = V.sum(axis=0)
        dV = np.zeros(V.shape)

        s_end = self.n_groups
        e_end = 2 * s_end
        i_end = s_end + e_end # 3 * self.n_groups, but more efficient...

        exposed = V[0:s_end,:] * ((self.trans_matrix @ (self.alpha * V[s_end:e_end,:] + V[e_end:i_end,:]))/N)
        #exposed = V[0:s_end,:] * ((self.trans_matrix @ (self.alpha * V[s_end:e_end,:] + V[e_end:i_end,:])))
        infectious = self.eta * V[s_end:e_end,:]
        recovered = self.gamma * V[e_end:i_end,:]

        dV[0:s_end,:] = -1 * exposed
        dV[s_end:e_end,:] = exposed - infectious
        dV[e_end:i_end,:] = infectious - recovered
        dV[i_end:,:] = recovered

        #print('N:{},E:{}'.format(str(N.shape),str(exposed.shape)))

        return dV

    def solve(self,method='RK45'):
        """
        Return the solution provided by scipy.integrate.solve_ivp
        """
        # Note the order of the arguments is important here
        #print(self.group_ratios)
        V0 = np.array(list(it.product([self.S0,self.E0,self.I0,self.R0],self.group_ratios))).prod(axis=1)

        return scint.solve_ivp(fun=self.odes, t_span=self.t_span, y0=V0, t_eval=self.t_eval,vectorized=True,method=method)

    def copy(self):
        return None

    def to_dict(self):
        return {}

    def __repr__(self):
        return self.to_dict().__repr__()

    def __str__(self):
        return self.to_dict().__str__()
