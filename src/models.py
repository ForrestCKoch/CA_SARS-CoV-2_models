import math
import itertools as it

import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pymc3 as pm
import scipy as sc
import scipy.integrate as scint 
import scipy.stats as scstats

def get_transmission_matrix(beta,X,Y):
    Z = np.array(list(it.product(X.flatten(),Y.flatten()))).prod(axis=1)
    return beta * Z.reshape(np.repeat(len(X)*len(Y),2))

"""
@njit(cache=True)
def jit_odes(V, trans_matrix, alpha, beta, gamma, eta, n_groups):
    N = V.sum(axis=0)
    dV = np.zeros(V.shape)

    s_end = n_groups
    e_end = 2 * s_end
    i_end = s_end + e_end # 3 * self.n_groups, but more efficient...

    exposed = beta * V[0:s_end] * ((trans_matrix @ (alpha * V[s_end:e_end] + V[e_end:i_end]))/N)
    infectious = eta * V[s_end:e_end]
    recovered = gamma * V[e_end:i_end]

    dV[0:s_end] = -1 * exposed
    dV[s_end:e_end] = exposed - infectious
    dV[e_end:i_end] = infectious - recovered
    dV[i_end:] = recovered
    return dV
"""
@njit(cache=True)
def jit_odes(t, V, trans_matrix, alpha, beta, gamma, eta, n_groups, beta2, k, x0):
    N = V.sum(axis=0)
    dV = np.zeros(V.shape)

    s_end = n_groups
    e_end = 2 * s_end
    i_end = s_end + e_end # 3 * self.n_groups, but more efficient...

    b = (beta-beta2)/(1+np.exp(-k*(x0-t))) + beta2

    """
    exposed = b * V[0:s_end] * ((trans_matrix @ (alpha * V[s_end:e_end] + V[e_end:i_end]))/N)
    infectious = eta * V[s_end:e_end]
    recovered = gamma * V[e_end:i_end]

    dV[0:s_end] = -1 * exposed
    dV[s_end:e_end] = exposed - infectious
    dV[e_end:i_end] = infectious - recovered
    dV[i_end:] = recovered
    """

    exposed = 0
    infectious = b * V[0:s_end] * ((trans_matrix @ (V[e_end:i_end]))/N)
    #infectious = eta * V[s_end:e_end]
    recovered = gamma * V[e_end:i_end]

    dV[0:s_end] = -1 * infectious
    dV[s_end:e_end] = exposed
    dV[e_end:i_end] = infectious - recovered
    dV[i_end:] = recovered
    return dV

class StratifiedSEIR(object):

    def __init__(self, trans_matrix, group_ratios, t_eval, alpha, beta, gamma, eta, 
            k, x0,
            S0=(1-1e-4), E0=1e-4, I0=0, R0=0, beta2=0):

        self.trans_matrix = trans_matrix
        self.group_ratios = group_ratios
        self.t_eval = t_eval
        self.t_span = (np.min(t_eval),np.max(t_eval))
        self.alpha = alpha
        self.beta = beta
        self.beta2 = beta2
        self.gamma = gamma
        self.eta = eta
        self.S0 = S0
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0

        self.k = k
        self.x0 =x0

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

        exposed = V[0:s_end] * ((self.trans_matrix @ (self.alpha * V[s_end:e_end] + V[e_end:i_end]))/N)
        #exposed = V[0:s_end,:] * ((self.trans_matrix @ (self.alpha * V[s_end:e_end,:] + V[e_end:i_end,:])))
        infectious = self.eta * V[s_end:e_end]
        recovered = self.gamma * V[e_end:i_end]

        dV[0:s_end] = -1 * exposed
        dV[s_end:e_end] = exposed - infectious
        dV[e_end:i_end] = infectious - recovered
        dV[i_end:] = recovered

        #print('N:{},E:{}'.format(str(N.shape),str(exposed.shape)))

        return dV

    def solve(self,method='RK45'):
        """
        Return the solution provided by scipy.integrate.solve_ivp
        """
        # Note the order of the arguments is important here
        #print(self.group_ratios)
        V0 = np.array(list(it.product([self.S0,self.E0,self.I0,self.R0],self.group_ratios))).prod(axis=1)

        return scint.solve_ivp(fun=self.odes, t_span=self.t_span, y0=V0, t_eval=self.t_eval,method=method)

    def jit_solve(self,method='RK45'):
        """
        Return the solution provided by scipy.integrate.solve_ivp
        """
        # Note the order of the arguments is important here
        #print(self.group_ratios)
        V0 = np.array(list(it.product([self.S0,self.E0,self.I0,self.R0],self.group_ratios))).prod(axis=1)
        f = lambda t,V: jit_odes(t, V, self.trans_matrix, self.alpha, self.beta, self.gamma, self.eta, self.n_groups, self.beta2, self.k, self.x0)

        return scint.solve_ivp(fun=f, t_span=self.t_span, y0=V0, t_eval=self.t_eval,method=method)

    def plot_compartment(self, comp=0, label='Susceptible'):
        soln = self.jit_solve().y
        comp_sum = soln[(comp*self.n_groups):((comp+1)*self.n_groups),:].sum(axis=0)
        plt.plot(comp_sum.T,label=label)
        plt.xlabel('Time')
        plt.ylabel('Population')

    def plot_group(self, group=0, label=''):
        soln = self.jit_solve().y
        susc = soln[group,:].T
        expo = soln[(self.n_groups+group),:].T
        infe = soln[(2*self.n_groups+group),:].T
        reco = soln[(3*self.n_groups+group),:].T
        plt.plot(susc, label='Susceptible')
        plt.plot(expo, label='Exposed')
        plt.plot(infe, label='Infectious')
        plt.plot(reco, label='Recoverd')
        plt.ylim(bottom=0)
        plt.title(label)
        plt.legend()
        plt.show()

    def plot_incidence(self):
        soln = self.jit_solve().y
        incd = -np.diff(soln[0,:]).T
        plt.plot(incd, label='incidence')
        

    def copy(self):
        return None

    def to_dict(self):
        return {}

    def __repr__(self):
        return self.to_dict().__repr__()

    def __str__(self):
        return self.to_dict().__str__()
