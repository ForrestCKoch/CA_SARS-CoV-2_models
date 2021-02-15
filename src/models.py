import math
import itertools as it

import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import scipy as sc
import scipy.integrate as scint 
import scipy.stats as scstats

def get_transmission_matrix(beta,X,Y):
    Z = np.array(list(it.product(X.flatten(),Y.flatten()))).prod(axis=1)
    return beta * Z.reshape(np.repeat(len(X)*len(Y),2))

@njit(cache=True)
def get_infected(t, beta_start, beta_end, k, m, V):
    N = V.sum()
    beta_eff = beta_end + (beta_start-beta_end)/(1+np.exp(-k*(m-t)))
    return beta_eff * V[0] * (0.2 * V[1:17].sum() + 1.2*V[17:19].sum() + V[19:29].sum())/N

@njit(cache=True)
def jit_odes(t, V, beta_start, beta_end, k, m):

    dV = np.zeros(V.shape)
    infected = get_infected(t, beta_start, beta_end, k, m, V)

    dV[0] = -infected # S
    dV[1] = infected - V[1] # L1
    dV[2:5] = V[1:4] - V[2:5] #L2-4
    dV[5] = 0.2 * V[4] - V[5] # A1
    dV[6:17] = V[5:16] - V[6:17] #A2-12
    dV[17] = 0.8 * V[4] - V[17] # P1
    dV[18:29] = V[17:28] - V[18:29] # P2,I1-10
    dV[29] = V[16] + V[28] # R
    return dV

class TimeVaryingSLAPIR(object):

    def __init__(self, t_eval, beta_start, beta_end, k, m, init):

        self.t_eval = t_eval
        self.t_span = (np.min(t_eval),np.max(t_eval))
            
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.m = m
        self.k = k

        self.init = init

    def _check_params(self):
        pass

    def jit_solve(self,method='RK45'):
        """
        Return the solution provided by scipy.integrate.solve_ivp
        """
        f = lambda t,V: jit_odes(t, V, self.beta_start, self.beta_end, self.k, self.m)

        return scint.solve_ivp(fun=f, t_span=self.t_span, y0=self.init, t_eval=self.t_eval,method=method)

    def plot_compartment(self, comp=0, label='Susceptible'):
        soln = self.jit_solve().y
        comp_sum = soln[(comp),:].sum(axis=0)
        plt.plot(comp_sum.T,label=label)

    def plot_group(self, group='S'):
        soln = self.jit_solve().y
        if group == 'S':
            x = soln[0,:].T
            label = 'Susceptilbe'
        elif group == 'L':
            x = soln[1:5,:].sum(axis=0).T
            label = 'Latent'
        elif group == 'A':
            x = soln[5:17,:].sum(axis=0).T
            label = 'Asymptomatic'
        elif group == 'P':
            x = soln[17:19,:].sum(axis=0).T
            label = 'Pre-symptomatic'
        elif group == 'I':
            x = soln[19:29,:].sum(axis=0).T
            label = 'Infectious'
        else:
            x = soln[29,:].T
            label = 'Recovered'

        plt.plot(x,label=label)
        plt.ylim(bottom=0)

    def plot_incidence(self):
        soln = self.jit_solve().y
        plt.plot(soln[18,:], label='incidence')
        

    def copy(self):
        return None

    def to_dict(self):
        return {}

    def __repr__(self):
        return self.to_dict().__repr__()

    def __str__(self):
        return self.to_dict().__str__()
