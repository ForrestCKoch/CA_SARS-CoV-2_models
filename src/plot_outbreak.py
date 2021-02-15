import timeit
import json

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numba

from models import TimeVaryingSLAPIR
from basic_reproduction_number import get_reproduction_number

def get_beta_eff(t,s,e,k,m):
    return e+(s-e)/(1+np.exp(-k*(m-t)))

beta_start = 1.1547e-01
beta_end = 8.1647e-02
k = 1.4741e-01

ca_cases = pd.read_csv('../data/CA_covid_data/statewide_cases.csv')
daily_cases = ca_cases.groupby('date').newcountconfirmed.sum().to_frame().reset_index()
daily_cases.date = pd.to_datetime(daily_cases.date)

init_prop = [36e6,4000,12000,20000]
init = np.zeros(30)
init[0] = init_prop[0]
init[1:5] = np.repeat(init_prop[1]/4,4)
init[5:17] = np.repeat(0.2*init_prop[2]/12,12)
init[17:29] = np.repeat(0.8*init_prop[2]/12,12)
init[29] = init_prop[3]

t_eval = np.arange(0,180)

inference_data = az.from_cmdstan('../results/outputs/*.csv')
chains = [i for i in range(18)]
samples = [i for i in range(20000)]
incidence = []
for i in range(5000):
    chain = np.random.choice(chains)
    sample = np.random.choice(samples)
    beta_start = inference_data.posterior.data_vars['beta_start'][chain,sample].data
    beta_end = inference_data.posterior.data_vars['beta_end'][chain,sample].data
    k = inference_data.posterior.data_vars['k'][chain,sample].data
    seir = TimeVaryingSLAPIR(
            t_eval = t_eval, 
            beta_start = beta_start,
            beta_end = beta_end,
            k = k,
            m = 90,
            init = init
    )
    incidence.append(seir.jit_solve().y[18,:])

az.plot_hdi(t_eval, incidence, hdi_prob=0.95)
plt.plot(np.arange(0,180),daily_cases.newcountconfirmed[20:200])
plt.xlabel('Time (days)')
plt.ylabel('Incidence')
plt.tight_layout()
#plt.show()
plt.savefig('../results/plots/model_fit.pdf')
