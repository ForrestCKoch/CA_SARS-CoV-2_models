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
seir = TimeVaryingSLAPIR(
        t_eval = t_eval, 
        beta_start = .115,
        beta_end = 0.081,
        k = .15,
        m = 90,
        init = init
)

beta_eff = np.zeros(180)
for i in range(180):
    beta_eff[i] = get_beta_eff(i,beta_start,beta_end,k,90)

plt.plot(np.arange(0,180),beta_eff)
plt.xlabel('Time (days)')
plt.ylabel('Beta (effective)')
plt.savefig('../results/plots/beta_over_time.pdf')
plt.clf()

"""
seir.plot_incidence()
plt.plot(np.arange(0,180),daily_cases.newcountconfirmed[20:200],label='observed')
plt.legend()
plt.tight_layout()
plt.savefig('../results/plots/model_fit.pdf')
"""

inference_data = az.from_cmdstan('../results/outputs/*.csv')
az.plot_trace(inference_data)
plt.savefig('../results/plots/model_trace.pdf')
az.plot_joint(inference_data,var_names=['beta_start','beta_end'],kind='kde',figsize=(6,6))
plt.tight_layout()
plt.savefig('../results/plots/model_joint_betas.pdf')
az.plot_posterior(inference_data)
plt.savefig('../results/plots/model_posterior.pdf')
az.plot_autocorr(inference_data,combined=True)
plt.savefig('../results/plots/model_autocorrelation.pdf')
az.plot_rank(inference_data)
plt.savefig('../results/plots/model_rank.pdf')

