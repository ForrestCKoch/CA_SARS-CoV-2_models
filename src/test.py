import timeit

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numba

from models import get_transmission_matrix, StratifiedSEIR

#age_matrix = pd.read_csv('../data/Valle-2007_Table-2_Age-Transmission-Matrix.csv').drop('Age',axis=1).values
#race_matrix = np.repeat(1/3,9).reshape((3,3))
#trans_matrix = get_transmission_matrix(1.0,age_matrix,race_matrix)

#n_groups = len(age_matrix)*len(race_matrix)
#group_ratios = np.repeat(1/n_groups,n_groups)

ca_cases = pd.read_csv('../data/CA_covid_data/statewide_cases.csv')
daily_cases = ca_cases.groupby('date').newcountconfirmed.sum().to_frame().reset_index()
daily_cases.date = pd.to_datetime(daily_cases.date)

trans_matrix = np.matrix([[1,1],[1,1.0]])
group_ratios = np.array([0.5,0.5])

seir = StratifiedSEIR(trans_matrix,group_ratios,np.arange(0,150),
        alpha=1, beta=.45, beta2=.12, gamma=1/7, eta=1/3,
        S0=38e6, E0=600, I0=6000, R0=2e6, k=0.032, x0=.039)

"""
print('RK23: {}'.format(timeit.timeit(lambda: seir.solve(method='RK23'),number=1000)))
print('DOP853: {}'.format(timeit.timeit(lambda: seir.solve(method='DOP853'),number=1000)))
print('Radau: {}'.format(timeit.timeit(lambda: seir.solve(method='Radau'),number=1000)))
print('BDF: {}'.format(timeit.timeit(lambda: seir.solve(method='BDF'),number=1000)))
print('LSODA: {}'.format(timeit.timeit(lambda: seir.solve(method='LSODA'),number=1000)))
"""
#print('Baseline: {}'.format(timeit.timeit(seir.solve,number=1000)))
#print('Jit: {}'.format(timeit.timeit(seir.jit_solve,number=1000)))

#seir.plot_compartment(comp=0)
#seir.plot_compartment(comp=1,label='Exposed')
#seir.plot_compartment(comp=2,label='Infectious')
#seir.plot_compartment(comp=3,label='Recovered')
seir.plot_incidence()
plt.plot(np.arange(0,150),daily_cases.newcountconfirmed[50:200],label='observed')
plt.legend()
plt.tight_layout()
plt.show()
