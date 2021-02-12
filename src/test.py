import timeit

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numba

from models import get_transmission_matrix, StratifiedSEIR

age_matrix = pd.read_csv('../data/Valle-2007_Table-2_Age-Transmission-Matrix.csv').drop('Age',axis=1).values
race_matrix = np.repeat(1/3,9).reshape((3,3))
trans_matrix = get_transmission_matrix(1.0,age_matrix,race_matrix)

n_groups = len(age_matrix)*len(race_matrix)
group_ratios = np.repeat(1/n_groups,n_groups)

seir = StratifiedSEIR(trans_matrix,group_ratios,np.arange(0,100),
        0.1,0.22,1/7,1/3)

"""
print('RK23: {}'.format(timeit.timeit(lambda: seir.solve(method='RK23'),number=1000)))
print('DOP853: {}'.format(timeit.timeit(lambda: seir.solve(method='DOP853'),number=1000)))
print('Radau: {}'.format(timeit.timeit(lambda: seir.solve(method='Radau'),number=1000)))
print('BDF: {}'.format(timeit.timeit(lambda: seir.solve(method='BDF'),number=1000)))
print('LSODA: {}'.format(timeit.timeit(lambda: seir.solve(method='LSODA'),number=1000)))
"""
#print('Baseline: {}'.format(timeit.timeit(seir.solve,number=1000)))
#print('Jit: {}'.format(timeit.timeit(seir.jit_solve,number=1000)))

"""
seir.plot_compartment(comp=0)
seir.plot_compartment(comp=1,label='Exposed')
seir.plot_compartment(comp=2,label='Infectious')
seir.plot_compartment(comp=3,label='Recovered')
plt.legend()
plt.tight_layout()
plt.show()
"""
seir.plot_group(0)
plt.show()
