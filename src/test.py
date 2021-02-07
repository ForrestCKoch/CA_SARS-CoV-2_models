import timeit

import pandas as pd
import numpy as np

from models import get_transmission_matrix, StratifiedSEIR

age_matrix = pd.read_csv('../data/Valle-2007_Table-2_Age-Transmission-Matrix.csv').drop('Age',axis=1).values
race_matrix = np.repeat(1/3,9).reshape((3,3))
trans_matrix = get_transmission_matrix(1.0,age_matrix,race_matrix)

n_groups = len(age_matrix)*len(race_matrix)
group_ratios = np.repeat(1/n_groups,n_groups)

seir = StratifiedSEIR(trans_matrix,group_ratios,np.arange(0,100),
        0.1,0.22,1/7,1/3)

#print(seir.solve())
print('RK45: {}'.format(timeit.timeit(lambda: seir.solve(method='RK45'),number=100)))
print('RK23: {}'.format(timeit.timeit(lambda: seir.solve(method='RK23'),number=100)))
print('DOP853: {}'.format(timeit.timeit(lambda: seir.solve(method='DOP853'),number=100)))
print('Radau: {}'.format(timeit.timeit(lambda: seir.solve(method='Radau'),number=100)))
print('BDF: {}'.format(timeit.timeit(lambda: seir.solve(method='BDF'),number=100)))
print('LSODA: {}'.format(timeit.timeit(lambda: seir.solve(method='LSODA'),number=100)))
