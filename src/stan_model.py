import timeit

import stan
import pandas as pd
import numpy as np

from models import get_transmission_matrix, StratifiedSEIR

with open('StratifiedSEIR.stan') as fh:
    stan_code = fh.read()


age_matrix = pd.read_csv('../data/Valle-2007_Table-2_Age-Transmission-Matrix.csv').drop('Age',axis=1).values
race_matrix = np.repeat(1/3,9).reshape((3,3))
trans_matrix = get_transmission_matrix(1.0,age_matrix,race_matrix)
n_groups = len(age_matrix)*len(race_matrix)
group_ratios = np.repeat(1/n_groups,n_groups)



stan_model = stan.build(stan_code, data=k)
