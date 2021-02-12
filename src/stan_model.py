import timeit

import stan
import pandas as pd
import numpy as np

from models import get_transmission_matrix, StratifiedSEIR

age_matrix = pd.read_csv('../data/Valle-2007_Table-2_Age-Transmission-Matrix.csv').drop('Age',axis=1).values
race_matrix = np.repeat(1/3,9).reshape((3,3))
trans_matrix = get_transmission_matrix(1.0,age_matrix,race_matrix)
n_groups = len(age_matrix)*len(race_matrix)
group_ratios = np.repeat(1/n_groups,n_groups)

ca_cases = pd.read_csv('../data/CA_covid_data/statewide_cases.csv') 
daily_cases = ca_cases.groupby('date').newcountconfirmed.sum().to_frame().reset_index() 
daily_cases.date = pd.to_datetime(daily_cases.date)

with open('StratifiedSEIR.stan','r') as fh:
    stan_code = fh.read()

#stan_model = stan.build(stan_code, data=daily_cases.newcountconfirmed)
data = {
    "init_compartments": [39e9,600,600,0],
    "group_ratios": group_ratios,
    "n_groups": n_groups,
    "number_time_points": len(daily_cases.date),
    "number_parameters": 4,
    "number_variables": 4 * n_groups,
    "t_start": 0,
    "time_series": np.arange(0,314),
    "incidence_data": daily_cases.newcountconfirmed,
    "trans_matrix": trans_matrix.flatten()
}

stan_model = stan.build(stan_code, data=data)
fit = stan_model.sample(num_chains=4,num_samples=1000)
