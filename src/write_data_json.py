import timeit
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numba

from models import TimeVaryingSLAPIR

ca_cases = pd.read_csv('../data/CA_covid_data/statewide_cases.csv')
daily_cases = ca_cases.groupby('date').newcountconfirmed.sum().to_frame().reset_index()
daily_cases.date = pd.to_datetime(daily_cases.date)

init_prop = [36e6,4000,12000,20000]

t_eval = np.arange(1,182)

data = {
    'init_props':init_prop,
    'number_time_points':len(t_eval),
    'number_parameters':30,
    'number_variables':30,
    't_start':0,
    'time_series':t_eval.tolist(),
    'incidence_data':daily_cases.newcountconfirmed[20:200].values.tolist()
}

with open('data.json','w') as fh:
    json.dump(data,fh)
