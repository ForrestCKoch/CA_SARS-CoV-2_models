import numpy as np
import pandas as pd

def get_reproduction_number(beta=0.115,S=1):

    F = np.zeros(shape=(28,28))
    F[0,0:16] = 0.2*beta*S
    F[0,16:18] = 1.2*beta*S
    F[0,18:] = beta * S


    V = np.eye(28)
    for i in range(1,28):
        v = -1
        if i == 4:
            v = -0.2
            V[i,i-1] = v
        elif i == 16:
            v = -0.8
            V[i,3] = v
        else:
            V[i,i-1] = v

    w,v = np.linalg.eigh(F @ np.linalg.inv(V))
    return w.max()

if __name__ == '__main__':
    ca_cases = pd.read_csv('../data/CA_covid_data/statewide_cases.csv')
    daily_cases = ca_cases.groupby('date').newcountconfirmed.sum().to_frame().reset_index()
    daily_cases.date = pd.to_datetime(daily_cases.date)
    S_20 = (36e6-daily_cases.newcountconfirmed[:20].sum())/36e6
    S_200 = (36e6-daily_cases.newcountconfirmed[:20].sum())/36e6
    print('R0: {}'.format(get_reproduction_number(beta=.1155)))
    print('R20: {}'.format(get_reproduction_number(beta=.115,S=S_20)))
    print('R200: {}'.format(get_reproduction_number(beta=.08165,S=S_200)))

