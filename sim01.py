import numpy as np
import pandas as pd
import pastas as ps
from pandas import DataFrame

from gwsym import GwSym

ps.set_log_level("ERROR")

# define simulation cases with varying alpha and beta
alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
beta_list = np.arange(-0.9, 1., 0.1).round(2)

cases = []
for alpha in alpha_list:
    for beta in beta_list:
        case = {'Atrue': 800, 'ntrue': 1.1, 'atrue': 50, 'dtrue': 20,
                'alpha': alpha, 'beta': beta}
        cases.append(case)

cases = DataFrame(cases)
casenames = [f'case{nr}' for nr in range(len(cases))]
cases = cases.set_index([casenames])
cases.index.name = 'casename'
cases.to_csv('cases.csv', index=False)

# simulate deterministic head
rain = ps.read.read_knmi('etmgeg_260.txt', variables='RH').series

sims = []
tests = []
figdir = "fig//"

# Go through all cases
for idx, cs in cases.iterrows():
    print(idx)
    gwsym = GwSym()
    head = gwsym.generate_head(rain=rain, Atrue=cs['Atrue'], ntrue=cs['ntrue'],
                               atrue=cs['atrue'], dtrue=cs['dtrue'])

    gwsym.generate_noise(alpha=cs['alpha'], beta=cs['beta'], head=head)

    #gwsym.plot_noise_check(figdir=figdir)
    gwsym.pastas_model(figdir=figdir)

    # figdir = '..\\03_result\\sim01\\simulated_series\\'
    # myfig1 = gwsym.plot_series(figdir=figdir)

    par = gwsym.parameters()
    par.index = [idx for row in range(len(par))]
    sims.append(par)

    # add test statistics to list
    tst = gwsym.test_statistics()
    tst.insert(0, 'Test', tst.index.values)
    tst.index = [idx for row in range(len(tst))]
    tests.append(tst)

sims = pd.concat(sims, axis=0)
sims.index.name = 'casename'

sims.to_csv('simulations.csv', index=True)

tests = pd.concat(tests, axis=0)
tests.index.name = 'casename'
tests.to_csv('test_statistics.csv', index=True)

# alpha = sims['alpha'].values
# Aest = sims['A_est'].values
# plt.scatter(alpha,Aest)
