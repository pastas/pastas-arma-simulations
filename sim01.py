



import collections
import numpy as np
import matplotlib.pyplot as plt

from pandas import Series, DataFrame
import pandas as pd

from scipy.stats import norm
import pastas as ps

from gwsym import GwSym


if __name__ == '__main__':


    if 1: # define simulation cases

        # define simulation cases with varying alpha and beta
        alpha_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999]
        beta_list = [x/10 for x in list(range(-9,10,1))]

        cases = []
        for alpha in alpha_list:
            for beta in beta_list:

                case = {}
                case['Atrue'] = 800
                case['ntrue'] = 1.1
                case['atrue'] = 200
                case['dtrue'] = 20

                case['alpha'] = alpha
                case['beta'] = beta
                cases.append(case)

        cases = DataFrame(cases)
        casenames = [f'case{100+nr}' for nr in range(len(cases))]
        cases = cases.set_index([casenames])
        cases.index.name = 'casename'
        cases.to_csv('..\\02_data\\sim01\\cases.csv',index=False)

    if 0: # for testing only!

        cases = cases[:3]


    if 1: # perform simulations

        # simulate deterministic head
        rain = ps.read.read_knmi('..\\01_source\\etmgeg_260.txt', variables='RH').series


        sims = []
        tests = []
        for idx,cs in cases.iterrows():

            gwsym = GwSym()
            head = gwsym.generate_head(rain=rain,Atrue=cs['Atrue'],ntrue=cs['ntrue'],
                                       atrue=cs['atrue'],dtrue=cs['dtrue'])

            noise = gwsym.generate_noise(alpha=cs['alpha'],beta=cs['beta'],head=head)

            figdir = '..\\03_result\\sim01\\simulated_series\\'
            myfig1 = gwsym.plot_series(figdir=figdir)

            figdir = '..\\03_result\\sim01\\noise_check\\'
            myfig2 = gwsym.plot_noise_check(figdir=figdir)

            figdir = '..\\03_result\\sim01\\model_check\\'
            ml = gwsym.pastas_model(figdir=figdir)

            par = gwsym.parameters()
            par.index = [idx for row in range(len(par))]
            sims.append(par)

            # add test statistics to list
            tst = gwsym.test_statistics()
            tst.insert(0,'Test',tst.index.values)
            tst.index = [idx for row in range(len(tst))]
            tests.append(tst)

        sims = pd.concat(sims,axis=0)
        sims.index.name = 'casename'
        
        sims.to_csv('..\\02_data\\sim01\\simulations.csv',index=True)

        tests = pd.concat(tests, axis=0)
        tests.index.name = 'casename'
        tests.to_csv('..\\02_data\\sim01\\test_statistics.csv',index=True)

        #alpha = sims['alpha'].values
        #Aest = sims['A_est'].values
        #plt.scatter(alpha,Aest)

