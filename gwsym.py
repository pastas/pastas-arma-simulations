
import collections
import numpy as np
import matplotlib.pyplot as plt

from pandas import Series, DataFrame
import pandas as pd

from scipy.stats import norm
import pastas as ps


class GwSym:
    """Generate synthetic groundwater series with noise for simulations"""

    def __repr__(self):
        return (f'GwSyn object')

    def __init__(self):

        self.detpar = {}
        self.noisepar = {}
        self.head = None
        self.noise = None
        self.name = None


    def name(self,name=None):
        """Return name of GwSym object. If name is given, object is 
        renamed and new name returned"""
        if name is not None:
            self.name = name
        return self.name


    def generate_head(self,rain=None,Atrue=800,ntrue=1.1,atrue=200,dtrue=20):
        """ Generate the heads from rain with deterministic model 
        parameters """

        self.rain = rain
        self.detpar['Atrue'] = Atrue
        self.detpar['ntrue'] = ntrue
        self.detpar['atrue'] = atrue
        self.detpar['dtrue'] = dtrue

        # from Pastas notebook 15
        step = ps.Gamma().block([Atrue, ntrue, atrue])
        h = dtrue * np.ones(len(rain) + step.size)
        for i in range(len(rain)):
            h[i:i + step.size] += rain[i] * step
        head = pd.Series(index=rain.index, data=h[:len(rain)],name='head')

        ##head = head['1990':'2015']
        # ignore first ten years
        year = str(head.first_valid_index().year+10)
        head = head[f'{year}-01-01':].copy()
        
        self.head = head
        return head


    def generate_noise(self,alpha=0.7,beta=0.7, noise_perc=0.2,head=None):
        """Generate series of random distirbuted noise """

        self.noisepar['alpha'] = alpha
        self.noisepar['beta'] = beta

        if head is None:
            head = self.head

        # from Pastas notebook 15
        np.random.seed(1234)

        # generate samples using Numpy
        random_seed = np.random.RandomState(1234)
        n = len(head)
        innovation = random_seed.normal(0,1,n) * np.std(head.values) * noise_perc
        ##noise = np.zeros_like(head.values)
        noise = np.zeros(n)

        for i in range(1, n):
            # beta = theta, alpha = phi
            noise[i] = innovation[i] + innovation[i - 1] * beta + noise[i - 1] * alpha

        #head_noise = head[0] + noise
        self.noise = Series(noise,head.index)
        return self.noise


    def plot_series(self,figdir=None):
        """Plot head and noise"""

        fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1, figsize=(15,15))

        sr = self.head + self.noise
        sr.plot(ax=ax1,title='head+noise')
        self.head.plot(ax=ax2,title='head')
        self.noise.plot(ax=ax3,title='noise')

        title = self.model_name()
        fig.suptitle(title, fontsize=14)

        if figdir is not None:
            figname = self.model_name()
            figpath = f'{figdir}{figname}.jpg'
            fig.savefig(figpath)

        return fig


    def plot_noise_check(self,sr=None,figtitle=None,figdir=None):
        """Plot noise and nois histogram """

        if sr is None:
            sr = self.noise

        if figtitle is None:
            alpha = self.noisepar['alpha']
            beta = self.noisepar['beta']
            figtitle = 'alfa='+str(alpha)+' beta='+str(beta)

        fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

        sr.plot(ax=ax1,title=figtitle)
        sr.plot.hist(grid=False, bins=20, rwidth=0.9,color='#607c8e',ax=ax2, 
                     density=True, title=figtitle)

        # find minimum and maximum of xticks, so we know
        # where we should compute theoretical distribution
        xt = ax2.get_xticks()
        xmin, xmax = min(xt), max(xt)  
        lnspc = np.linspace(xmin, xmax, len(sr))

        # plot normal distribution
        m, s = norm.fit(sr) # get mean and standard deviation  
        pdf_g = norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
        ax2.plot(lnspc, pdf_g)

        if figdir is not None:
            figpath = f'{figdir}{figtitle}.jpg'
            fig.savefig(figpath)

        return fig


    def pastas_model(self,figdir=None):
        """Create and sove Pastas model using generated head and noise"""

        # create Pasytas model
        head_noise = self.head + self.noise
        self.ml = ps.Model(head_noise)
        self.sm = ps.StressModel(self.rain, ps.Gamma, name='recharge', settings='prec')
        self.ml.add_stressmodel(self.sm)
        self.ml.add_noisemodel(ps.ArmaModel())

        # solve Pastas model
        self.ml.solve(tmin="1991", tmax='2015-06-29', warmup=0, noise=True, report=False)

        if figdir is not None:
            # plot figure with model diagnostics

            axes = self.ml.plots.results(figsize=(10,5));
            fig = axes[0].get_figure()

            # add real step function to plot
            Atrue = self.detpar['Atrue']
            ntrue = self.detpar['ntrue']
            atrue = self.detpar['atrue']
            axes[-1].plot(ps.Gamma().step([Atrue, ntrue, atrue]))

            #figname = f'Atrue={Atrue} ntrue={ntrue} atrue={atrue}'
            figname = self.model_name()
            fig.suptitle(figname, fontsize=12)

            figpath = f'{figdir}Model {figname}.jpg'            
            fig.savefig(figpath)       

        return self.ml

    def model_name(self):
        """Return model name"""
        Atrue = self.detpar['Atrue']
        ntrue = self.detpar['ntrue']
        atrue = self.detpar['atrue']
        alpha = self.noisepar['alpha']
        beta = self.noisepar['beta']
        name = f'Atrue={Atrue} ntrue={ntrue} atrue={atrue} alpha={alpha} beta={beta}'
        return name

    def parameters(self):
        """Return table with true and estimated parameters"""

        par = collections.OrderedDict()

        par['A_true'] = self.detpar['Atrue']
        par['n_true'] = self.detpar['ntrue']
        par['a_true'] = self.detpar['atrue']
        par['alpha']  = self.noisepar['alpha']
        par['beta']   = self.noisepar['beta']

        sr = self.ml.parameters['optimal']                 
        par['A_est'] = sr['recharge_A']
        par['n_est'] = sr['recharge_n']
        par['a_est'] = sr['recharge_a']
        par['alpha_est'] = np.exp(-1./sr["noise_alpha"]).round(2)
        par['beta_est'] = np.exp(-1./sr["noise_beta"]).round(2)

        self.par = DataFrame([par])
        self.par.index.name = 'casename'

        return self.par


    def test_statistics(self):
        """Return test statistics for innovations"""
        self.test_stats = ps.stats.diagnostics(self.ml.noise(),nparam=2)
        self.test_stats.index.name = 'casename'
        return self.test_stats


if __name__ == '__main__':

    sourcepath = '..\\01_source\\etmgeg_260.txt'
    rain = ps.read.read_knmi(sourcepath, variables='RH').series
    #evap = ps.read.read_knmi('etmgeg_260.txt', variables='EV24').series

    gwsym = GwSym()
    name = gwsym.name('testcase')
    head = gwsym.generate_head(rain=rain,Atrue=800,ntrue=1.1,atrue=200,dtrue=20)
    noise = gwsym.generate_noise(alpha=0.7,beta=0.7,head=head)

    myfig1 = gwsym.plot_series(figdir='.\\')
    myfig2 = gwsym.plot_noise_check(figdir='.\\')

    ml = gwsym.pastas_model(figdir='.\\')
    par = gwsym.parameters()



