import pastas as ps

from gwsym import GwSym

sourcepath = 'etmgeg_260.txt'
rain = ps.read.read_knmi(sourcepath, variables='RH').series
# evap = ps.read.read_knmi('etmgeg_260.txt', variables='EV24').series

gwsym = GwSym()
head = gwsym.generate_head(rain=rain, Atrue=800, ntrue=1.1, atrue=50,
                           dtrue=20)
noise = gwsym.generate_noise(alpha=0.95, beta=0.9, head=head, noise_perc=0.1)

# myfig1 = gwsym.plot_series()
# myfig2 = gwsym.plot_noise_check()

ml = gwsym.pastas_model()
ml.plots.results()

par = gwsym.parameters()
print(par.transpose())

print(ml.residuals().var(), ml.observations().var())