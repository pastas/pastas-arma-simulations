import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

srcpath = 'simulations.csv'
sim = pd.read_csv(srcpath)

srcpath = 'test_statistics.csv'
stats = pd.read_csv(srcpath)

fig1 = plt.figure()
grd1 = sim.pivot("alpha", "beta", "A_est")
ax1 = sns.heatmap(grd1)
ax1.set_title('A_est (Atrue=800)')
ax1.invert_yaxis()

plt.savefig("fig/results1.png", dpi=300, bbox_inches="tight")

fig1 = plt.figure()
mask = (grd1 < 750) | (grd1 > 850)
grd2 = grd1[mask]
ax2 = sns.heatmap(grd2)
ax2.set_title('A_est (A_true=800)')
ax2.invert_yaxis()
plt.savefig("fig/results2.png", dpi=300, bbox_inches="tight")


fig, [ax3, ax4] = plt.subplots(1, 2, figsize=[12, 6])
sim['dalpha'] = (sim['alpha_est'] - sim['alpha']) / sim['alpha'] * 100.
sim['dbeta'] = (sim['beta_est'] - sim['beta']) / sim['beta'] * 100.

grd3 = sim.pivot("alpha", "beta", "dalpha")
mask3 = (grd3 > -10) & (grd3 < 10)
sns.heatmap(grd3, ax=ax3, mask=mask3, vmax=100.)
ax3.set_title('deviation of alpha_est (%)\n (only deviations >10% are shown)')
ax3.invert_yaxis()
#
grd4 = sim.pivot("alpha", "beta", "dbeta")
mask4 = (grd4 > -10) & (grd4 < 10)
sns.heatmap(grd4, ax=ax4, mask=mask4, cbar=True, vmax=100.,
            vmin=-100, robust=True)
ax4.set_title('deviation of beta_est (%)\n (only deviations >10% are shown)')
ax4.invert_yaxis()
plt.savefig("fig/results3.png", dpi=300, bbox_inches="tight")

if False:
    shapiroo = stats[stats['Test'] == 'Shapiroo']['Reject H0'].values
    sim['shapiroo'] = np.where(shapiroo, 1, 0)

    agostino = stats[stats['Test'] == 'D\'Agostino']['Reject H0'].values
    sim['agostino'] = np.where(agostino, 1, 0)

    fig = plt.figure(figsize=[12, 6])
    ax1, ax2 = fig.subplots(1, 2, sharey=True)

    bluered = sns.color_palette(["#6666ff", "#ff3333"])

    grd1 = sim.pivot('alpha', 'beta', 'shapiroo')
    sns.heatmap(grd1, ax=ax1, cmap=bluered)
    ax1.set_title('Shapiroo normality test on innovations\n(0=H0 not '
                  'rejected, 1=H0 rejected)')

    grd2 = sim.pivot('alpha', 'beta', 'agostino')
    sns.heatmap(grd2, ax=ax2, cmap=bluered)
    ax2.set_title('D\'Agostino normality test on innovations\n(0=H0 not '
                  'rejected, 1=H0 rejected)')

    # heatmap of autocorrelation

    runs = stats[stats['Test'] == 'Runs test']['Reject H0'].values
    sim['runs'] = np.where(runs, 1, 0)

    ljungbox = stats[stats['Test'] == 'Ljung-Box']['Reject H0'].values
    sim['ljungbox'] = np.where(ljungbox, 1, 0)

    duwa = stats[stats['Test'] == 'Durbin-Watson']['Reject H0'].values
    sim['duwa'] = np.where(duwa, 1, 0)

    fig = plt.figure(figsize=[16, 6])
    ax1, ax2, ax3 = fig.subplots(1, 3, sharey=True)

    grd1 = sim.pivot('alpha', 'beta', 'runs')
    sns.heatmap(grd1, ax=ax1, cmap=bluered)
    ax1.set_title('Runs test on innovations\n(0=H0 not rejected, '
                  '1=H0 rejected)')

    grd2 = sim.pivot('alpha', 'beta', 'ljungbox')
    sns.heatmap(grd2, ax=ax2, cmap=bluered)
    ax2.set_title('Ljung-Box test on innovations\n(0=H0 not rejected, '
                  '1=H0 rejected)')

    grd3 = sim.pivot('alpha', 'beta', 'duwa')
    sns.heatmap(grd2, ax=ax3, cmap=bluered)
    ax3.set_title('Durbin Watson test on innovations\n(0=H0 not '
                  'rejected, 1=H0 rejected)')

    # heatmap of normality and autocorreletion

    sim['norm'] = np.where(shapiroo + agostino != 0, 1, 0)
    sim['autocor'] = np.where(runs + ljungbox + duwa != 0, 1, 0)

    fig = plt.figure(figsize=[12, 6])
    ax1, ax2 = fig.subplots(1, 2, sharey=True)

    grd1 = sim.pivot('alpha', 'beta', 'norm')
    sns.heatmap(grd1, ax=ax1, cmap=bluered)
    ax1.set_title(
        'normality of innovations\n(0=H0 not rejected, 1=H0 rejected)')

    grd2 = sim.pivot('alpha', 'beta', 'autocor')
    sns.heatmap(grd2, ax=ax2, cmap=bluered)
    ax2.set_title('autocrrelation of innovations\n(0=H0 not rejected, '
                  '1=H0 rejected)')
