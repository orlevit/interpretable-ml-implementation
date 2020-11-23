import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence as pdp

def plot_2d_pd(model, df, features, fig_size=(16, 6)):
  fig, axes = plt.subplots(ncols=len(features), figsize=fig_size)
  for ii, (feature, ax) in enumerate(zip(features, axes)):
    pplot = pdp(model, df, [feature], n_jobs=1, grid_resolution=100,percentiles=(0,1),ax=ax)
    xticks = np.int16(np.linspace(df[feature].min(), df[feature].max(),5))
    pplot.axes_[0, 0].set_xticks(xticks)
    pplot.axes_[0, 0].set_xlabel(f'feature: {feature[1:]}')
	
def plot_3d_pd(model, df, features, features_comb, grid_resolution=20,azim=120,fig_size=(20, 6)):
  fig = plt.figure(figsize=fig_size)
  all_comb = [i for i in features_comb]

  for ii, fc in enumerate(all_comb, 1):
    ax3d=fig.add_subplot(1, len(all_comb), ii, projection='3d')
    pdp, axes = partial_dependence(model, df, fc, grid_resolution=grid_resolution)
    XX, YY = np.meshgrid(axes[0], axes[1])
    Z = pdp[0].T
    surf = ax3d.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                          cmap=plt.cm.BuPu, edgecolor='k')
    ax3d.set_xlabel(fc[0])
    ax3d.set_ylabel(fc[1])
    ax3d.set_zlabel('Partial dependence')
    #  pretty init view
    ax3d.view_init(elev=22, azim=azim)
    plt.colorbar(surf)
    plt.suptitle('Partial dependence of bike sales\n'
                'on temapture, humidity and wind speed')
    plt.subplots_adjust(top=0.9)

