import scipy
import numpy as np
import statsmodels.api as sm
from tabulate import tabulate
from matplotlib import pyplot as plt

def anova_table(x, y, plot_graph=False, x_plot_loc=0):
    """
    Author: Or Levitas.
    Description: Calculating ANOVA by fitiing OLS.
                 The first column must be the number '1'
    """
    
    model = sm.OLS(y,x).fit()
    predictions = model.predict(x) # make the predictions by the model

    if plot_graph:
        fig, ax = plt.subplots()
        fig = sm.graphics.plot_fit(model, x_plot_loc, ax=ax, vlines=False)
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        ax.set_title("Linear Regression")
        plt.show()
    
    #Print out the statistics
    print('\n***********************************************************************************************\n')
    print(f'The model statistics: {model.summary()}')
    print('\n***********************************************************************************************\n')



    # Sum of squares
    y_mean = np.average(y)
    SSR = round(sum(np.square(y_mean - predictions)),2)
    SSE = round(sum(np.square(predictions - y)),2)
    SST = round(sum(np.square(y_mean - y)),2)


    # unique values
    c = np.unique(x[:,1:]) 

    # SSLF & SSPE calculation
    SSLF = 0
    SSPE = 0
    for val in c:
        locs = np.where(x[:,1] == val)
        x_values = x[locs,:]
        y_values = y[locs]
        y_preds = predictions[locs]
        y_avg = np.average(y[locs])
        SSLF += sum(np.square(y_preds - y_avg))
        SSPE += sum(np.square(y_avg - y_values))


    SSLF = round(SSLF,2)
    SSPE = round(SSPE,2)


    # degrees of freedom
    dof_regression = x.shape[1] - 1
    dof_error = x.shape[0] - x.shape[1]
    dof_lack_of_fit = len(c) - x.shape[1]
    dof_pure_error = x.shape[0] - len(c)
    dof_total = x.shape[0] - (x.shape[1] - 1)


    MSR = round(SSR/dof_regression,2)
    MSE = round(SSE/dof_error,2)
    MSLF = round(SSLF/dof_lack_of_fit,2)
    MSPE = round(SSPE/dof_pure_error,2)

    F1 = round(MSR/MSE,2)
    F2 = round(MSLF/MSPE,2)


    # F test
    p_vale_f1 = round(scipy.stats.f.sf(F1,dof_regression,dof_error),2)
    p_vale_f2 = round(scipy.stats.f.sf(F2,dof_lack_of_fit,dof_pure_error),2)


    table = [['Regression', dof_regression, f'SSR={SSR}', f'MSR={MSR}', f'MSR/MSE={F1}', f'{p_vale_f1}'],\
    ['Residual Error',dof_error, f'SSE={SSE}', f'MSE={MSE}'],\
    ['Lack of Fit',dof_lack_of_fit, f'SSLF={SSLF}', f'MSLF={MSLF}', f'MSLF/MSPE={F2}', f'{p_vale_f2}'],\
    ['Pure Error',dof_pure_error ,f'SSPE={SSPE}',f'MSPE={MSPE}'],\
    ['Total',dof_total, f'SST={SST}']]


    headers = ['Source','DF','SS','MS','F','P value']
    print('\n                                       ANOVA Table\n')
    print(tabulate(table, headers, tablefmt="github",stralign='left'))
    print('\nGeneral rules:')
    print('*SSR + SSE = SST')
    print('*SSE = SSLF + SSPE')
    print('*SSE = SSLF + SSPE')
    print('*Hypothesis F(MSR/MSE) - slope = 0')
    print('*Hypothesis F(MSLF/MSPE) - linear != 0')


    #WHy is the value 199 so big(different form 1), but yet the F value is 0 - meaning beta1 different than 0?

