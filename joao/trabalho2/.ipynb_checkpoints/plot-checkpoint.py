import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import statsmodels.tsa.stattools as smt
import numpy as np
from itertools import combinations,permutations

def lines_plot(df, stations, freq = 'M', figsize = (20,5), tittle = ''):
    ls = ['date_time'] + stations
    temp = df[ls]
    temp.index = temp.date_time
    temp = temp.resample(freq).mean()
    temp.reset_index(inplace=True)
    plt.figure(figsize = figsize)
    _ = sns.lineplot(x = temp['date_time'],y = temp[stations[0]]).set_title(tittle,fontsize=20)
    for i in stations[1:]:
        _ = sns.lineplot(x = temp['date_time'],y = temp[i])
        
def columnX(code): 
    return f'{code} - temperatura maxima na hora ant. (aut) (°c)'

def plot_ccf_sm(target, exog, unbiased=False, nlags=72):
    """Plot CCF using Statsmodels"""
    ccfs = smt.ccf(target, exog, adjusted=False)[:nlags+1]
    lags = np.arange(len(ccfs))[:nlags+1]
    _ = plt.stem(lags, ccfs, use_line_collection=True)
    _ = plt.title(f"Cross Correlation (Statsmodels): {target.name} & {exog.name}")
    _ = plt.show()

def cross_plots(df):
    ls = list(permutations(range(df.shape[1]),2))
    for i in ls:
        plot_ccf_sm(df.iloc[:,i[0]],df.iloc[:,i[1]])  