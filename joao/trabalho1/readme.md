# Descritiva dos dados de Neoplasia divididos em macroregiões

Aluno: João Paulo de Paiva Holz    
Professor: Ivan Robert Enriquez Guzman  
Materia: Laboratorio de estatistica   
Categoria de cid: Cid 91-95

### Importando as bibliotecas que iremos utilizar


```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import warnings
sns.set()
plt.rcParams['figure.figsize'] = [17, 6]
warnings.filterwarnings('ignore')
```

## Juntando os dados


```python
dfmenor1 = pd.read_csv('data/menor1ano.csv',';')
df1a4 = pd.read_csv('data/1a4anos.csv',';')
df5a9 = pd.read_csv('data/5a9.csv',';')
df10a15 = pd.read_csv('data/10a15.csv',';')

dfmenor1 = dfmenor1.replace('-',0)
dfmenor1['faixa'] = 'menor 1 ano'

df1a4['faixa'] = '1 a 4 anos'
df1a4 = df1a4.replace('-',0)

df5a9['faixa'] = '5 a 9 anos'
df5a9 = df5a9.replace('-',0)

df10a15['faixa'] = '10 a 14 anos'
df10a15 = df10a15.replace('-',0)


pop = pd.read_csv('data/popu.csv',';')
colnames = pop.iloc[:,0]
pop = pop.transpose().iloc[5:]
pop.columns = colnames
pop = pop.apply(pd.to_numeric)
pop = pop.reset_index()

df = pd.concat([dfmenor1,df1a4,df5a9,df10a15])
df = df.drop(df[df['Ano do Óbito']=='Total'].index) 
df = df.replace('-',0)

df['Ano do Óbito'] = pd.to_numeric(df['Ano do Óbito'])
df['SUL'] = pd.to_numeric(df['SUL'])
df['METROPOLITANA']=pd.to_numeric(df['METROPOLITANA'])
df['CENTRAL NORTE']=pd.to_numeric(df['CENTRAL NORTE'])
df['Total']=pd.to_numeric(df['Total'])

```

## Group By
Fazendo um simples group by vemos abaixo a tabela cruzada de faixa etaria x macrorregião.


```python
group = df.groupby(['faixa']).sum()
group.iloc[:,1:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SUL</th>
      <th>METROPOLITANA</th>
      <th>CENTRAL NORTE</th>
      <th>Total</th>
    </tr>
    <tr>
      <th>faixa</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1 a 4 anos</th>
      <td>13</td>
      <td>49</td>
      <td>25</td>
      <td>87</td>
    </tr>
    <tr>
      <th>10 a 14 anos</th>
      <td>19</td>
      <td>61</td>
      <td>29</td>
      <td>109</td>
    </tr>
    <tr>
      <th>5 a 9 anos</th>
      <td>12</td>
      <td>60</td>
      <td>20</td>
      <td>92</td>
    </tr>
    <tr>
      <th>menor 1 ano</th>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



---
Vemos que a região metropolitava consiste da maioria da população como já se é esperado


```python
_ = plt.pie(pop.iloc[-1,1:4],labels=['SUL', 'METROPOLITANA', 'CENTRAL NORTE'],autopct='%.0f%%')
_ = plt.title('Distribuição da população por macroregião',fontsize=25)
```


    
![png](images/output_8_0.png)
    


## Distribuição dos obitos por macroregião


```python
_ = plt.pie(df.sum()[1:4],labels=['SUL', 'METROPOLITANA', 'CENTRAL NORTE'],autopct='%.0f%%')
_ = plt.title('Distribuição dos óbitos por macroregião',fontsize=25)
```


    
![png](images/output_10_0.png)
    



```python
_ = plt.pie(group['Total'],labels=group.index,autopct='%.0f%%')
_ = plt.title('Distribuição dos óbitos por Faixa Etaria',fontsize=25)
```


    
![png](images/output_11_0.png)
    


Vemos que a maioria dos óbitos se encontra entre 10 e 14 anos. E vemos que menos de 1 ano é a menor proporção com apenas 5%. 
  
O que é um bom indício

---

# Series Historicas

Nas series historicas vemos a população do espirito santo chegando a quase 4 milhões de pessoas.
As series historicas dos Cids parecem estar estacionarias, onde nas idades mais baixas existem menos óbitos e nas idades mais altas vemos mais mortes.


```python
from get_pop import get_pop_by_faixa

def plot_lines(df,
               xcol = 'index',
               title = "População do espirito santo",
               xlab = "Ano",
               ylab = "População"):
    palette = plt.get_cmap('Set1')
    num=0
    for column in df.columns[df.columns!=xcol]:
        num+=1
        plt.plot(df[xcol], df[column], marker='o', color=palette(num), linewidth=1, alpha=0.9, label=column)
    plt.legend(loc=2, ncol=2)
    plt.title(title, loc='left', fontsize=25, fontweight=0, color='Black')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.yscale('linear')
    plt.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    plt.show()
```


```python
title = 'Casos de Neoplasia'
title = 'Taxa de ' + title
from get_pop import get_pop_by_faixa
def plot_lines_to_0a5():
    aaa = pd.read_csv('data/unificadomenor5.csv')
    aaa['ano'] = aaa.iloc[:,0]
    teste = aaa
    teste = teste.replace(np.NaN,0)
    ano = pd.DataFrame(np.arange(2000,2020))
    ano.columns = ['ano']
    denominador = pop.iloc[3:,:].iloc[:,1:4].apply(lambda x: x*get_pop_by_faixa()['0a4']).dropna()
    taxa = pd.concat([(teste.iloc[:,1:4]/denominador).iloc[:-1,:],ano],axis=1)*100_000
    plot_lines(taxa,xcol = 'ano',title= title + ' - 0 a 4 Anos')
    
def plot_lines_to_5a9():
    df_plot = df5a9.drop('faixa',axis=1)
    df_plot.index = df_plot['Ano do Óbito']
    df_plot = df_plot.drop('Ano do Óbito',axis=1)
    df_plot = df_plot.iloc[:-1,:]
    df_plot = df_plot.apply(pd.to_numeric)
    df_plot = df_plot.reset_index()
    df_plot = df_plot.apply(pd.to_numeric)
    df_plot2 = pd.DataFrame(np.arange(2000,2020))
    df_plot2.columns = ['ano']
    df_plot2 = df_plot2.apply(pd.to_numeric)
    teste = df_plot.merge(df_plot2,how='right',right_on='ano',left_on='Ano do Óbito')
    teste = teste.replace(np.NaN,0)
    ano = pd.DataFrame(np.arange(2000,2020))
    ano.columns = ['ano']
    denominador = pop.iloc[3:,:].iloc[:,1:4].apply(lambda x: x*get_pop_by_faixa()['5a9']).dropna()
    taxa = pd.concat([(teste.iloc[:,1:4]/denominador).iloc[:-1,:],ano],axis=1)*100_000
    #taxa = pd.concat([(teste.iloc[:,1:4]/pop.iloc[:,1:4]).iloc[:-1,:],ano],axis=1)
    plot_lines(taxa,xcol = 'ano',title= title + ' - 5 a 9 Anos')
    
def plot_lines_to_10a15():
    df_plot = df10a15.drop('faixa',axis=1)
    df_plot.index = df_plot['Ano do Óbito']
    df_plot = df_plot.drop('Ano do Óbito',axis=1)
    df_plot = df_plot.iloc[:-1,:]
    df_plot = df_plot.apply(pd.to_numeric)
    df_plot = df_plot.reset_index()
    df_plot = df_plot.apply(pd.to_numeric)
    df_plot2 = pd.DataFrame(np.arange(2000,2020))
    df_plot2.columns = ['ano']
    df_plot2 = df_plot2.apply(pd.to_numeric)
    teste = df_plot.merge(df_plot2,how='right',right_on='ano',left_on='Ano do Óbito')
    teste = teste.replace(np.NaN,0)
    ano = pd.DataFrame(np.arange(2000,2020))
    ano.columns = ['ano']
    denominador = pop.iloc[3:,:].iloc[:,1:4].apply(lambda x: x*get_pop_by_faixa()['5a9']).dropna()
    taxa = pd.concat([(teste.iloc[:,1:4]/denominador).iloc[:-1,:],ano],axis=1)*100_000
    #taxa = pd.concat([(teste.iloc[:,1:4]/pop.iloc[:,1:4]).iloc[:-1,:],ano],axis=1)
    plot_lines(taxa,xcol = 'ano',title= title + ' - 10 a 14 Anos')
```

## Serie historica do crescimento populacional do Espirito Santo


```python
plot_lines(pop,xcol='index')
```


    
![png](images/output_18_0.png)
    



```python
plot_lines_to_0a5()
```


    
![png](images/output_19_0.png)
    



```python
plot_lines_to_5a9()
```


    
![png](images/output_20_0.png)
    



```python
plot_lines_to_10a15()
```


    
![png](images/output_21_0.png)
    



```python
newdf = pd.DataFrame(df.groupby(['faixa','Ano do Óbito']).sum()['Total']).reset_index()
_ = sns.lineplot(x = newdf['Ano do Óbito'],y=newdf['Total'],hue = newdf['faixa'], marker='o')
_ = plt.title('Evolução de Óbitos por faixa étaria - ' + title, loc='left', fontsize=25, fontweight=0, color='Black')
```


    
![png](images/output_22_0.png)
    


# Plot Espirito Santo


```python
import geopandas as gpd
```


```python
x = gpd.read_file('shapefiles/Lim_Macrorregiao.shp')
x = x[x.nome!='Norte']
```


```python
TOTAL = df.groupby(by='Ano do Óbito').sum()
percen_pop = get_pop_by_faixa()
percen_pop['sumpercen'] = percen_pop['0a4'] + percen_pop['5a9'] + percen_pop['10a14']
```


```python
def plot_evolucao_mapa():
    fig, axes = plt.subplots(2,3)
    fig.suptitle('Evolução da taxa de Neoplasia',fontsize=20)

    vmax = 7/(732490*0.300418)*100_000
    # [norte,sul,metropolitana]
    x['value'] = list(map(lambda x: x*(1/0.300418)*100_000,[7/732490,1/545491, 7/1660081]))
    _ = x.plot(x['value'],cmap="crest",legend=True,ax=axes[0][0],vmax=vmax)
    _ = axes[0][0].title.set_text('1999')
    _ = axes[0,0].axis('off')

    x['value'] = [4*100_000/(773930*0.286621),2*100_000/(609420*0.286621),6*100_000/(1866869*0.286621)]
    _ = x.plot(x['value'],cmap="crest",legend=True,ax=axes[0][1],vmax=vmax)
    _ = axes[0][1].title.set_text('2003')
    _ = axes[0,1].axis('off')

    x['value'] = list(map(lambda x: x*(1/0.268982)*100_000,[4/786492,2/607461,6/1957716]))
    _ = x.plot(x['value'],cmap="crest",legend=True,ax=axes[0][2],vmax=vmax)
    _ = axes[0][2].title.set_text('2007')
    _ = axes[0,2].axis('off')

    x['value'] = list(map(lambda x: x*(1/0.250804)*100_000,[4/854539,2/626503,6/2066013]))
    _ = x.plot(x['value'],cmap="crest",legend=True,ax=axes[1][0],vmax=vmax)
    _ = axes[1][0].title.set_text('2011')
    _ = axes[1,0].axis('off')

    x['value'] = list(map(lambda x: x*(1/0.231885)*100_000,[4/943971,2/680871,6/2305069]))
    _ = x.plot(x['value'],cmap="crest",legend=True,ax=axes[1][1],vmax=vmax)
    _ = axes[1][1].title.set_text('2015')
    _ = axes[1,1].axis('off')

    x['value'] = list(map(lambda x: x*(1/0.213184)*100_000,[4/961446,2/678071,6/2379133]))
    _ = x.plot(x['value'],cmap="crest",legend=True,ax=axes[1][2],vmax=vmax)
    _ = axes[1][2].title.set_text('2019')
    _ = axes[1,2].axis('off')
```


```python
plot_evolucao_mapa()
```


    
![png](images/output_28_0.png)
    


## Modelo


```python
import statsmodels.tsa.stattools as smt
from statsmodels.tsa.api import VAR
```


```python
totalizador = df.groupby(['Ano do Óbito']).sum()
serie_met = totalizador.iloc[:,1]
serie_sul = totalizador.iloc[:,0]
serie_nor = totalizador.iloc[:,2]
```


```python
def plot_ccf_sm(target, exog, unbiased=False, nlags=30):
    """Plot CCF using Statsmodels"""
    ccfs = smt.ccf(target, exog, adjusted=False)[:nlags+1]
    lags = np.arange(len(ccfs))[:nlags+1]
    _ = plt.stem(lags, ccfs, use_line_collection=True)
    _ = plt.title(f"Cross Correlation (Statsmodels): {target.name} & {exog.name}")
    _ = plt.show()
```


```python
plot_ccf_sm(serie_sul,serie_met)
plot_ccf_sm(serie_met,serie_sul)
plot_ccf_sm(serie_sul,serie_nor)
plot_ccf_sm(serie_nor,serie_sul)
plot_ccf_sm(serie_nor,serie_met)
plot_ccf_sm(serie_met,serie_nor)
```


    
![png](images/output_33_0.png)
    



    
![png](images/output_33_1.png)
    



    
![png](images/output_33_2.png)
    



    
![png](images/output_33_3.png)
    



    
![png](images/output_33_4.png)
    



    
![png](images/output_33_5.png)
    



```python
model = VAR(totalizador.iloc[:,0:3])
fitted = model.fit()
fitted.summary()
```

    /home/joaoholz/miniconda3/envs/lestat/lib/python3.10/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: An unsupported index was provided and will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)





      Summary of Regression Results   
    ==================================
    Model:                         VAR
    Method:                        OLS
    Date:           Mon, 19, Dec, 2022
    Time:                     13:02:18
    --------------------------------------------------------------------
    No. of Equations:         3.00000    BIC:                    4.19363
    Nobs:                     24.0000    HQIC:                   3.76088
    Log likelihood:          -133.419    FPE:                    37.1151
    AIC:                      3.60461    Det(Omega_mle):         23.3728
    --------------------------------------------------------------------
    Results for equation SUL
    ===================================================================================
                          coefficient       std. error           t-stat            prob
    -----------------------------------------------------------------------------------
    const                    0.601047         1.013667            0.593           0.553
    L1.SUL                  -0.165286         0.216538           -0.763           0.445
    L1.METROPOLITANA         0.220977         0.122649            1.802           0.072
    L1.CENTRAL NORTE         0.013458         0.174905            0.077           0.939
    ===================================================================================
    
    Results for equation METROPOLITANA
    ===================================================================================
                          coefficient       std. error           t-stat            prob
    -----------------------------------------------------------------------------------
    const                    8.162002         1.620434            5.037           0.000
    L1.SUL                  -0.896446         0.346155           -2.590           0.010
    L1.METROPOLITANA         0.272347         0.196065            1.389           0.165
    L1.CENTRAL NORTE        -0.349771         0.279600           -1.251           0.211
    ===================================================================================
    
    Results for equation CENTRAL NORTE
    ===================================================================================
                          coefficient       std. error           t-stat            prob
    -----------------------------------------------------------------------------------
    const                    3.623198         1.206409            3.003           0.003
    L1.SUL                  -0.387451         0.257711           -1.503           0.133
    L1.METROPOLITANA         0.165595         0.145970            1.134           0.257
    L1.CENTRAL NORTE        -0.322966         0.208161           -1.552           0.121
    ===================================================================================
    
    Correlation matrix of residuals
                          SUL  METROPOLITANA  CENTRAL NORTE
    SUL              1.000000       0.098027       0.234534
    METROPOLITANA    0.098027       1.000000      -0.074210
    CENTRAL NORTE    0.234534      -0.074210       1.000000
    





```python

```
