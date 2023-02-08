import pandas as pd

def get_pop_by_faixa():
    pop = pd.read_csv('data/popu.csv',';')
    colnames = pop.iloc[:,0]
    pop = pop.transpose().iloc[5:]
    pop.columns = colnames
    pop = pop.apply(pd.to_numeric)
    pop = pop.reset_index()
    pop = pop.iloc[3:]
    df = pd.read_csv('data/populacao_idade.csv',';')
    df = df.iloc[:-2,:]
    pop_ano = df.iloc[:,1:4].apply(lambda x: x/df['Total']).dropna()
    pop_ano['Ano'] = df['Ano']
    pop_ano.columns = ['0a4','5a9','10a14','Ano']
    return pop_ano