import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from jh_utils.data.pandas.preprocessing import make_dummies
import seaborn as sns

keep_cols = [
'A612 - temperatura maxima na hora ant. (aut) (°c)',
'A613 - temperatura maxima na hora ant. (aut) (°c)',
'A614 - temperatura maxima na hora ant. (aut) (°c)',
'A634 - temperatura maxima na hora ant. (aut) (°c)',
'hour_9', 'hour_9**2', 'hour_9**3',
'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
'month_8', 'month_9', 'month_10', 'month_11', 'month_12']

small_column_names_for_mldf = ['A612','A613',
       'A614','A634', 'hour_9',
       'hour_9**2', 'hour_9**3', 'month_2', 'month_3', 'month_4', 'month_5',
       'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11',
       'month_12']

def get_data():
    df = pd.read_csv('ES_1.csv')
    df.date_time = pd.to_datetime(df.date_time)
    ## adding covariables
    df['hour'] = df.date_time.dt.hour
    df['month'] = df.date_time.dt.month
    df['year'] = df.date_time.dt.year
    df['day_of_year'] = df.date_time.dt.day_of_year
    df['weekofyear'] = df.date_time.dt.weekofyear

    ## transforming start hour in 9, to use hour**3, some models are hierarchical so is necessary to keep hour**2 
    df['hour_9'] = df['hour'].apply(lambda x: (x-9)%24)
    df['hour_9**2'] = df['hour_9']**2
    df['hour_9**3'] = df['hour_9']**3
    return df

def get_data_for_ml():
    df = get_data()
    return pd.concat([df,make_dummies(df.month)],axis = 1)