import sklearn.metrics as me
import pandas as pd

def evaluate(y_true,y_pred):
    metrics_names = ['max_error','r2_score','mean_squared_error','median_absolute_error','mean_absolute_error','explained_variance_score']
    metrics = [me.max_error(y_true,y_pred),me.r2_score(y_true,y_pred),me.mean_squared_error(y_true,y_pred),me.median_absolute_error(y_true,y_pred),me.mean_absolute_error(y_true,y_pred),me.explained_variance_score(y_true,y_pred)]
    df = pd.DataFrame(metrics_names)
    df.columns = ['metrics_name']
    df['metrics'] = metrics
    return df