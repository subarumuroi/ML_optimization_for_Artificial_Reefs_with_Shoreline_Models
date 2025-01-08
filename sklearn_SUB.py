'''
custom sub functions to go with sklearn_testing notebook
'''

#%%
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#%%
def LoadData(csvFile):
    
    df = pd.read_csv(csvFile)
    
    df_Inputs = df.iloc[2:8,1:].astype(float).T
    df_Inputs = df_Inputs.reset_index(drop=True)
    df_Inputs.columns = ['H','T','Dir','Dist','L','W']

    df_Targets = df.iloc[44,1:].astype(float).T # selects central 
    df_Targets = df_Targets.reset_index(drop=True)
    df_Targets.columns = ['SL']
    
    return df_Inputs, df_Targets 

#%%
def getMetrics(actual,predicted):

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    return mae, mse, r2

#%%
def addMetricRank(df):
    
    # Rank models based on each metric
    df['Rank_MAE'] = df['MAE'].rank(ascending=True)
    df['Rank_MSE'] = df['MSE'].rank(ascending=True)
    df['Rank_R2'] = df['R2'].rank(ascending=False)

    # Calculate average rank
    df['Rank'] = df[['Rank_MAE', 'Rank_MSE', 'Rank_R2']].mean(axis=1)
    df = df.drop(['Rank_MAE', 'Rank_MSE', 'Rank_R2'], axis=1)

    # Display the result
    df = df.sort_values(by='Rank')
    
    return df

#%%
def plot_actual_vs_predicted(ax, y_actual, y_pred, model_name, set_type, mae, mse, r2, y0):
    ax.scatter(y_actual, y_pred, s=20, alpha=0.6, label=set_type)
    ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--', lw=2)  # Add 1:1 line
    ax.set_title(f'{model_name}')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to make axes square
    ax.set_xlim([min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())])
    ax.set_ylim([min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())])
    
    ax.text(0.05, y0, f'{set_type} \nMAE: {mae:.4f}\nMSE: {mse:.4f}\nR2: {r2:.4f}', transform=ax.transAxes)
    ax.grid(True)
    ax.legend()
    
#%%
