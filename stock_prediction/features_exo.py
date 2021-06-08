import pandas as pd
import numpy as np
from math import sqrt
import os

def exo_selection_euro_stoxx_50(df, period,columns) :
    '''This function will select the indexes we want to be part of the df'''
    # we build the path of the euro stocks csv file
    path = os.getcwd()[:-16] + 'raw_data/' + '^STOXX50E.csv'
    df_es50 = pd.read_csv(path)
    df['Return_euro_stoxx_50'] = df_es50['Close'].pct_change(1)
    df['Period_Volum_euro_stoxx_50'] = df_es50['Volume'] / df_es50['Volume'].rolling(period).mean() - 1
    for colomn in columns :
        df[f'{colomn}_relatif'] = df[colomn] -df['Return_euro_stoxx_50']
    return df
