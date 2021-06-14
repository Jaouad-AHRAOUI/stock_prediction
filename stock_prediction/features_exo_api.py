import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import os

def exo_selection_api(df, names, max = False) :
    '''This function will select the indexes we want to be part of the df
    for the modelisation.
    names is a list of all exogene feature'''
    # we want the path of where we rare
    company_dict = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/company_dict.csv"))
    company_dict.set_index("name",inplace = True)                   
    names_without_vix = names.copy()
    if "vix" in names :
        names_without_vix.remove("vix")
        
    # we need a list to store the column name  to be able to access easily when rebasing
    exo_col_name = []
    # we need to store the df in a list to know at the end which df we need to merge
    exo_df_list = []
    
    # we load the files we select and create the feature Return
    if max==False :
        for name in names_without_vix :
            # we load the data from yfinance
            data = yf.download(company_dict.loc[name][0], start=str(date.today() - timedelta(weeks=52*5)), end=str(date.today()))
            data.reset_index(inplace=True)
            # now we want the return of the Adj Close price
            data[f'Return_{name}'] = data['Close'].pct_change(1)
            # then as we only need the return (in case keep other features) we just keep this column and the Date
            data = data[['Date', f'Return_{name}']]
            # we store the name of the column
            exo_col_name.append(f'Return_{name}')
            # we store the df in the list to merge it later
            exo_df_list.append(data)
    else: 
        for name in names_without_vix :
            # we load the data from yfinance
            data = yf.download(company_dict.loc[name][0], periode = "max")
            data.reset_index(inplace=True)
            # now we want the return of the Adj Close price
            data[f'Return_{name}'] = data['Close'].pct_change(1)
            # then as we only need the return (in case keep other features) we just keep this column and the Date
            data = data[['Date', f'Return_{name}']]
            # we store the name of the column
            exo_col_name.append(f'Return_{name}')
            # we store the df in the list to merge it later
            exo_df_list.append(data)
    
    if "vix" in names:
        if max == False:
            #we load the data from yfinance
            data = yf.download(company_dict.loc["vix"][0], start=str(date.today() - timedelta(weeks=52*5)), end=str(date.today()))
            data.reset_index(inplace=True)
            # for the VIX, index of volatility of US markets, we don't want thye return
            # we mill have to keep that index un-rebased
            data['Vix_No_Rebase'] = data['Close'] / 100
            data = data[['Date', 'Vix_No_Rebase']]
            exo_df_list.append(data)
        else:
            #we load the data from yfinance
            data = yf.download(company_dict.loc["vix"][0], period="max")
            data.reset_index(inplace=True)
            # for the VIX, index of volatility of US markets, we don't want thye return
            # we mill have to keep that index un-rebased
            data['Vix_No_Rebase'] = data['Close'] / 100
            data = data[['Date', 'Vix_No_Rebase']]
            exo_df_list.append(data)
        

    # now that we have all the df we need to merge them on Date to keep only one column
    # for date and one column for each index return
    # we merge it on the DataFrame returned by Data_prep in data_prep.py

    # first we need to merge the first df of the list on the dataframe data_prep
    for dataframe in exo_df_list :
        df = df.merge(dataframe, how='inner', on='Date')

    # here we need to fill NaN to 0 for all Retuns columns
    # but for vix, we need to value of the row -1
    # vix_return column is not in the "exo_col_name" because no need to rebase later

    dict_to_fill = {column: 0.0 for column in exo_col_name}
    df.fillna(value=dict_to_fill, inplace=True)
    df.fillna(method='ffill', inplace=True)

    # we now have the dataframe ready for the modelling
    # this function returns the dataframe
    return df

