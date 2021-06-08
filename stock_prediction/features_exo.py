import pandas as pd
import numpy as np
import os

def exo_selection(df, sp500=True, gold=True, eurusd=True, nasdaq=True, crude=True, vix=True) :
    '''This function will select the indexes we want to be part of the df
    for the modelisation.
    We need to True/False the indexes in the parameters'''

    # we want the path of where we rare
    we_are = os.getcwd()
    path = we_are[:-16] + 'raw_data/'
    # we need a list to store the column name  to be able to access easily when rebasing
    exo_col_name = []
    # we need to store the df in a list to know at the end which df we need to merge
    exo_df_list = []

    # we load the files we select and create the feature Return
    if sp500 :
        # we build the path of the csv file
        path_of = path + 'S&P500.csv'
        # we load the csv to pd.dtaframe
        data_sp500 = pd.read_csv(path_of)
        # now we want the return of the Adj Close price
        data_sp500['Return_S&P500'] = data_sp500['Close/Last'].pct_change(1)
        # then as we only need the return (in case keep other features) we just keep this column and the Date
        data_sp500 = data_sp500[['Date', 'Return_S&P500']]
        # we store the name of the column
        exo_col_name.append('Return_S&P500')
        # we store the df in the list to merge it later
        exo_df_list.append(data_sp500)
    elif gold :
        # we build the path of the csv file
        path_of = path + 'GC=F.csv'
        # we load the csv to pd.dtaframe
        data_gold = pd.read_csv(path_of)
        # now we want the return of the Adj Close price
        data_gold['Return_Gold'] = data_gold['Close'].pct_change(1)
        # then as we only need the return (in case keep other features) we just keep this column and the Date
        data_gold = data_gold[['Date', 'Return_Gold']]
        exo_col_name.append('Return_Gold')
        # we store the df in the list to merge it later
        exo_df_list.append(data_gold)
    elif eurusd :
        # we build the path of the csv file
        path_of = path + 'EURUSD=X.csv'
        # we load the csv to pd.dtaframe
        data_usd = pd.read_csv(path_of)
        # now we want the return of the Adj Close price
        data_usd['Return_Usd'] = data_usd['Close'].pct_change(1)
        # then as we only need the return (in case keep other features) we just keep this column and the Date
        data_usd = data_usd[['Date', 'Return_Usd']]
        exo_col_name.append('Return_Usd')
        exo_df_list.append(data_usd)
    elif nasdaq:
        # we build the path of the csv file
        path_of = path + '^IXIC.csv'
        # we load the csv to pd.dtaframe
        data_nasdaq = pd.read_csv(path_of)
        # now we want the return of the Adj Close price
        data_nasdaq['Return_Nasdaq'] = data_nasdaq['Adj Close'].pct_change(1)
        # then as we only need the return (in case keep other features) we just keep this column and the Date
        data_nasdaq = data_nasdaq[['Date', 'Return_Nasdaq']]
        exo_col_name.append('Return_Nasdaq')
        exo_df_list.append(data_nasdaq)
    elif crude:
        # we build the path of the csv file
        path_of = path + 'CL=F.csv'
        # we load the csv to pd.dtaframe
        data_crude = pd.read_csv(path_of)
        # now we want the return of the Adj Close price
        data_crude['Return_Crude'] = data_crude['Close'].pct_change(1)
        # then as we only need the return (in case keep other features) we just keep this column and the Date
        data_crude = data_crude[['Date', 'Return_Crude']]
        exo_col_name.append('Return_Crude')
        exo_df_list.append(data_crude)
    elif vix:
        # we build the path of the csv file
        path_of = path + '^VIX.csv'
        # we load the csv to pd.dtaframe
        data_vix = pd.read_csv(path_of)
        # for the VIX, index of volatility of US markets, we don't want thye return
        # we mill have to keep that index un-rebased
        data_vix['Close'] = data_vix['Vix_No_Rebase']
        data_vix = data_vix[['Date', 'Vix_No_Rebase']]

    # now that we have all the df we need to merge them on Date to keep only one column
    # for date and one column for each index return
    # we merge it on the DataFrame returned by Data_prep in data_prep.py

    # first we need to merge the first df of the list on the dataframe data_prep
    data_exo = df.merge(exo_df_list[0], how='inner', on='Date')
    for dataframe in exo_df_list[1:] :
        data_exo = data_exo.merge(dataframe, how='inner', on='Date')

    # we now have the dataframe ready for the modelling
    # this function returns the dataframe & the list of indexes we decided to include
    # like that, when doing the rebasing, we will be able to rebase the selected indexes
    # without naming them
    return data_exo, exo_col_name

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
