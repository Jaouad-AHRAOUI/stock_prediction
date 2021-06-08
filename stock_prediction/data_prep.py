import pandas as pd
import numpy as np
from math import sqrt
import os


# list of the companies
# refer to this name to ask the csv file for the analysis
company_list = ['asml',
                'lvmh',
                'sap',
                'linde',
                'siemens',
                'total',
                'sanofi',
                'allianz',
                'loreal',
                'schneider',
                'iberdrola',
                'enel',
                'air-liquide',
                'basf',
                'bayer',
                'adidas',
                'airbus',
                'adyen',
                'deutsche-telecom',
                'daimler',
                'bnp',
                'anheuser-busch',
                'vinci',
                'prosus',
                'banco-santander',
                'philips',
                'kering',
                'deutsche-post',
                'axa',
                'safran',
                'danone',
                'essilor',
                'intensa',
                'munchener',
                'pernod',
                'vonovia',
                'vw',
                'ing',
                'crh',
                'industria-diseno',
                'kone',
                'deutsche-borse',
                'ahold',
                'flutter',
                'amadeus',
                'engie',
                'bmw',
                'vivendi',
                'eni',
                'nokia']

# we create a dictionary with name in key and csv file in value
company_dict = {
    'asml' : 'ASML.AS',
    'lvmh': 'MC.PA',
    'sap' : 'SAP.DE',
    'linde' : 'LIN',
    'siemens' : 'SIE.DE',
    'total' : 'FP.PA',
    'sanofi' : 'SAN.PA',
    'allianz' : 'ALV.DE', 'loreal' : '',
    'schneider' : 'SU.PA',
    'iberdrola' : 'IBE.MC',
    'enel' : 'ENEL.MI',
    'air-liquide' : 'AI.PA',
    'basf' : 'BAS.DE',
    'bayer' : 'BAYN.DE',
    'adidas' : 'ADS.DE',
    'airbus' : 'AIR.PA',
    'adyen' : 'ADYEN.AS',
    'deutsche-telecom' : 'DTE.DE',
    'daimler' : 'DAI.DE',
    'bnp' : 'BNP.PA',
    'anheuser-busch' : 'ABI.BR',
    'vinci' : 'DG.PA',
    'prosus' : 'PRX.AS',
    'banco-santander' : 'SAN.MC',
    'philips' : 'PHIA.AS',
    'kering' : 'KER.PA',
    'deutsche-post' : 'DPW.DE',
    'axa' : 'CS.PA',
    'safran' : 'SAF.PA',
    'danone'  : 'BN.PA',
    'essilor' : 'EL.PA',
    'intensa' : 'ISP.MI',
    'munchener' : 'MUV2.DE',
    'pernod' : 'RI.PA',
    'vonovia' : 'VNA.DE',
    'vw' : 'VOW3.DE',
    'ing' : 'INGA.AS',
    'crh' : 'CRG.IR',
    'industria-diseno' : 'ITX.MC',
    'kone' : 'KNEBV.HE',
    'deutsche-borse' : 'DB1.DE',
    'ahold' : 'AHOG.DE',
    'flutter' : 'FLTR.IR',
    'amadeus' : 'AMS.MC',
    'engie' : 'ENGI.PA',
    'bmw' : 'BMW.DE',
    'vivendi' : 'VIV.PA',
    'eni' : 'ENI.MI',
    'nokia' : 'NOKIA.HE'
}

class Data_Prep :

    def __init__(self, name):
        self.name = name

    def find_csv_path(self) :
        '''this function gives us the right csv name file for a specific company'''
        # we retrieve the name of the file in the dict
        csv_file = company_dict(self.name)
        # we want the path of where we rare
        we_are = os.getcwd()
        # we build the path of the csv file
        path = we_are[:-16] + 'raw_data/' + csv_file + '.csv'

        return path

    def load_csv(self) :
        '''laod the csv file to pandas dataframe'''
        data  = pd.read_csv(self.find_csv_path(self.name))
        return data

    def data_prep(self, period=252) :
        '''Function that make the data preparation for analysis'''
        # first we retrieve the df
        data = self.load_csv(self.name)

        # to be able to know the columns we use when df contains several stocks
        # we put the code of the company in each column name
        col_name = company_dict[self.name]

        # we create the column "RETURN" on "Adj Close"
        # why ? Because no impact on dividends and stock splits
        data[f'Return_{col_name}'] = data['Adj Close'].pct_change(1)
        # we create the feature "LOG RETURN" to test which one is working better
        data[f'Log_Return_{col_name}'] = np.log(data["Close"] / data["Close"].shift())
        # we create the feature "HIGH-LOW"
        data[f'High-Low_{col_name}'] = (data['High'] - data['Low']) / data['Low']
        # we create the feature "HIGH-CLOSE" --> difference between closing price and higher price
        data[f'High-Close_{col_name}'] = (data['High'] - data['Close']) / data['Close']
        # same for the difference between the lowest point and the closing price
        # we compute it as a positive value
        data[f'Low-Close_{col_name}'] = ((data['Close'] - data['Low']) / data['Low'])
        # we create the feature "daily evolution of volume", day by day
        data[f'Volume-Change_{col_name}'] = data['Volume'].pct_change(1)
        # we create the feature "volume difference to the mean"
        # we compute the mean of daily volumes in the time period of the analysis
        # then find the difference for each day
        data[f'Period_Volum_{col_name}'] = data['Volume'] / data['Volume'].rolling(period).mean() - 1
        # finally volatility
        # one annual vl-olatility, computed on 252 days
        data[f'Annual_Vol_{col_name}'] = data['Return'].rolling(252).std() * sqrt(252)
        # another volatility if we work on a pecific time period
        # or if we want to change that parameter
        data[f'Period_Vol_{col_name}'] = data['Return'].rolling(period).std() * sqrt(252)

        # we can remove the columns we used to compute the new features
        # Open, High, Low, Close, Adj Close

        # we will delete the columns directly rather than select the ones we want
        # like that we don't need to name each time with the company name
        del data['Open']
        del data['High']
        del data['Low']
        del data['Close']
        del data['Adj Close']

        # finally we remove the rows with NaN (because volatility calculation)
        # and reset the index
        data = data.dropna().reset_index()

        # we return a df with 4 years of prices
        return data

    def select_features(self, df, Log_Return=True, High_Low=True, High_Close=True, Low_Close=True,
                        Volume_Change=True, Period_Volum=True, Annual_Vol=True,
                        Period_Vol=True) :
        '''Function to be able to remove easily features'''

        col_name = company_dict[self.name]
        # we retrieve our dataframe prepared
        data = df
        if Log_Return == False:
            del data[f'Log_Return_{col_name}']
        elif High_Low==False:
            del data[f'High-Low_{col_name}']
        elif High_Close==False:
            del data[f'High-Close_{col_name}']
        elif Low_Close == False:
            del data[f'Low-Close_{col_name}']
        elif Volume_Change == False:
            del data[f'Volume-Change_{col_name}']
        elif Period_Volum == False:
            del data[f'Period_Volum_{col_name}']
        elif Annual_Vol == False:
            del data[f'Annual_Vol_{col_name}']
        elif Period_Vol == False:
            del data[f'Period_Vol_{col_name}']

        return data


    def Price_Rebase(df, columns=[]) :
        '''This function allows us to rebase 100 at the beginning of our time period
        and follow only the return and be able to compare it with exogenous features
        COLUMNS parameter is a list of features to rebase
        This function needs to be used once the final dataframe is ready to modelling'''

        # we retrieve our df
        data = df
        # to avoid errors on the index we reset_index the df
        data = data.reset_index()

        # we select only the columns we need
        data_rebased = data[columns]
        # we create the new columns rebased
        for col in columns :
            data_rebased.loc[0, f'{col}_R'] = 100
            # we make a for loop to apply the return to the base 100
            for i in range(1, len(data_rebased)-1) :
                data_rebased.loc[i, f'{col}_R'] = data_rebased.loc[i-1, f'{col}_R'] * (data_rebased.loc[i, col] + 1)

        # we now want to delete the columns with the returns
        # and keep only the new rebased time series
        # to do that we delete them from data and data_rebased df
        del data_rebased[columns]
        del data[columns]
        # then we want to merge the rebased df to the data df
        data = pd.concat([data, data_rebased], axis=1)

        return data
