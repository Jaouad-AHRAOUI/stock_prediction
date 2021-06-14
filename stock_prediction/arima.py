from numpy.core.fromnumeric import std
from stock_prediction.data_prep import Data_Prep
from stock_prediction.features_exo import exo_selection

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error as MAPE

company_dict = {
    'asml': 'ASML.AS',
    'lvmh': 'MC.PA',
    'sap': 'SAP.DE',
    'linde': 'LIN',
    'siemens': 'SIE.DE',
    'total': 'FP.PA',
    'sanofi': 'SAN.PA',
    'allianz': 'ALV.DE',
    'loreal': 'OR.PA',
    'schneider': 'SU.PA',
    'iberdrola': 'IBE.MC',
    'enel': 'ENEL.MI',
    'air-liquide': 'AI.PA',
    'basf': 'BAS.DE',
    'bayer': 'BAYN.DE',
    'adidas': 'ADS.DE',
    'airbus': 'AIR.PA',
    'adyen': 'ADYEN.AS',
    'deutsche-telecom': 'DTE.DE',
    'daimler': 'DAI.DE',
    'bnp': 'BNP.PA',
    'anheuser-busch': 'ABI.BR',
    'vinci': 'DG.PA',
    'prosus': 'PRX.AS',
    'banco-santander': 'SAN.MC',
    'philips': 'PHIA.AS',
    'kering': 'KER.PA',
    'deutsche-post': 'DPW.DE',
    'axa': 'CS.PA',
    'safran': 'SAF.PA',
    'danone': 'BN.PA',
    'essilor': 'EL.PA',
    'intensa': 'ISP.MI',
    'munchener': 'MUV2.DE',
    'pernod': 'RI.PA',
    'vonovia': 'VNA.DE',
    'vw': 'VOW3.DE',
    'ing': 'INGA.AS',
    'crh': 'CRG.IR',
    'industria-diseno': 'ITX.MC',
    'kone': 'KNEBV.HE',
    'deutsche-borse': 'DB1.DE',
    'ahold': 'AHOG.DE',
    'flutter': 'FLTR.IR',
    'amadeus': 'AMS.MC',
    'engie': 'ENGI.PA',
    'bmw': 'BMW.DE',
    'vivendi': 'VIV.PA',
    'eni': 'ENI.MI',
    'nokia': 'NOKIA.HE'
}

# in that dictionary,
# we store the best max_train we found during the analysis
dict_max_train = {
  'asml': 43,
  'lvmh': 43,
  'sap': 46,
  'linde': 45,
  'siemens': 46,
  'total': 46,
  'sanofi': 44,
  'allianz': 46,
  'loreal': 46,
  'schneider': 45,
  'iberdrola': 43,
  'enel': 44,
  'air-liquide': 40,
  'basf': 43,
  'bayer': 45,
  'adidas': 42,
  'airbus': 44,
  'adyen': 40,
  'deutsche-telecom': 45,
  'daimler': 44,
  'bnp': 42,
  'anheuser-busch': 44,
  'vinci': 41,
  'prosus': 44,
  'banco-santander': 43,
  'philips': 42,
  'kering': 42,
  'deutsche-post': 46,
  'axa': 46,
  'safran': 44,
  'danone': 46,
  'essilor': 41,
  'intensa': 46,
  'munchener': 45,
  'pernod': 40,
  'vonovia': 45,
  'vw': 46,
  'ing': 44,
  'crh': 44,
  'industria-diseno': 44,
  'kone': 43,
  'deutsche-borse': 46,
  'ahold': 46,
  'flutter': 43,
  'amadeus': 45,
  'engie': 45,
  'bmw': 41,
  'vivendi': 41,
  'eni': 45,
  'nokia': 46
}

# in that dictionary, we store the exogenous features
# used to train the model
exo_dict = {
    'asml': ['High-Close_ASML.AS', 'Low-Close_ASML.AS'],
    'lvmh': ['High-Close_MC.PA', 'Low-Close_MC.PA'],
    'sap': ['High-Close_SAP.DE', 'Low-Close_SAP.DE'],
    'linde': ['High-Close_LIN', 'Low-Close_LIN'],
    'siemens': ['High-Close_SIE.DE', 'Low-Close_SIE.DE'],
    'total': ['High-Close_FP.PA', 'Low-Close_FP.PA'],
    'sanofi': ['High-Close_SAN.PA', 'Low-Close_SAN.PA'],
    'allianz': ['High-Close_ALV.DE', 'Low-Close_ALV.DE'],
    'loreal': ['High-Close_OR.PA', 'Low-Close_OR.PA'],
    'schneider': ['High-Close_SU.PA', 'Low-Close_SU.PA'],
    'iberdrola': ['High-Close_IBE.MC', 'Low-Close_IBE.MC'],
    'enel': ['High-Close_ENEL.MI', 'Low-Close_ENEL.MI'],
    'air-liquide': ['High-Close_AI.PA', 'Low-Close_AI.PA'],
    'basf': ['High-Close_BAS.DE', 'Low-Close_BAS.DE'],
    'bayer': ['High-Close_BAYN.DE', 'Low-Close_BAYN.DE'],
    'adidas': ['High-Close_ADS.DE', 'Low-Close_ADS.DE'],
    'airbus': ['High-Close_AIR.PA', 'Low-Close_AIR.PA'],
    'adyen': ['High-Close_ADYEN.AS', 'Low-Close_ADYEN.AS'],
    'deutsche-telecom': ['High-Close_DTE.DE', 'Low-Close_DTE.DE'],
    'daimler': ['High-Close_DAI.DE', 'Low-Close_DAI.DE'],
    'bnp': ['High-Close_BNP.PA', 'Low-Close_BNP.PA'],
    'anheuser-busch': ['High-Close_ABI.BR', 'Low-Close_ABI.BR'],
    'vinci': ['High-Close_DG.PA', 'Low-Close_DG.PA'],
    'prosus': ['High-Close_PRX.AS', 'Low-Close_PRX.AS'],
    'banco-santander': ['High-Close_SAN.MC', 'Low-Close_SAN.MC'],
    'philips': ['High-Close_PHIA.AS', 'Low-Close_PHIA.AS'],
    'kering': ['High-Close_KER.PA', 'Low-Close_KER.PA'],
    'deutsche-post': ['High-Close_DPW.DE', 'Low-Close_DPW.DE'],
    'axa': ['High-Close_CS.PA', 'Low-Close_CS.PA'],
    'safran': ['High-Close_SAF.PA', 'Low-Close_SAF.PA'],
    'danone': ['High-Close_BN.PA', 'Low-Close_BN.PA'],
    'essilor': ['High-Close_EL.PA', 'Low-Close_EL.PA'],
    'intensa': ['High-Close_ISP.MI', 'Low-Close_ISP.MI'],
    'munchener': ['High-Close_MUV2.DE', 'Low-Close_MUV2.DE'],
    'pernod': ['High-Close_RI.PA', 'Low-Close_RI.PA'],
    'vonovia': ['High-Close_VNA.DE', 'Low-Close_VNA.DE'],
    'vw': ['High-Close_VOW3.DE', 'Low-Close_VOW3.DE'],
    'ing': ['High-Close_INGA.AS', 'Low-Close_INGA.AS'],
    'crh': ['High-Close_CRG.IR', 'Low-Close_CRG.IR'],
    'industria-diseno': ['High-Close_ITX.MC', 'Low-Close_ITX.MC'],
    'kone': ['High-Close_KNEBV.HE', 'Low-Close_KNEBV.HE'],
    'deutsche-borse': ['High-Close_DB1.DE', 'Low-Close_DB1.DE'],
    'ahold': ['High-Close_AHOG.DE', 'Low-Close_AHOG.DE'],
    'flutter': ['High-Close_FLTR.IR', 'Low-Close_FLTR.IR'],
    'amadeus': ['High-Close_AMS.MC', 'Low-Close_AMS.MC'],
    'engie': ['High-Close_ENGI.PA', 'Low-Close_ENGI.PA'],
    'bmw': ['High-Close_BMW.DE', 'Low-Close_BMW.DE'],
    'vivendi': ['High-Close_VIV.PA', 'Low-Close_VIV.PA'],
    'eni': ['High-Close_ENI.MI', 'Low-Close_ENI.MI'],
    'nokia': ['High-Close_NOKIA.HE', 'Low-Close_NOKIA.HE']
}

def arima_multi_day(name, days, alpha) :
    '''This function compute the ARIMA model for a specific stock
    on a period of time.
    It returns a df with predictions, true values, confidence interval
    confidence interval = 1 - alpha
    and the value of the day before to be able to compute the returns
    in the df, features taht we will need later to improve our application
    return prediction, return true, return low confidence, return high confidence
    and dates to be able to merge with other features
    '''

    # we need the number of days we need to train the model
    # days in parameters are the number of days we want with predictions
    # we retrieve the best max_train for the selected company
    best_max_train = dict_max_train[name]
    global_length = best_max_train + days

    # we instantiate the Data_Prep class to create the df
    stock = Data_Prep(name, best_max_train)
    # with data_prep function we add the features
    #*********************************************
    # WE MUST MODIFY THE data_prep FUNCTION we_are
    #*********************************************
    data_global = stock.data_prep()
    # with function select_features we select the best exo features
    best_exo_features = exo_dict[name]
    # after several tests, we found that the best exo features for all stocks
    # were High_Close, and Low_Close # if more tests bring us to specific features
    # we will have to make a if for the stocks concerned
    data_exo = stock.select_features(data_global,
                                     Return=True,
                                     Log_Return=False,
                                     High_Low=False,
                                     High_Close=True,
                                     Low_Close=True,
                                     Volume_Change=False,
                                     Period_Volum=False,
                                     Annual_Vol=False,
                                     Period_Vol=False,
                                     Return_Index=False,
                                     Volum_Index=False,
                                     Relative_Return=False)
    # we select the rows needed
    data_exo = data_exo[- global_length : ]

    # we need the code of the company to be able to re-create the name of the rebased return
    code_name = company_dict[name]

    # we rebase 100 the return
    data_exo = stock.Price_Rebase(data_exo)

    # we create the y_endogenous and y_exogenous
    y_endo = data_exo[f'Return_{code_name}_R']
    y_exo = np.array(data_exo[best_exo_features])

    # now we prepare the for loop that will train the model on max_train
    # and gives us a prediction price (rebase) for each day of the time period

    # we need lists to store the results of the loop
    list_y_pred = []
    baseline = []
    real_value = []
    y_before = []
    y_conf_low = []
    y_conf_high = []
    std_conf = []

    # we fix our parameters for the ARIMA
    order = (0,1,0)
    # number of splits to cover the full time period
    splits = len(y_endo) - best_max_train

    # with function TimeSeriesSplit we create the indexes
    folds = TimeSeriesSplit(n_splits=splits,
                            max_train_size=best_max_train,
                            test_size=1)

    # now the for loop to compute the ARIMA model on each day
    for (train_idx, test_idx) in folds.split(y_endo) :

        # we retrieve the real data in y
        #corresponding of the indexes splited
        y_train = y_endo[train_idx]
        y_exo_train = y_exo[train_idx]
        y_test = y_endo[test_idx]
        y_exo_test = y_exo[test_idx]
        y_true = y_endo[test_idx]
        #base = y_train.iloc[-1]

        # fit our model on y_train of the split n°X
        model = ARIMA(endog=y_train, exog=y_exo_train, order=order).fit()

        # we find our y_pred for this slice on that part of the TS
        y_pred, std_pred, conf = model.forecast(steps=len(y_test), exog=y_exo_test, alpha=alpha)
        #pdb.set_trace()
        # we store the low price of confidence
        y_conf_low.append(conf[0][0])
        # high price of confidence
        y_conf_high.append(conf[0][1])
        # we store y_pred
        list_y_pred.append(y_pred[0])
        # we store the std of the conf interval, see if it helps us
        std_conf.append(std_pred[0])

        # we store the value for basescore
        #baseline.append(base)

        # and the rel value to compare
        real_value.append(y_true.values[0])
        y_before.append(y_train.iloc[-1])

    # once we have all values we can compute MAPE and basescore
    # for purpose of application, we don't need them,
    # used before to compare models
    #base_score = MAPE(baseline, real_value)
    #mape_metric = MAPE(list_y_pred, real_value)

    # we now need to store the results found to be able to work on them
    multi_days_results = np.array([y_before, list_y_pred, y_conf_low, y_conf_high, real_value])
    multi_days_results_df = pd.DataFrame({'yesterday' : y_before,
                                      'prediction' : list_y_pred,
                                      'conf_low' : y_conf_low,
                                      'conf_high' : y_conf_high,
                                      'true' : real_value,
                                      'conf_std' : std_conf,
                                      'Date' : data_exo['Date'][-splits :]})

    # now we have a numpy array that will helps to make calculus on features
    # and the df that will store the resulted columns

    # pred / before -1
    perf_pred = ((multi_days_results[1, :] / multi_days_results[0, :]) - 1)
    # true / before - 1
    perf_true = ((multi_days_results[4, :] / multi_days_results[0, :]) - 1)

    # direction de pred
    # that can gives us tha accuracy of UP/DOWN
    # if needed in the future we can add it to the df from here
    #dir_pred = perf_pred > 0
    # direction de true
    #dir_true = perf_true > 0
    # accurate direction
    #dir_acc = dir_pred == dir_true

    # perf of Low conf
    perf_low = (multi_days_results[2, :] / multi_days_results[0, :] - 1)
    # perf high conf
    perf_high = (multi_days_results[3, :] / multi_days_results[0, :] - 1)

    # we try to analyze the confidence
    # if we cannot make a new model on our results,
    # we will have to analyze the conf interval and try improving results
    #conf_ana = perf_high - (-perf_low)
    #dir_conf_low = perf_low > 0
    #dir_conf_high = perf_high > 0
    #conf_confirm = dir_conf_low == dir_conf_high

    # we store the results in the global df
    multi_days_results_df['perf_pred'] = perf_pred
    multi_days_results_df['perf_true'] = perf_true
    multi_days_results_df['perf_low'] = perf_low
    multi_days_results_df['perf_high'] = perf_high

    return multi_days_results_df
