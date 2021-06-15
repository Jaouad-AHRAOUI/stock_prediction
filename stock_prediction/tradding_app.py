import pandas as pd
import numpy as np

def best_stocks(df, sell=True, eq_weight=False) :
    '''This function allows us to select the best 10 stocks in our
    predictions of returns for each day of the tradding experience.
    We can use ut for our predictions and ask for a inequal weight ponderation
    of the stocks in the portfolio, or call it for the 'True' comparaison
    and ask for an equal ponderation. We can also have the best 10 stocks
    to sell or buy or just to buy. '''

    # the df has a 'stocks' features and one column per day of prediction
    # the first column is 'stocks'
    # as we need to for loop on each day and we don't want to know the number
    # of days, we are going to drop 'stocks' in the column list and loop
    day_pred_list = df.columns[1:]
    # we create a list to store portfolio for each day
    ptf_day_list = []

    for days in day_pred_list :

        # we select only the day we want to analyze
        day_search = df[['stocks', days]].copy()
        # if we work with BUY & SELL we need the absolute returns
        if sell :
            # we create a column with the absolute returns
            day_search['abs_returns'] = day_search[days].abs()
            ten_best = day_search.nlargest(10, 'abs_returns')
            # we need to create the weight column depending on eq_weight
            if eq_weight :
                ten_best['weights'] = 0.10
            else :
                ten_best['weights'] = ten_best['abs_returns'] / ten_best['abs_returns'].sum()
            # finally we can drop the abs_returns column
            ten_best.drop(columns='abs_returns', inplace=True)
        else :
            ten_best = day_search.nlargest(10, days)
            if eq_weight:
                ten_best['weights'] = 0.10
            else:
                ten_best['weights'] = ten_best[days] / ten_best[days].sum()

        # we have a df with the list of 10 best stocks
        # their predicted returns for the day
        # the weight in the portfolio
        # we store it in the list
        ptf_day_list.append(ten_best)

    return ptf_day_list


