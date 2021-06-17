import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import datetime
import matplotlib.pyplot as plt
from numpy.random import default_rng

from stock_prediction.data_prep_api import Data_Prep_Api
from stock_prediction.tradding_app import best_stocks, true_returns, portfolio
from stock_prediction.workflow import data_collection
from stock_prediction.tradding_app import true_returns, portfolio, best_stocks



#---Set a title
# st.title('Stock return prediction')
st.markdown("<h1 style='text-align: center; color: black;'>Stock return prediction</h1>", unsafe_allow_html=True)


# #---Button
# if st.button('Run'):
#     #---Put all you want to execute when button is clicked

    
#     print('button clicked!')
#     st.write('I was clicked 🎉')
#     st.write('Further clicks are not visible but are executed')
# else:
#     st.write('I was not clicked 😞')


#---Select start date and end date for dataframe generation to show prediction for best stocks

start_date = st.sidebar.date_input(
    "Please, enter start date:",
    datetime.date(2021, 1, 12))

end_date = st.sidebar.date_input(
    "Please, enter end date:",
    datetime.date(2021, 1, 15))

# Select price
'''
## Are you ready to invest ?
'''
invest = st.sidebar.slider('Please, enter the amount from 10.000€ to 100.000€:',10000,100000,50000, 10000)



#---Visualize all stocks table
st.markdown("""
**Predictive returns for Euro Stoxx 50**
            """)
@st.cache
def visualize_stocks():

    #---Retrieve df to visualize
    dict_hard_data, dict_prep_data, df_es50 = data_collection('2021-01-01', 20)
        
    company_list = [
        'asml', 'lvmh', 'sap', 'linde', 'siemens', 'total', 'sanofi', 'allianz',
        'loreal', 'schneider', 'iberdrola', 'enel', 'air-liquide', 'basf', 'bayer',
        'adidas', 'airbus', 'deutsche-telecom', 'daimler', 'bnp',
        'anheuser-busch', 'vinci', 'banco-santander', 'philips', 'kering',
        'deutsche-post', 'axa', 'safran', 'danone', 'essilor', 'intensa',
        'munchener', 'pernod', 'vonovia', 'vw', 'ing', 'crh', 'industria-diseno',
        'kone', 'deutsche-borse', 'ahold', 'flutter', 'amadeus', 'engie', 'bmw',
        'vivendi', 'eni', 'nokia']

    #---Temp random df generation
    rng = default_rng()
    day_one = rng.standard_normal(48)
    day_two = rng.standard_normal(48)
    day_three = rng.standard_normal(48)
    day_four = rng.standard_normal(48)
    day_five = rng.standard_normal(48)
    
    # we create the dataframe of predicted returns for all stocks in the index
    predictive_returns = pd.DataFrame({
        'stocks' : company_list,
        '2021-01-12' : day_one,
        '2021-01-13' : day_two,
        '2021-01-14' : day_three,
        '2021-01-15' : day_four
                                    })  
    return predictive_returns, dict_hard_data, dict_prep_data, df_es50

predictive_returns, dict_hard_data, dict_prep_data, df_es50 = visualize_stocks()
st.write(predictive_returns)


#---Selectbob to select stock to plot
@st.cache
def get_select_box_data():
    print('get_select_box_data called')
    return predictive_returns

df = get_select_box_data()
st.sidebar.markdown("""
**Select stock to visualize**
            """)
option =  st.sidebar.selectbox('', df['stocks'])


#---Pred returns df
pred_df = df[df['stocks'] == option]
pred_df = pred_df.T
pred_df.drop(index= pred_df.index[0], axis= 0, inplace= True)
pred_df.columns= [option]


#---True returns df
df_close_prices, df_true_returns = true_returns('2021-01-12', '2021-01-15', dict_hard_data)

true_df = df_true_returns[option]
true_df.drop(index= true_df.index[0], axis= 0, inplace= True)


#---Add a pred true values plot
@st.cache
def get_line_chart_data():
    print('get_line_chart_data called')
    df = pd.concat([pred_df,true_df], axis= 1)
    df.columns= ['prediction', 'real']
    return df

df = get_line_chart_data()

#---Plot chosen stock

# st.write('Prediction vs Real return for selected stock: ', option)
# st.line_chart(df)

st.markdown("<h2 style='text-align: center; color: black;'>Prediction vs Real return for selected stock</h2>", unsafe_allow_html=True)
st.line_chart(df)










