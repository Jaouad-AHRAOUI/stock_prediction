import streamlit as st
import pandas as pd
import numpy as np

from numpy.random import default_rng
from stock_prediction.data_prep_api import Data_Prep_Api
from stock_prediction.tradding_app import best_stocks, true_returns, portfolio
from stock_prediction.workflow import data_collection
from stock_prediction.tradding_app import true_returns, portfolio, best_stocks
import yfinance as yf
from datetime import datetime 
import pandas as pd 
import numpy as np

st.title('Stock return prediction ')

#Input date
import datetime

d_start = st.date_input(
    "Please, enter your date:",
    datetime.date(2021, 1, 12))


d_end = st.date_input(
    "Please, enter your date:",
    datetime.date(2021, 1, 15))

# Select price
'''
## Are you ready to invest ?
### Please, enter the amount:
'''
line_count = st.slider('from 100€ to 100.000€:',1000,100000,10000)




    
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

rng = default_rng()
day_one = rng.standard_normal(48)
day_two = rng.standard_normal(48)
day_three = rng.standard_normal(48)
day_four = rng.standard_normal(48)
day_five = rng.standard_normal(48)
# day_six = rng.standard_normal(48)
# day_seven = rng.standard_normal(48)
# day_eight = rng.standard_normal(48)
# day_nine = rng.standard_normal(48)
# day_ten = rng.standard_normal(48)
    
    # we create the dataframe of predicted returns for all stocks in the index
predictive_returns = pd.DataFrame({
    'stocks' : company_list,
    '2021-01-12' : day_one,
    '2021-01-13' : day_two,
    '2021-01-14' : day_three,
    '2021-01-15' : day_four,
    #'2021-01-16' : day_five,
#     '2021-01-07' : day_six,
#     '2021-01-08' : day_seven,
#     '2021-01-09' : day_eight,
#     '2021-01-10' : day_nine,
#     '2021-01-11' : day_ten
})  

st.markdown("""
    - **Predictive returns**
""")

st.write(predictive_returns)

#Select stock to plot

@st.cache
def get_select_box_data():
    print('get_select_box_data called')
    return predictive_returns

df = get_select_box_data()

option = st.selectbox('Select stock to visualize', df['stocks'])

pred_df = df[df['stocks'] == option]

st.write(pred_df)

#Filter df inverse
pred_df.reindex(index= pred_df.columns)

# True returns
df_close_prices, df_true_returns = true_returns('2021-01-12', '2021-01-15', dict_hard_data)

true_df = df_true_returns.columns(option)
st.write(df_true_returns)


# Add a Plot
@st.cache
def get_line_chart_data():
    print('get_line_chart_data called')
    return filtered_df

df = get_line_chart_data()

st.line_chart(df)







# st.markdown("""
#    - **Df Close Prices**
#""")



# st.write(df_close_prices)

# st.markdown("""
#    - **Df true returns**
# """)

# st.write(df_true_returns)










