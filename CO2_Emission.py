import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


st.title('CO2 Emission')

st.header('User Input Parameter')
CO2 = st.slider('Select Year', 2015, 2065)
number = CO2 - 2014
#st.write(number)

#Create a list
ListCO2 = []
for i in range(2015, 2065):
    ListCO2.append(i)
#save all the years in the list till the user has selected
N = CO2 + 1
temp = ListCO2.index(N)
res = ListCO2[:temp]
res_array = np.array(res)
#st.write(type(res_array))

# fit model
data=pd.read_excel('CO2 dataset.xlsx', header=0, index_col=0, parse_dates=True)
# prepare data
X = data.values
X = X.astype('float32')

model = ARIMA(X, order=(3,1,0))
model_fit = model.fit()

forecast=model_fit.forecast(steps=number)[0]
#st.write(type(forecast))
st.write("The CO2 Emission of the Year you entered", forecast[number - 1])

#dataframe of a table for chart.

#create a table to display values.
fig = go.Figure(data=[go.Table(header=dict(values=['Year', 'CO2'],fill_color = '#FD8E72',align = 'center', font_color = 'black'),
                 cells=dict(values=[res_array, forecast],fill_color = '#E5ECF6',align = 'left',font_color = 'black'))
                     ])
fig.update_layout(margin=dict(l=5,r=5,b=10,t=10))
st.write(fig)

