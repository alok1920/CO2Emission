import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt


st.title('CO2 Emission')

st.header('User Input Parameter')
CO2 = st.slider('Select Year', 2015, 2065)
number = CO2 - 2014
#st.write(number)

#df = pd.read_excel('CO2 dataset.xlsx', header=0, index_col=0, parse_dates=True)

#adding training data
#train = pd.read_csv('dataset.csv', header=None, index_col=0,parse_dates=True, squeeze=True)
# fit model
data=pd.read_excel('CO2 dataset.xlsx', header=0, index_col=0, parse_dates=True)
# prepare data
X = data.values
X = X.astype('float32')

model = ARIMA(X, order=(3,1,0))
model_fit = model.fit()

forecast=model_fit.forecast(steps=number)[0]

st.write("The CO2 Emission of the Year you entered", forecast[number - 1])

