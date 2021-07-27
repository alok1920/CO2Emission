import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA


st.title('CO2 Emission')

st.header('User Input Parameter')
CO2 = st.slider('Till waht year you like to see CO2 emission', 2015, 2065)
number = CO2 - 2014
st.write(number)


# fit model
train=pd.read_excel('CO2 dataset.xlsx', header=0, index_col=0, parse_dates=True)
# prepare data
X = train.values
X = X.astype('float32')

model = ARIMA(X, order=(3,1,0))
model_fit = model.fit()

forecast=model_fit.forecast(steps=number)[0]

st.write(forecast)