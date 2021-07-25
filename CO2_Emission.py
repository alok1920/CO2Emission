import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

st.title('CO2 Emission')

st.header('User Input Parameter')


