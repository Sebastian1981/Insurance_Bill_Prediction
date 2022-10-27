import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import *
from PIL import Image

st.title('Demo App: Predicting Health Insurance Costs')


# Initialize session state vars
if 'sex' not in st.session_state:
    st.session_state.sex = 'male'
if 'smoker' not in st.session_state:
    st.session_state.smoker = 'no'


# load trained model
model = load_model('deployment_14092022')

# input features
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
age = st.slider('select age', 25,65,40)
sex = st.selectbox('select gender', options = ['male', 'female','other'], key='sex')
children = st.slider('select number of children', 0,5,1)
smoker = st.selectbox('select if smoker', options = ['no', 'yes'], key='smoker')
bmi = st.slider('select bmi', 20,45,30)
region = st.selectbox('select region', options = ['southwest', 'southeast', 'northwest', 'northeast', 'other'])
input = np.array([age, sex, bmi, children, smoker, region])
data_unseen = pd.DataFrame([input], columns = features)

# make new prediction
prediction = predict_model(model, data=data_unseen, round = 0)
prediction = int(prediction.Label[0])
st.subheader('Insurance Bill Prediction: {} dollar'.format(prediction))




