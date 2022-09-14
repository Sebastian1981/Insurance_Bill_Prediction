import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import *

st.title('Demo App: Predicting Insurance Costs')
# load trained model
model = load_model('deployment_14092022')

# input features
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
age = st.slider('select age', 15,85,40)
sex = st.selectbox('select sex', options = ['male', 'female','other'])
children = st.slider('select children', 0,5,1)
smoker = st.selectbox('select smoker', options = ['no', 'yes'])
bmi = st.slider('select bmi', 15,55,30)
region = st.selectbox('select region', options = ['southwest', 'southeast', 'northwest', 'northeast', 'other'])
input = np.array([age, sex, bmi, children, smoker, region])
data_unseen = pd.DataFrame([input], columns = features)

# make new prediction

if st.button('predict bill'):
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    st.header('The expected bill will be {} dollars.'.format(prediction))


