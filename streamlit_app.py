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

# load images
if st.session_state.sex == 'male':
    image = Image.open('.\images\male.png')
    st.image(image, caption='You selected "male" gender.')
elif st.session_state.sex == 'female':
    image = Image.open('.\images\female.png')
    st.image(image, caption='You selected "female" gender.')
elif st.session_state.sex == 'other':
    image = Image.open('.\images\no_gender.png')
    st.image(image, caption='You selected "other" gender.')    

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
if st.button('predict bill'):
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    st.header('The health insurance bill is expected to be {} dollars.'.format(prediction))



