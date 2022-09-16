import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import *
from PIL import Image

st.title('Demo App: Predicting Health Insurance Costs for Weini')

# Initialize session state vars
if 'sex' not in st.session_state:
    st.session_state.sex = 'male'
if 'smoker' not in st.session_state:
    st.session_state.smoker = 'no'

# load weini image
if st.session_state.sex == 'male' and st.session_state.smoker == 'no':
    image = Image.open('./images/basti_original.png')
    st.image(image, caption='This is handsome Basti!')
elif st.session_state.sex == 'male' and st.session_state.smoker == 'yes':
    image = Image.open('./images/basti_original_smoking.png')
    st.image(image, caption='This is unhealthy Basti!')

elif st.session_state.sex == 'female' and st.session_state.smoker == 'no':
    image = Image.open('./images/basti_female.png')
    st.image(image, caption='This is handsome Basti at its best!')
elif st.session_state.sex == 'female' and st.session_state.smoker == 'yes':
    image = Image.open('./images/basti_female_smoking.png')
    st.image(image, caption='This is unhealthy Basti at its best!')

elif st.session_state.sex == 'other' and st.session_state.smoker == 'no':
    image = Image.open('./images/basti_other.png')
    st.image(image, caption='Handsome Basti at its very best!')
elif st.session_state.sex == 'other' and st.session_state.smoker == 'yes':
    image = Image.open('./images/basti_other_smoking.png')
    st.image(image, caption='Unhealthy Basti at its very best!')    

# load trained model
model = load_model('deployment_14092022')

# input features
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
age = st.slider('select Basti´s age', 15,85,40)
sex = st.selectbox('choose Bastis perceived gender', options = ['male', 'female','other'], key='sex')
children = st.slider('select Basti´s children', 0,5,1)
smoker = st.selectbox('choose if Basti a smoker', options = ['no', 'yes'], key='smoker')
bmi = st.slider('choose Bastis perceived bmi', 15,55,30)
region = st.selectbox('choose Basti´s desired region', options = ['southwest', 'southeast', 'northwest', 'northeast', 'other'])
input = np.array([age, sex, bmi, children, smoker, region])
data_unseen = pd.DataFrame([input], columns = features)

# make new prediction
if st.button('predict bill'):
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    st.header('Basti´s health insurance bill is expected to be {} happy dollars.'.format(prediction))



