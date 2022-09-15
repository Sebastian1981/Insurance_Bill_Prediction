# Web app and web api to predict insurance bills

- first check out the experiment folder with the notebook
- then check out the main folder with the streamlit app implementation
- finally check out the flask-api in the respective folder
- note: that the app and the api are in different folders having different requirement files and docker files; mixing them is no good
- note: the streamlit api is kinda selfexplaining whereas the api can be tested like this:\
    `import requests`\
    `url = 'https://insurancebillapi.azurewebsites.net/predict_api'`\
    `pred = requests.post(url,json={'age':35, 'sex':'male', 'bmi':59, 'children':1, 'smoker':'no', 'region':'northwest'})`\
    `print(pred.json())`