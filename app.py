import streamlit as st
import pandas as pd
import numpy as np
import pickle

# predict function
def pred(Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp):
    features = np.array([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]])
    prediction = xgb.predict(features).reshape(1,-1)
    return prediction[0]

# load model from pickle file
with open('xgboost_model.pkl','rb') as file:
    xgb = pickle.load(file)

# load X_train data
X_train = pd.read_csv('X_train.csv')

# Web App
st.markdown("### Calories Burn Prediction")
st.image('image.jpg',width=100)

Gender = st.selectbox('Gender', X_train['Gender'])
Age = st.selectbox('Age', X_train['Age'])
Height = st.selectbox('Height', X_train['Height'])
Weight = st.selectbox('Weight', X_train['Weight'])
Duration = st.selectbox('Duration', X_train['Duration'])
Heart_Rate = st.selectbox('Heart_Rate', X_train['Heart_Rate'])
Body_Temp = st.selectbox('Body_Temp', X_train['Body_Temp'])

result = pred(Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp)

if st.button("Submit"):
    if result:
        st.write("Amount of Calories Burnt :",result)