import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Predict function
def pred(Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp):
     # Convert Gender to numeric (0 for Male, 1 for Female)
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1
    features = np.array([[Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]])
    prediction = xgb.predict(features).reshape(1, -1)
    return prediction[0]

# Load model from pickle file
with open('xgboost_model.pkl', 'rb') as file:
    xgb = pickle.load(file)

# Load X_train data
X_train = pd.read_csv('X_train.csv')

# Map 0 and 1 to Male and Female
gender_map = {0: "Male", 1: "Female"}
X_train['Gender'] = X_train['Gender'].map(gender_map)

# Web App
st.set_page_config(page_title="Calories Burn Prediction", layout="wide", initial_sidebar_state="expanded")

# Custom styling
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
    }
    .subheader {
        font-size: 22px;
        text-align: center;
        color: #888;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px;
    }
    .stSelectbox>div>div>input {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and image
st.markdown('<div class="title">Calories Burn Prediction</div>', unsafe_allow_html=True)
st.image('image.jpg', width=200)

# Sidebar
st.sidebar.header('Input Features')

# Gender (Radio Buttons for selection)
Gender = st.sidebar.radio('Gender', X_train['Gender'].unique())

# Age (Number Input with range)
Age = st.sidebar.number_input('Age', min_value=int(X_train['Age'].min()), max_value=int(X_train['Age'].max()), step=1)

# Height (Number Input with range)
Height = st.sidebar.number_input('Height (cm)', min_value=int(X_train['Height'].min()), max_value=int(X_train['Height'].max()), step=1)

# Weight (Number Input with range)
Weight = st.sidebar.number_input('Weight (kg)', min_value=int(X_train['Weight'].min()), max_value=int(X_train['Weight'].max()), step=1)

# Duration (Number Input for exercise duration)
Duration = st.sidebar.number_input('Duration (minutes)', min_value=int(X_train['Duration'].min()), max_value=int(X_train['Duration'].max()), step=1)

# Heart Rate (Number Input for heart rate)
Heart_Rate = st.sidebar.number_input('Heart Rate (bpm)', min_value=int(X_train['Heart_Rate'].min()), max_value=int(X_train['Heart_Rate'].max()), step=1)

# Body Temperature (Number Input for body temperature)
Body_Temp = st.sidebar.number_input('Body Temperature (°C)', min_value=float(X_train['Body_Temp'].min()), max_value=float(X_train['Body_Temp'].max()), step=0.1)

# Button to trigger prediction
if st.sidebar.button("Submit"):
    result = pred(Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp)
    
    # Extract the scalar value from the numpy.ndarray and format it
    result_value = result.item()  # Converts the single value ndarray to a scalar

    st.markdown('<div class="subheader">Amount of Calories Burnt: </div>', unsafe_allow_html=True)
    st.write(f"**{result_value:.2f} Calories**")
