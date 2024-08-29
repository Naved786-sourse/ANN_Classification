import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the scaler and encoder
with open('encoder_1.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('scaler_1.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title("Good/Bad Prediction")

# Input fields for user input
A_id = st.number_input('A_id')
Size = st.number_input('Size')
Weight = st.number_input('Weight')
Sweetness = st.number_input('Sweetness')
Crunchiness = st.number_input('Crunchiness')
Juiciness = st.number_input('Juiciness')
Ripeness = st.number_input('Ripeness')
Acidity = st.number_input('Acidity')

# Collect input data into a DataFrame
input_data = pd.DataFrame({
    'A_id': [A_id],
    'Size': [Size],
    'Weight': [Weight],
    'Sweetness': [Sweetness],
    'Crunchiness': [Crunchiness],
    'Juiciness': [Juiciness],
    'Ripeness': [Ripeness],
    'Acidity': [Acidity]
})

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make a prediction using the trained model
prediction = model.predict(input_data_scaled)

# Get the prediction probability
prediction_proba = prediction[0][0]

# Display the prediction result
st.write(f'Predicted Quality Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The product is predicted to be Good.')
else:
    st.write('The product is predicted to be Bad.')
