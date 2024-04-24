import pickle
import streamlit as st
import numpy as np


model = pickle.load(open('LR-model.pkl', 'rb'))

def predict(values):
    return model.predict(values)

def main():
    st.title("Score Predictor")

    # Use st.slider instead of st.number_input for better user experience
    a = st.slider("Distance Covered (meters)", min_value=0.0, max_value=10.0, step=0.1)
    b = st.slider("Goals Scored", min_value=0, max_value=100, step=1)
    c = st.slider("Minutes to Goal Ratio", min_value=0.0, max_value=100.0, step=0.1)
    d = st.slider("Shots Per Game", min_value=0, max_value=20, step=1)
    e = st.slider("Agent Charges (USD)", min_value=0, max_value=100, step=1)
    f = st.slider("BMI", min_value=10.0, max_value=40.0, step=0.1)
    g = st.slider("Cost (USD)", min_value=0, max_value=200, step=1)
    h = st.slider("Previous Club Cost (USD)", min_value=0, max_value=200, step=1)
    i = st.slider("Height (cm)", min_value=100, max_value=250, step=1)
    j = st.slider("Weight (kg)", min_value=30, max_value=150, step=1)
    
    # Convert the inputs to a numpy array and reshape it for the model
    inps = np.array([a, b, c, d, e, f, g, h, i, j]).reshape(1, -1)
    
    # Predict the score and display it
    score = predict(inps)
    st.text(f"Predicted Score: {score[0]:.2f}")
    
if __name__ == '__main__':
    main()