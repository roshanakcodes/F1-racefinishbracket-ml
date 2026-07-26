import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

st.set_page_config(page_title="F1 Finish Predictor", page_icon="🏎️", layout="centered")

st.title("F1 Finish Predictor 🏎️")
st.write("Predict a driver's finish, based on qualifying data and team pace.")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('f1_predictor_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading files. Make sure 'f1_predictor_model.keras' and 'scaler.pkl' exist in this folder.\nDetails: {e}")
    st.stop()

st.header("Pre-Race Conditions")

col1, col2 = st.columns(2)

with col1:
    quali_pos = st.slider("Qualifying Position (P1 - P20)", min_value=1, max_value=20, value=3)
    grid_penalty = st.slider("Grid Penalty (Grid Slots Dropped)", min_value=0, max_value=10, value=0)
    constructor_pos = st.slider("Constructor Standing Position (1st - 10th)", min_value=1, max_value=10, value=2)

with col2:
    quali_delta = st.number_input("Quali Pace Delta to Pole (Seconds)", min_value=0.0, max_value=5.0, value=0.12, step=0.01)
    team_ppg = st.number_input("Team Points Per Race (PPG)", min_value=0.0, max_value=44.0, value=25.0, step=0.5)

effective_grid = min(20, quali_pos + grid_penalty)
st.caption(f"**Effective Starting Grid Slot:** P{effective_grid}")

# Features matching exact training schema order:
# [QualiPosition, GridPenalty, QualiPaceDelta, TeamPPG, ConstructorPos]
input_data = pd.DataFrame([{
    'QualiPosition': quali_pos,
    'GridPenalty': grid_penalty,
    'QualiPaceDelta': quali_delta,
    'TeamPPG': team_ppg,
    'ConstructorPos': constructor_pos
}])

st.markdown("---")

if st.button("Predict Race Outcome", type="primary", use_container_width=True):
    scaled_input = scaler.transform(input_data)
    predictions = model.predict(scaled_input)[0]
    
    brackets = [
        "P1 - P3 : Let's goooo! Podium mate, PODIUM", 
        "P4 - P10 - We got some points today guys !!!", 
        "P11 - P20 - Mid weekend"
    ]
    predicted_index = int(np.argmax(predictions))
    
    st.subheader(f"Race Result (predicted): **{brackets[predicted_index]}**")
    st.write(f"Confidence: **{predictions[predicted_index] * 100:.1f}%**")
    
    st.write("#### Probability:")
    for bracket_name, prob in zip(brackets, predictions):
        st.write(f"**{bracket_name}** - {prob * 100:.1f}%")
        st.progress(float(prob))

    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px; padding: 10px; font-size: 14px; color: gray;">
            <hr>
            Built by Your Name 🏎️
        </div>
        """,
        unsafe_allow_html=True
    )