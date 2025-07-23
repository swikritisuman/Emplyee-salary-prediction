import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- Page Config & Styling ----------
st.set_page_config(page_title=" Employee Salary Prediction", page_icon="💼", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
    }
    .stApp {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
        color: white;
    }
    h1 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Models with Caching ----------
@st.cache_resource
def load_model():
    return joblib.load("best_model (1).pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load("label_encoders.pkl")

model = load_model()
scaler = load_scaler()
label_encoders = load_encoders()

# ---------- Title & Instructions ----------
st.title("💼 Employee Salary Prediction")
st.markdown("Enter employee details to predict whether income is **>50K** or **<=50K** 👇")

# ---------- User Input Form ----------
def user_input():
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("👤 Age", 18, 100, step=1)
        workclass = st.selectbox("🏢 Workclass", label_encoders['workclass'].classes_)
        education = st.selectbox("🎓 Education", label_encoders['education'].classes_)
        educational_num = st.slider("📚 Education Number", 1, 16)
        marital_status = st.selectbox("💍 Marital Status", label_encoders['marital-status'].classes_)

    with col2:
        occupation = st.selectbox("🛠️ Occupation", label_encoders['occupation'].classes_)
        relationship = st.selectbox("👨‍👩‍👧 Relationship", label_encoders['relationship'].classes_)
        race = st.selectbox("🌍 Race", label_encoders['race'].classes_)
        gender = st.selectbox("🚻 Gender", label_encoders['gender'].classes_)
        hours_per_week = st.slider("🕒 Hours Per Week", 1, 99)

    with st.expander("📦 Additional Info (Optional)", expanded=False):
        capital_gain = st.number_input("💰 Capital Gain", 0, 99999, step=100)
        capital_loss = st.number_input("📉 Capital Loss", 0, 99999, step=100)
        native_country = st.selectbox("🌎 Native Country", label_encoders['native-country'].classes_)

    # Encode user input
    return pd.DataFrame([{
        'age': age,
        'workclass': label_encoders['workclass'].transform([workclass])[0],
        'education': label_encoders['education'].transform([education])[0],
        'educational-num': educational_num,
        'marital-status': label_encoders['marital-status'].transform([marital_status])[0],
        'occupation': label_encoders['occupation'].transform([occupation])[0],
        'relationship': label_encoders['relationship'].transform([relationship])[0],
        'race': label_encoders['race'].transform([race])[0],
        'gender': label_encoders['gender'].transform([gender])[0],
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': label_encoders['native-country'].transform([native_country])[0],
    }])

# ---------- Predict Button ----------
input_df = user_input()

if st.button("🔍 Predict Income"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    income = label_encoders['income'].inverse_transform([prediction])[0]

    st.success(f"💰 **Predicted Income: {income}**")

# ---------- Footer ----------
st.markdown("<hr><center style='color: gray;'>Made with ❤️ by Swikriti Suman</center>", unsafe_allow_html=True)
