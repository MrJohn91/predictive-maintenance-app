import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #1E1E1E;  /* Dark Background */
        color: white;  /* Light Text */
    }
    .sidebar {
        background-color: #333333;  /* Dark Sidebar */
        color: white;
    }
    h1 {
        color: white;  /* White Title */
        text-align: center;
    }
    .test-sample-box {
        background-color: #F4E1D2;  /* Light Brown Background for Test Data Section */
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        color: #4D2C0C;  /* Dark Brown Text */
    }
    .metric-value {
        font-size: 20px !important;  /* Adjust this to make metric values smaller, but readable */
    }
    .metric-label {
        font-size: 14px !important;  /* Adjust the size of the metric label if needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1>Predictive Maintenance for Vehicle Engines</h1>", unsafe_allow_html=True)

model = joblib.load('svm_predictive_maintenance_model.pkl')
scaler = joblib.load('scaler.pkl')  

# Load the test data
test_data = pd.read_csv('test_data.csv')

# Sidebar
st.sidebar.header("Choose a Sample for Prediction")

sample_index = st.sidebar.selectbox("Pick a Vehicle Engine Data:", test_data.index)

selected_sample = test_data.iloc[sample_index]

features_real = selected_sample.drop('Engine Condition')  
actual_condition = selected_sample['Engine Condition']  

st.markdown('<div class="test-sample-box">', unsafe_allow_html=True)
st.write("## Vehicle Readings")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric(label="Engine RPM", value=features_real['Engine rpm'], label_visibility="visible")
    st.metric(label="Lub Oil Pressure", value=features_real['Lub oil pressure'], label_visibility="visible")

with col2:
    st.metric(label="Fuel Pressure", value=features_real['Fuel pressure'], label_visibility="visible")
    st.metric(label="Coolant Pressure", value=features_real['Coolant pressure'], label_visibility="visible")

with col3:
    st.metric(label="Lub Oil Temp (°C)", value=features_real['lub oil temp'], label_visibility="visible")
    st.metric(label="Coolant Temp (°C)", value=features_real['Coolant temp'], label_visibility="visible")

st.markdown('</div>', unsafe_allow_html=True)

features_for_model = features_real.values.reshape(1, -1)  

scaled_input = scaler.transform(features_for_model)

prediction = model.predict(scaled_input)

st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
if prediction == 1:
    st.subheader("🔴 Needs Maintenance")
    st.write("The engine has reached thresholds, and maintenance is advised.")
else:
    st.subheader("🟢 No Maintenance Needed")
    st.write("The engine is operating within normal parameters.")
st.markdown('</div>', unsafe_allow_html=True)



import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("API key not found. Please check your .env file.")

def generate_gpt3_response(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # model
        messages=[
            {"role": "system", "content": "You are a helpful maintenance assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# Sidebar
st.sidebar.header("Chat with the Maintenance Assistant")

# User input
user_input = st.sidebar.text_input("Ask me anything about vehicle maintenance:")

# Generate a response
if user_input:
    with st.spinner("generating a response..."):
        chatbot_response = generate_gpt3_response(user_input)
        st.sidebar.markdown(f"**Assistant:** {chatbot_response}")


