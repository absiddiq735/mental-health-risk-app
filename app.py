import streamlit as st
from predictor import MentalHealthRiskPredictor

st.set_page_config(
    page_title="Mental Health Risk Prediction",
    layout="wide"
)

@st.cache_resource
def load_model():
    return MentalHealthRiskPredictor("deployment_artifacts")

predictor = load_model()

st.title("Transformer-Based Mental Health Risk Prediction")
st.warning(
    "This tool is for research demonstration only. It is not a medical diagnosis or crisis intervention system."
)

text = st.text_area("Enter patient text:", height=180)

threshold = st.slider(
    "Emotion threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

if st.button("Predict"):
    if len(text.strip()) < 3:
        st.error("Please enter a longer text.")
    else:
        result = predictor.predict(text, threshold)

        st.subheader("Prediction Result")
        st.metric("Risk Level", result["risk_level"])
        st.metric("Risk Score", result["risk_score"])
        st.metric("Confidence", result["confidence"])

        st.write("### Predicted Emotions")
        st.write(result["predicted_emotions"])

        st.write("### Patient Cluster")
        st.write(result["cluster_name"])

        st.write("### Emotion Probabilities")
        st.json(result["emotion_probabilities"])