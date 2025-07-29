import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_dataset
from preprocess import prepare_symptom_data
from model import train_model
from predict import predict_disease

# Load and cache model/data
@st.cache_resource
def load_all():
    df, precaution_df = load_dataset()
    X, y, mlb = prepare_symptom_data(df)
    clf, _, _ = train_model(X, y)
    return clf, mlb, precaution_df

# --- UI setup ---
st.set_page_config(page_title="Mednose", layout="centered")
st.title("🩺 Mednose")
st.markdown("Enter symptoms to get a list of likely diseases and recommended precautions.")

# Load everything
clf, mlb, precaution_df = load_all()
all_symptoms = sorted(mlb.classes_)

# --- Symptom Input ---
selected_symptoms = st.multiselect("Select symptoms", options=all_symptoms)

# --- Prediction Logic ---
if st.button("Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        st.success("Prediction complete!")

        # Use shared prediction logic
        top_diseases, unknown = predict_disease(clf, mlb, selected_symptoms)

        if unknown:
            st.warning(f"Unknown symptoms ignored: {', '.join(unknown)}")

        if not top_diseases:
            st.error("No valid symptoms were entered. Please try again.")
        else:
            # Display bar chart
            st.markdown("### 📊 Top 3 Predicted Diseases")
            fig, ax = plt.subplots()
            ax.barh([d[0] for d in reversed(top_diseases)],
                    [d[1] * 100 for d in reversed(top_diseases)])
            ax.set_xlabel("Confidence (%)")
            st.pyplot(fig)

            # List predictions + precautions
            for disease, confidence in top_diseases:
                st.markdown("---")
                st.write(f"**{disease}** — {confidence * 100:.1f}%")

                match = precaution_df[precaution_df["Disease"].str.lower() == disease.lower()]
                if not match.empty:
                    st.markdown("**🩹 Recommended Precautions:**")
                    for col in ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]:
                        val = match.iloc[0].get(col)
                        if pd.notna(val):
                            st.write(f"- {val}")

# --- Disclaimer ---
st.markdown("---")
st.caption("⚠️ This is an AI-powered tool for educational purposes only and does not replace medical advice. Please consult a healthcare professional.")
