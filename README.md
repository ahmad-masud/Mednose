# Mednose

This is a machine learning-powered web app that predicts likely diseases based on selected symptoms using a Random Forest classifier. Built with **Streamlit**, it's fully interactive and runs in your browser.

## Features
- Select symptoms from a checklist
- Get top-3 predicted diseases with confidence scores
- View recommended precautions for each disease
- Visualize predictions with a confidence bar chart
- Medical disclaimer included

## Dataset
Uses the Kaggle dataset: [`Disease and Symptoms Dataset`](https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset)

Fetched using `kagglehub`.

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the app:
```bash
streamlit run main.py
```

3. Open `http://localhost:8501` in your browser.

## Requirements

All dependencies are listed in `requirements.txt`, including:
- pandas
- scikit-learn
- streamlit
- matplotlib
- kagglehub

## Disclaimer
> This app is for **educational purposes only** and is **not a substitute for professional medical advice**. Always consult a healthcare provider for medical diagnosis and treatment.
