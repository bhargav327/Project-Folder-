
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# -----------------------
# Load models
# -----------------------
models = {
    "Logistic Regression": joblib.load("model/logistic_model.pkl"),
    "Decision Tree": joblib.load("model/decision_tree_model.pkl"),
    "KNN": joblib.load("model/knn_model.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes_model.pkl"),
    "Random Forest": joblib.load("model/random_forest_model.pkl")
}

scaler = joblib.load("model/scaler.pkl")

# -----------------------
# Streamlit UI
# -----------------------
st.title("Stroke Prediction ML Models")

st.write("BITS Machine Learning Assignment 2")

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    X = data.drop("stroke", axis=1)
    y = data["stroke"]

    model = models[model_name]

    if model_name == "KNN":
        X_input = scaler.transform(X)
    else:
        X_input = X

    y_pred = model.predict(X_input)
    y_prob = model.predict_proba(X_input)[:,1]

    st.subheader("Evaluation Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy",
                  round(accuracy_score(y, y_pred),3))
        st.metric("Precision",
                  round(precision_score(y,y_pred,zero_division=0),3))
        st.metric("F1 Score",
                  round(f1_score(y,y_pred,zero_division=0),3))

    with col2:
        st.metric("AUC",
                  round(roc_auc_score(y,y_prob),3))
        st.metric("Recall",
                  round(recall_score(y,y_pred,zero_division=0),3))
        st.metric("MCC",
                  round(matthews_corrcoef(y,y_pred),3))

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred,
        ax=ax,
        cmap="Blues"
    )

    st.pyplot(fig)
