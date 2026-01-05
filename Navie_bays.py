import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="Naive Bayes App")

st.title("Naive Bayes Prediction App")
st.write("Enter only important feature values")

# ===============================
# FILE NAMES
# ===============================
CSV_FILE = "Bank_Personal_Loan_Modelling.csv"
MODEL_FILE = "naive_bayes_model.pkl"
ENCODER_FILE = "label_encoders.pkl"

# ===============================
# LOAD DATASET
# ===============================
data = pd.read_csv(CSV_FILE)

# ===============================
# IMPORTANT FEATURES ONLY
# ===============================
USEFUL_FEATURES = [
    "Age",
    "Income",
    "Family",
    "CCAvg",
    "Education",
    "Securities Account",
    "CD Account",
    "Online"
]

# ===============================
# CREATE / LOAD LABEL ENCODERS
# ===============================
if (not os.path.exists(ENCODER_FILE)) or os.path.getsize(ENCODER_FILE) < 100:
    le_dict = {}
    for col in data.columns:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            le_dict[col] = le
    joblib.dump(le_dict, ENCODER_FILE)
else:
    le_dict = joblib.load(ENCODER_FILE)

# ===============================
# SPLIT FEATURES & TARGET
# ===============================
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ===============================
# LOAD / TRAIN MODEL
# ===============================
if not os.path.exists(MODEL_FILE):
    model = GaussianNB()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
else:
    model = joblib.load(MODEL_FILE)

# ===============================
# FRONTEND INPUTS
# ===============================
input_data = {}

for col in USEFUL_FEATURES:
    if col in le_dict:
        val = st.selectbox(col, le_dict[col].classes_)
        input_data[col] = le_dict[col].transform([val])[0]
    else:
        input_data[col] = st.number_input(col, value=0.0)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict"):
    final_input = []
    for col in X.columns:
        if col in input_data:
            final_input.append(input_data[col])
        else:
            final_input.append(0)

    prediction = model.predict([final_input])[0]

    target_col = data.columns[-1]
    if target_col in le_dict:
        prediction = le_dict[target_col].inverse_transform([prediction])[0]

    st.success(f"Predicted Result : {prediction}")
