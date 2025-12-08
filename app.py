import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing

# -------------------------------
# Utility functions
# -------------------------------

# Ensure prediction is always positive
def sanitize_pred(pred, min_price=1):
    return max(pred, min_price)

# Normalize longitude to [-180, 180]
def normalize_longitude(lon):
    lon = lon % 360
    if lon > 180:
        lon -= 360
    return lon

# -------------------------------
# Load dataset (feature names only)
# -------------------------------
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)

# -------------------------------
# Sidebar model selection
# -------------------------------
st.sidebar.title("Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose model:",
    ["LinearRegression", "Ridge", "Lasso"]
)

mode = st.sidebar.selectbox(
    "App Mode:",
    ["California (form)", "Load Only"]
)

# -------------------------------
# Load trained model
# -------------------------------
model_file = f"{model_choice}_california.joblib"

try:
    model = joblib.load(model_file)
    st.sidebar.success(f"Loaded: {model_file}")
except:
    st.sidebar.error(f"Could not load {model_file}. Train it first using train.py.")
    st.stop()

# -------------------------------
# CALIFORNIA FORM MODE
# -------------------------------
if mode == "California (form)":

    st.header("California Housing Price Prediction")

    st.write("Enter the house features below:")

    numeric_cols = X.columns.tolist()
    user_input = {}

    for col in numeric_cols:
        default_val = float(X[col].mean())
        val = st.number_input(col, value=default_val)

        # Normalize longitude if needed
        if col.lower() == "longitude":
            val = normalize_longitude(val)

        user_input[col] = val

    X_user = pd.DataFrame([user_input])

    if st.button("Predict"):
        raw_pred = model.predict(X_user)[0]  # model output in 100k units + offset
        raw_pred = sanitize_pred(raw_pred, min_price=1)  # Prevent zero/negative
        price_dollars = raw_pred * 100000               # convert to dollars

        st.success(f"Predicted House Price: **${price_dollars:,.2f}**")

# -------------------------------
# LOAD ONLY MODE
# -------------------------------
else:
    st.header("Model Loaded Successfully")
    st.write("This mode is for confirming that the model file loads properly.")


st.write("### Feature Distribution (Histogram)")

selected_feature = st.selectbox(
    "Select a feature to visualize:",
    X.columns
)

fig, ax = plt.subplots()
ax.hist(X[selected_feature], bins=30, edgecolor='black')
ax.set_title(f"Distribution of {selected_feature}")
ax.set_xlabel(selected_feature)
ax.set_ylabel("Frequency")

st.pyplot(fig)