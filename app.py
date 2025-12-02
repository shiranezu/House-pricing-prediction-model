import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso

st.title(" California House Price Predictor (Linear, Ridge, Lasso)")


housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target


model_choice = st.sidebar.selectbox(
    "Choose model:",
    ["LinearRegression", "Ridge", "Lasso"]
)

# Choose training or loading
option = st.sidebar.radio(
    "Model Mode:",
    ["Train new model now", "Load saved model"]
)

# Train directly inside Streamlit
if option == "Train new model now":
    if model_choice == "LinearRegression":
        model = LinearRegression()
    elif model_choice == "Ridge":
        model = Ridge(alpha=1.0)
    elif model_choice == "Lasso":
        model = Lasso(alpha=0.1)

    model.fit(X, y)
    st.sidebar.success(f"{model_choice} trained!")

else:
    filename = f"{model_choice}_california.joblib"
    try:
        model = joblib.load(filename)
        st.sidebar.success(f"{filename} loaded!")
    except:
        st.sidebar.error("Model file not found. Train one first.")
        st.stop()


st.write("### Enter feature values:")
inputs = {}


for feature in X.columns:
    inputs[feature] = st.number_input(
        feature,
        value=float(X[feature].mean())
    )

input_df = pd.DataFrame([inputs])

if st.button("Predict House Price"):
    raw_pred = model.predict(input_df)[0]
    # Convert to dollars
    prediction = raw_pred * 100000
    # Set minimum realistic price
    MIN_PRICE =  np.random(30000, 40000)  # you can choose any value
    if prediction < MIN_PRICE:
        prediction = MIN_PRICE

    st.success(f"Predicted Median House Value: **${prediction:,.2f}**")


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