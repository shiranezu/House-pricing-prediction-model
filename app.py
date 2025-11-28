import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

# Load model from joblib file
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

# Create inputs for each dataset feature
for feature in X.columns:
    inputs[feature] = st.number_input(
        feature,
        value=float(X[feature].mean())
    )

input_df = pd.DataFrame([inputs])

if st.button("Predict House Price"):
    prediction = model.predict(input_df)[0] * 100000
    st.success(f"Predicted Median House Value: **${prediction:,.2f}**")



# st.write("### 📊 Feature Distribution (Bar Chart)")

# selected_feature = st.selectbox(
#     "Select a feature to visualize:",
#     X.columns
# )

# # Bin continuous values into categories (bar chart works better this way)
# binned_values = pd.cut(X[selected_feature], bins=20)

# value_counts = binned_values.value_counts().sort_index()

# fig, ax = plt.subplots(figsize=(10, 4))
# ax.bar(value_counts.index.astype(str), value_counts.values)
# ax.set_title(f"Bar Chart of {selected_feature}")
# ax.set_xlabel("Binned Values")
# ax.set_ylabel("Frequency")

# plt.xticks(rotation=45)

# st.pyplot(fig)


st.write("Histogram Feature Distribution")

selected_feature = st.selectbox(
    "Select a feature to visualize:",
    X.columns
)

fig, ax = plt.subplots()
ax.bar(X[selected_feature], bins=30, edgecolor='black', color= 'grey')
ax.set_title(f"Distribution of {selected_feature}")
ax.set_xlabel(selected_feature)
ax.set_ylabel("Frequency")

st.pyplot(fig)