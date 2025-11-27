import pandas as pd
import joblib
from math import sqrt 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target


model_type = "LinearRegression"   # change to "Ridge" or "Lasso" - Model type selection

if model_type == "LinearRegression":
    model = LinearRegression()
elif model_type == "Ridge":
    model = Ridge(alpha=1.0)
elif model_type == "Lasso":
    model = Lasso(alpha=0.1)
else:
    raise ValueError("Invalid model type")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# calculates accuracy
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
# rmse = mean_squared_error(y_test, y_pred, squared=False) # alternative way, but uses a deprecated parameter
r2 = r2_score(y_test, y_pred)

print(f"Model: {model_type}")
print("RMSE:", rmse)
print("R²:", r2)


joblib.dump(model, f"{model_type}_california.joblib")
print(f"Saved as {model_type}_california.joblib")
