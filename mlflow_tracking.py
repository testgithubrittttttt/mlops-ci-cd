import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load dataset
data = pd.DataFrame({'X': [1, 2, 3, 4, 5, 6], 'y': [2, 4, 6, 8, 10, 12]})
X_train, X_test, y_train, y_test = train_test_split(data[['X']], data['y'], test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("Linear Regression Experiment")

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_metric("mse", mse)

    # Save the model
    mlflow.sklearn.log_model(model, "linear_regression_model")

    print(f"Logged MSE: {mse}")

print("Run 'mlflow ui' to visualize the experiment logs.")
