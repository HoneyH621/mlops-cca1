import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# 1. Load the dataset
data = pd.read_csv("data/student_data.csv")

# 2. Select input features and target
X = data[["Study_Hours", "Attendance", "Previous_Score"]]
y = data["Result"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions
predictions = model.predict(X_test)

# 6. Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# 7. Start MLflow experiment tracking
mlflow.set_experiment("Student Performance Experiment")

with mlflow.start_run():

    mlflow.log_param("model", "DecisionTreeClassifier")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
    model,
    "model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

    print("Model Accuracy:", accuracy)
    print("MLflow tracking completed successfully.")

# 8. Save the model
joblib.dump(model, "student_model.pkl")

print("Model saved successfully as student_model.pkl")