# Databricks notebook source
# MAGIC %md
# MAGIC https://assets.docs.databricks.com/_extras/notebooks/source/machine-learning/customize-model-serving-output.html

# COMMAND ----------

# MAGIC %md #Customizing model serving output with MLflow PyFunc
# MAGIC
# MAGIC This notebook demonstrates how to customize the output of a served model when the raw outputs of the queried model needs to be post-processed for consumption.
# MAGIC

# COMMAND ----------

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import mlflow
import mlflow.pyfunc
import pandas as pd

# Model wrapper class
class ModelWrapper(mlflow.pyfunc.PythonModel):
    # Initialize model in the constructor
    def __init__(self, model):
        self.model = model

    # Prediction function
    def predict(self, context, model_input):
        # Predict the probabilities and class
        prediction_probabilities = self.model.predict_proba(model_input)
        predictions = self.model.predict(model_input)

        # Create a DataFrame to hold the results
        result = pd.DataFrame(prediction_probabilities, columns=['prob_0', 'prob_1', 'prob_2'])

        # Inside the predict function:
        class_labels = ["setosa", "versicolor", "virginica"]
        predictions = self.model.predict(model_input)
        predicted_probabilities = self.model.predict_proba(model_input)
        result = pd.DataFrame(predicted_probabilities, columns=[f'prob_{label}' for label in class_labels])
        result['prediction'] = [class_labels[prediction] for prediction in predictions]
        
        return result

# Load iris dataset and split the dataset into train and test sets
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md 
# MAGIC The following wraps the model and defines the input and output schemas. From there, you can run and log the model using `pyfunc`.
# MAGIC

# COMMAND ----------

# Wrap the model in the ModelWrapper
wrapped_model = ModelWrapper(rf)

# Define the input and output schemas
input_schema = Schema([ColSpec("double", name) for name in iris.feature_names])
output_schema = Schema([ColSpec("double", f'prob_{species}') for species in iris.target_names] + [ColSpec("string", 'prediction')])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define an example input for the model
input_example = pd.DataFrame(X_train[:1], columns=iris.feature_names)
input_example = input_example.to_dict(orient='list')

# Start an MLflow run and log the model
with mlflow.start_run():
    mlflow.pyfunc.log_model("model", 
                            python_model=wrapped_model, 
                            input_example=input_example, 
                            signature=signature)

# Load the model from the run
run_id = mlflow.last_active_run().info.run_id
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

# Create a DataFrame for the test data
df_test = pd.DataFrame(X_test[:1], columns=iris.feature_names)

# Use the loaded model to predict on the test data
loaded_model.predict(df_test)

# COMMAND ----------


