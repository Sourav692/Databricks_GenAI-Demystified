# Databricks notebook source
# DBTITLE 1,a
# MAGIC %md
# MAGIC # Provisioned Throughput BGE serving example
# MAGIC
# MAGIC Provisioned Throughput provides optimized inference for Foundation Models with performance guarantees for production workloads.
# MAGIC
# MAGIC This example walks through:
# MAGIC
# MAGIC 1. Downloading the model from Hugging Face `transformers`
# MAGIC 2. Logging the model in a provisioned throughput supported format into the Databricks Unity Catalog or Workspace Registry
# MAGIC 3. Enabling optimized serving on the model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC - Attach a cluster with sufficient memory to the notebook
# MAGIC - Make sure to have MLflow version 2.11 or later installed
# MAGIC - Make sure to enable **Models in UC**, especially when working with models larger than 7B in size
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 1: Log the model for optimized LLM serving

# COMMAND ----------

# Update and install required dependencies
!pip install -U mlflow
!pip install -U transformers
!pip install -U accelerate
dbutils.library.restartPython()

# COMMAND ----------

from transformers import AutoModel, AutoTokenizer

bge = "BAAI/bge-large-en-v1.5"
model = AutoModel.from_pretrained(bge)
tokenizer = AutoTokenizer.from_pretrained(bge)

# COMMAND ----------

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema, TensorSpec
import numpy as np

# Define the model input and output schema
input_schema = Schema([ColSpec(type="string", name=None)])
output_schema = Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define an example input
input_example = {
    "input": np.array([
        "Welcome to Databricks!"
    ])
}

# COMMAND ----------

# MAGIC %md
# MAGIC To enable optimized serving, when logging the model, include the extra metadata dictionary when calling `mlflow.transformers.log_model` as shown below:
# MAGIC
# MAGIC ```
# MAGIC metadata = {"task": "llm/v1/completions"}
# MAGIC ```
# MAGIC This specifies the API signature used for the model serving endpoint.
# MAGIC

# COMMAND ----------

import mlflow

# Comment out the line below if not using Models in UC 
# and simply provide the model name instead of three-level namespace
mlflow.set_registry_uri('databricks-uc')
CATALOG = "ml"
SCHEMA = "models"
registered_model_name = f"{CATALOG}.{SCHEMA}.bge-large"

# Start a new MLflow run
with mlflow.start_run():
    components = {
        "model": model,
        "tokenizer": tokenizer,
    }
    mlflow.transformers.log_model(
        transformers_model=components,
        task = "llm/v1/embeddings",
        registered_model_name=registered_model_name,
        input_example=input_example
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: View optimization information for your model
# MAGIC
# MAGIC Modify the cell below to change the model name. After calling the model optimization information API, you will be able to retrieve throughput chunk size information for your model. This is the number of tokens/second that corresponds to 1 throughput unit for your specific model.

# COMMAND ----------

import requests
import json

# Name of the registered MLflow model
model_name = registered_model_name

# Get the latest version of the MLflow model
model_version = 1

# Get the API endpoint and token for the current notebook context

API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.get(url=f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{model_name}/{model_version}", headers=headers)

print(json.dumps(response.json(), indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Configure and create your model serving GPU endpoint
# MAGIC
# MAGIC Modify the cell below to change the endpoint name. After calling the create endpoint API, the logged MPT-7B model is automatically deployed with optimized LLM serving.

# COMMAND ----------

# Set the name of the MLflow endpoint
endpoint_name = "bge-large"

# COMMAND ----------

# Specify the minimum provisioned throughput 
min_provisioned_throughput = response.json()['throughput_chunk_size']*2

# Specify the maximum provisioned throughput 
max_provisioned_throughput = response.json()['throughput_chunk_size']*3

# COMMAND ----------

data = {
    "name": endpoint_name,
    "config": {
        "served_entities": [
            {
                "entity_name": model_name,
                "entity_version": model_version,
                "min_provisioned_throughput": min_provisioned_throughput,
                "max_provisioned_throughput": min_provisioned_throughput,
            }
        ]
    },
}

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(url=f"{API_ROOT}/api/2.0/serving-endpoints", json=data, headers=headers)

print(json.dumps(response.json(), indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## View your endpoint
# MAGIC To see your more information about your endpoint, go to the **Serving** UI section on the left navigation bar and search for your endpoint name.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Query your endpoint
# MAGIC
# MAGIC Once your endpoint is ready, you can query it by making an API request. Depending on the model size and complexity, it can take 30 minutes or more for the endpoint to get ready.  

# COMMAND ----------

API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
data = {
    "input": ["Welcome to Databricks!"]
}

headers = {
    "Context-Type": "text/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

response = requests.post(
    url=f"{API_ROOT}/serving-endpoints/{endpoint_name}/invocations",
    json=data,
    headers=headers
)

print(json.dumps(response.json()))
