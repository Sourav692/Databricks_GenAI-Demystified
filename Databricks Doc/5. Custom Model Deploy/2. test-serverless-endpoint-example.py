# Databricks notebook source
# MAGIC %md
# MAGIC ## Test Serverless endpoint by querying your model 
# MAGIC
# MAGIC The notebook loads an input example that was logged with the registered model, `ElasticNetDiabetes`, and queries the model to test the servereless endpoint.
# MAGIC
# MAGIC ### Prerequisites
# MAGIC
# MAGIC - Logged and registered Python-based model in the Model Registry 
# MAGIC - Model Serving endpoint ([AWS](https://docs.databricks.com/machine-learning/model-serving/index.html)|[Azure](https://docs.microsoft.com/azure/databricks/machine-learning/model-serving/index))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Provide shard URL

# COMMAND ----------

shard_url = 'https://e2-demo-field-eng.cloud.databricks.com/'
model_name = 'shared.yash_fsi_smart_claims.elasticnetdiabetes'

# COMMAND ----------

# MAGIC %md
# MAGIC The following assumes an input example was logged with your model. When possible, make sure to log an input example so that it is easy to construct a sample query to your model version.
# MAGIC
# MAGIC You also need a Databricks token to issue requests to your model endpoint. You can replace `<YOUR_TOKEN>` with your token in the following example.  
# MAGIC
# MAGIC You can generate a token on the **User Settings** page of your Databricks workspace ([AWS](https://docs.databricks.com/sql/user/security/personal-access-tokens.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/user/security/personal-access-tokens)|[GCP](https://docs.gcp.databricks.com/sql/user/security/personal-access-tokens.html)).

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def process_input(dataset):
  if isinstance(dataset, pd.DataFrame):
    return {'dataframe_split': dataset.to_dict(orient='split')}
  elif isinstance(dataset, str):
    return dataset
  else:
    return create_tf_serving_json(dataset)

def score_model(dataset):
  url = f'{shard_url}/model-endpoint/{model_name}/1/invocations'
  databricks_token = ""
  headers = {'Authorization': f'Bearer {databricks_token}'}
  data_json = process_input(dataset)
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

# MAGIC %md Call the score_model() function and pass your input example as the parameter

# COMMAND ----------

import mlflow
path = mlflow.artifacts.download_artifacts(f'models:/{model_name}/1')
model = mlflow.pyfunc.load_model(f'models:/{model_name}/1')
input_example = model.metadata.load_input_example(path)

# COMMAND ----------

score_model(input_example)

# COMMAND ----------


