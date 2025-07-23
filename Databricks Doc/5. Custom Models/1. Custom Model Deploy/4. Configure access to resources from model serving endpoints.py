# Databricks notebook source
# MAGIC %md
# MAGIC # Configure access to resources from model serving endpoints
# MAGIC
# MAGIC This notebook example demonstrates how you can securely store credentials in a Databricks secrets scope and reference those secrets in model serving. This allows credentials to be fetched at serving time from model serving endpoints.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC &nbsp;
# MAGIC - This functionality currently is only supported via the Databricks REST API.
# MAGIC - To use this feature, you must store credentials like your API key or other tokens as a [Databricks secret](https://docs.databricks.com/api/workspace/secrets).
# MAGIC - The endpoint creator must have `Read` access to the Databricks secrets being referenced in the endpoint configuration.
# MAGIC - This notebook requireds Databricks SDK version 0.6.0 and above.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0: Install and upgrade any dependencies if necessary
# MAGIC &nbsp;
# MAGIC

# COMMAND ----------

# MAGIC %pip install databricks-sdk faiss-cpu --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Add secrets to Databricks Secret Store

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# COMMAND ----------

# MAGIC %md
# MAGIC You can modify the following variables to assign your secret and its corresponding key and values.

# COMMAND ----------

secret_scope_name = "MY_SECRET_SCOPE"
secret_key_name = "MY_SECRET_KEY_NAME"
secret_value = "<MY_SECRET_VALUE>"

databricks_host = "<YOUR_DATABRICKS_HOST>"
databricks_api_token = "<YOUR_DATABRICKS_API_TOKEN>"

# Add secrets to desired secret scope and key in Databricks Secret Store
w = WorkspaceClient(host=databricks_host, token=databricks_api_token)
w.secrets.create_scope(scope=secret_scope_name)
w.secrets.put_secret(scope=secret_scope_name, key=secret_key_name, string_value=secret_value)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Upload sample document data into DBFS
# MAGIC &nbsp;
# MAGIC - Download sample document data locally from https://github.com/mlflow/mlflow/blob/master/tests/langchain/state_of_the_union.txt.
# MAGIC - Click Catalog > Browse DBFS
# MAGIC - Upload the file into a FileStore location in DBFS

# COMMAND ----------

# List the file to verify that the file has been uploaded successfully.
# You might need to modify the path below based on where you uploaded the data.
dbutils.fs.ls("/FileStore/state_of_the_union.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Log and register the LangChain model
# MAGIC &nbsp;
# MAGIC This example model is adapted from https://github.com/mlflow/mlflow/blob/master/examples/langchain/retrieval_qa_chain.py.

# COMMAND ----------

import mlflow
import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# COMMAND ----------

# MAGIC %md
# MAGIC In this section, you can create a vector database and persist that database to a local file store folder. You also create a RetrievalQA chain and log it start your model run.

# COMMAND ----------

registered_model_name = "my_langchain_model"

os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope=secret_scope_name, key=secret_key_name)

# Create the vector db, persist the db to a local fs folder
persist_dir = "/tmp/faiss_index"
loader = TextLoader("/dbfs/FileStore/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
db.save_local(persist_dir)

# Create the RetrievalQA chain
retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=db.as_retriever())

# Log the RetrievalQA chain
def load_retriever(persist_directory):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(persist_directory, embeddings)
    return vectorstore.as_retriever()
  
with mlflow.start_run() as run:
    logged_model = mlflow.langchain.log_model(
        retrievalQA,
        artifact_path="retrieval_qa",
        loader_fn=load_retriever,
        persist_dir=persist_dir,
        registered_model_name=registered_model_name
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Next, you can load that logged model using MLflow `pyfunc`.

# COMMAND ----------

# Load and test the RetrievalQA chain
loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
loaded_model.predict([{"query": "What did the president say about Ketanji Brown Jackson"}])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create and query the serving endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC In this section you create a serving endpoint to serve your model and query it.

# COMMAND ----------

import requests

# COMMAND ----------

endpoint_name = "my_serving_endpoint"
model_name = registered_model_name
databricks_api_token = "<YOUR-API-TOKEN>"

# Create your endpoint
data = {
    "name": endpoint_name,
    "config": {
        "served_entities": [
            {
                "entity_name": model_name,
                "entity_version": model_version,
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
                "environment_vars": {
                    "OPENAI_API_KEY": f"{{{{secrets/{secret_scope_name}/{secret_key_name}}}}}"
                }
            }
        ]
    }
}

headers = {
    "Context-Type": "text/json",
    "Authorization": f"Bearer {databricks_api_token}"
}

response = requests.post(
    url=f"https://{databricks_host}/api/2.0/serving-endpoints",
    json=data,
    headers=headers
)
print("Response status:", response.status_code)
print("Response text:", response.text, "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC You can use the following code to check the endpoint status to verify it is ready.

# COMMAND ----------


data = {
    "name": endpoint_name
}

headers = {
    "Context-Type": "text/json",
    "Authorization": f"Bearer {databricks_api_token}"
}

response = requests.get(
    url=f"https://{databricks_host}/api/2.0/serving-endpoints/{endpoint_name}",
    json=data,
    headers=headers
)

print(response.status_code, response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, you can query the endpoint with sample data.

# COMMAND ----------

data_json = {
    "instances": [
        {
            "query": "What did the president say about Ketanji Brown Jackson"
        }
    ]
}

headers = {
    "Context-Type": "application/json",
    "Authorization": f"Bearer {databricks_api_token}"
}

response = requests.post(
    url=f"https://{databricks_host}/serving-endpoints/{endpoint_name}/invocations",
    headers=headers,
    json=data_json
)

if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
predictions = response.json()["predictions"]
predictions
