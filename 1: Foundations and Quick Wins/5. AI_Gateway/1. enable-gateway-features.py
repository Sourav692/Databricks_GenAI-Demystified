# Databricks notebook source
# MAGIC %md
# MAGIC [Link](https://docs.databricks.com/aws/en/notebooks/source/ai-gateway/enable-gateway-features.html) 

# COMMAND ----------

# MAGIC %md
# MAGIC [Mosaic AI Gateway introduction](https://docs.databricks.com/aws/en/ai-gateway/)
# MAGIC
# MAGIC [Configure AI Gateway on model serving endpoints](https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints)
# MAGIC
# MAGIC [Monitor served models using AI Gateway-enabled inference tables](https://docs.databricks.com/aws/en/ai-gateway/inference-tables)

# COMMAND ----------

# MAGIC %md
# MAGIC # Enable Databricks Mosaic AI Gateway features
# MAGIC
# MAGIC This notebook shows how to enable and use Databricks Mosaic AI Gateway features to manage and govern models from providers, such as OpenAI and Anthropic. 
# MAGIC
# MAGIC In this notebook, you use the Model Serving and AI Gateway API to accomplish the following tasks:
# MAGIC
# MAGIC - Create and configure an endpoint for OpenAI GPT-4o-Mini.
# MAGIC - Enable AI Gateway features including usage tracking, inference tables, guardrails, and rate limits. 
# MAGIC - Set up personally identifiable information (PII) detection for model requests and responses.
# MAGIC - Implement rate limits for model serving endpoints.
# MAGIC - Configure multiple models for A/B testing.
# MAGIC - Enable fallbacks for failed requests.
# MAGIC
# MAGIC If you prefer a low-code experience, you can create an external models endpoint and configure AI Gateway features using the Serving UI ([AWS](https://docs.databricks.com/ai-gateway/configure-ai-gateway-endpoints.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/ai-gateway/configure-ai-gateway-endpoints) | [GCP](https://docs.databricks.com/gcp/ai-gateway/configure-ai-gateway-endpoints)).

# COMMAND ----------

# MAGIC %pip install --quiet openai
# MAGIC %restart_python

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

DATABRICKS_HOST = w.config.host

# name of model serving endpoint
ENDPOINT_NAME = "<endpoint_name>"

# catalog and schema for inference tables
CATALOG_NAME = "<catalog_name>"
SCHEMA_NAME = "<schema_name>"

# openai API key in Databricks Secrets
SECRETS_SCOPE = "<secrets_scope>"
SECRETS_KEY = "OPENAI_API_KEY"

# if you need to add an OpenAI API key, you can do so with:

# w.secrets.put_secret(scope=SECRETS_SCOPE, key=SECRETS_KEY, string_value='<key_value>')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a model serving endpoint for OpenAI GPT-4o-Mini
# MAGIC
# MAGIC The following creates a model serving endpoint for GPT-4o Mini *without* AI Gateway enabled. First, you define a helper function for creating and updating the endpoint:

# COMMAND ----------

import requests
import json
import time
from typing import Optional


def configure_endpoint(
    name: str,
    databricks_token: str,
    config: dict,
    host: str,
    endpoint_path: Optional[str] = None,
):
    base_url = f"{host}/api/2.0/serving-endpoints"

    if endpoint_path:
        # Update operation
        api_url = f"{base_url}/{name}/{endpoint_path}"
        method = requests.put
        operation = "Updating"
    else:
        # Create operation
        api_url = base_url
        method = requests.post
        operation = "Creating"

    headers = {
        "Authorization": f"Bearer {databricks_token}",
        "Content-Type": "application/json",
    }

    print(f"{operation} endpoint...")
    response = method(api_url, headers=headers, json=config)

    if response.status_code == 200:
        return response.json()
    else:
        print(
            f"Failed to {operation.lower()} endpoint. Status code: {response.status_code}"
        )
        return response.text

# COMMAND ----------

# MAGIC %md
# MAGIC Next, write a simple configuration to set up the endpoint. See [POST
# MAGIC /api/2.0/serving-endpoints](https://docs.databricks.com/api/workspace/servingendpoints/create) for API details.

# COMMAND ----------

create_endpoint_request_data = {
    "name": ENDPOINT_NAME,
    "config": {
        "served_entities": [
            {
                "name": "gpt-4o-mini",
                "external_model": {
                    "name": "gpt-4o-mini",
                    "provider": "openai",
                    "task": "llm/v1/chat",
                    "openai_config": {
                        "openai_api_key": f"{{{{secrets/{SECRETS_SCOPE}/{SECRETS_KEY}}}}}",
                    },
                },
            }
        ],
    },
}

# COMMAND ----------

import time

tmp_token = w.tokens.create(
    comment=f"sdk-{time.time_ns()}", lifetime_seconds=120
).token_value

configure_endpoint(
    ENDPOINT_NAME, tmp_token, create_endpoint_request_data, DATABRICKS_HOST
)

# COMMAND ----------

# MAGIC %md
# MAGIC One of the immediate benefits of using OpenAI models (or models from other providers) using Databricks is that you can immediately query the model using the any of the following methods:
# MAGIC  - Databricks Python SDK
# MAGIC  - OpenAI Python client
# MAGIC  - REST API calls
# MAGIC  -  MLflow Deployments SDK
# MAGIC  - Databricks SQL `ai_query` function 
# MAGIC
# MAGIC See the **Query foundation models and external models** article ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/score-foundation-models) |  [GCP](https://docs.databricks.com/gcp/en/machine-learning/model-serving/score-foundation-models)).
# MAGIC
# MAGIC For example, you can use `ai_query` to query the model with Databricks SQL.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   ai_query(
# MAGIC     "<endpoint name>",
# MAGIC     "What is a mixture of experts model?"
# MAGIC   )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add an AI Gateway configuration
# MAGIC
# MAGIC After you set up a model serving endpoint, you can query the OpenAI model using any of the various querying methods accessible in Databricks.
# MAGIC
# MAGIC You can further enrich the model serving endpoint by enabling the Databricks Mosaic AI Gateway, which offers a variety of features for monitoring and managing your endpoint. These features include inference tables, guardrails, and rate limits, among other things.
# MAGIC
# MAGIC To start, the following is a simple configuration that enables inference tables for monitoring endpoint usage. Understanding how the endpoint is being used and how often, helps to determine what usage limits and guardrails are beneficial for your use case.

# COMMAND ----------

gateway_request_data = {
    "usage_tracking_config": {"enabled": True},
    "inference_table_config": {
        "enabled": True,
        "catalog_name": CATALOG_NAME,
        "schema_name": SCHEMA_NAME,
    },
}

# COMMAND ----------

tmp_token = w.tokens.create(
    comment=f"sdk-{time.time_ns()}", lifetime_seconds=120
).token_value

configure_endpoint(
    ENDPOINT_NAME, tmp_token, gateway_request_data, DATABRICKS_HOST, "ai-gateway"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the inference table
# MAGIC
# MAGIC The following displays the inference table that was created when enabled in AI Gateway. Note: For example purposes, a number of queries were run on this endpoint in the AI playground after running the above update to add inference tables, but before querying them.
# MAGIC
# MAGIC

# COMMAND ----------

spark.sql(
    f"""select request_time, status_code, request, response
        from {CATALOG_NAME}.{SCHEMA_NAME}.`{ENDPOINT_NAME}_payload`
        where status_code=200
        limit 10"""
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC You can extract details such as the request messages, response messages, and token counts using SQL:

# COMMAND ----------

query = f"""SELECT
  request_time,
  from_json(
    request,
    'array<struct<messages:array<struct<role:string, content:string>>>>'
  ).messages [0].content AS request_messages,
  from_json(
    response,
    'struct<choices:array<struct<message:struct<role:string, content:string>>>>'
  ).choices [0].message.content AS response_messages,
  from_json(
    response,
    'struct<choices:array<struct<message:struct<role:string, content:string>>>, usage:struct<prompt_tokens:int, completion_tokens:int, total_tokens:int>>'
  ).usage.prompt_tokens AS prompt_tokens,
  from_json(
    response,
    'struct<choices:array<struct<message:struct<role:string, content:string>>>, usage:struct<prompt_tokens:int, completion_tokens:int, total_tokens:int>>'
  ).usage.completion_tokens AS completion_tokens,
  from_json(
    response,
    'struct<choices:array<struct<message:struct<role:string, content:string>>>, usage:struct<prompt_tokens:int, completion_tokens:int, total_tokens:int>>'
  ).usage.total_tokens AS total_tokens
FROM
  {CATALOG_NAME}.{SCHEMA_NAME}.`{ENDPOINT_NAME}_payload`
WHERE
  status_code = 200
LIMIT
  10;"""

spark.sql(query).display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Set up AI Guardrails
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Set up PII detection
# MAGIC
# MAGIC Now, the endpoint blocks messages referencing `SuperSecretProject`. You can also make sure the endpoint doesn't accept requests with or respond with messages containing any PII. 
# MAGIC
# MAGIC The following updates the guardrails configuration for `pii`:

# COMMAND ----------

gateway_request_data.update(
    {
        "guardrails": {
            "input": {
                "pii": {"behavior": "BLOCK"},

            },
            "output": {
                "pii": {"behavior": "BLOCK"},
            },
        }
    }
)

# COMMAND ----------

tmp_token = w.tokens.create(
    comment=f"sdk-{time.time_ns()}", lifetime_seconds=120
).token_value

configure_endpoint(
    ENDPOINT_NAME, tmp_token, gateway_request_data, DATABRICKS_HOST, "ai-gateway"
)

# COMMAND ----------

# MAGIC %md
# MAGIC The following tries to prompt the model to work with PII, but returns the message, `"Error: PII (Personally Identifiable Information) detected. Please try again."`.

# COMMAND ----------

fictional_data = """
Samantha Lee, slee@fictional-corp.com, (555) 123-4567, Senior Marketing Manager
Raj Patel, rpatel@imaginary-tech.net, (555) 987-6543, Software Engineer II
Elena Rodriguez, erodriguez@pretend-company.org, (555) 246-8135, Director of Operations
"""

prompt = f"""
You are an AI assistant for a company's HR department. Using the employee data provided below, answer the following question:

What is Raj Patel's phone number and email address?

Employee data:
{fictional_data}
"""


client = OpenAI(
    api_key=w.tokens.create(
        comment=f"sdk-{time.time_ns()}", lifetime_seconds=120
    ).token_value,
    base_url=f"{w.config.host}/serving-endpoints",
)

try:
    response = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
    )
    print(response)
except Exception as e:
    if "pii_detection': True" in str(e):
        print(
            "Error: PII (Personally Identifiable Information) detected. Please try again."
        )
    else:
        print(f"An error occurred: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Add rate limits
# MAGIC
# MAGIC Say you are investigating the inference tables further and you see some steep spikes in usage suggesting a higher-than-expected volume of queries. Extremely high usage could be costly if not monitored and limited.

# COMMAND ----------

query = f"""SELECT
  DATE_TRUNC('minute', request_time) AS minute,
  COUNT(DISTINCT databricks_request_id) AS queries_per_minute
FROM
  {CATALOG_NAME}.{SCHEMA_NAME}.`{ENDPOINT_NAME}_payload`
WHERE
  request_time >= CURRENT_TIMESTAMP - INTERVAL 20 HOURS
GROUP BY
  DATE_TRUNC('minute', request_time)
ORDER BY
  minute DESC;
"""

spark.sql(query).display()

# COMMAND ----------

# MAGIC %md
# MAGIC You can set a rate limit to prevent excessive queries. In this case, you can set the limit on the endpoint, but it is also possible to set per-user limits.

# COMMAND ----------

gateway_request_data.update(
    {
        "rate_limits": [{"calls": 10, "key": "endpoint", "renewal_period": "minute"}],
    }
)

# COMMAND ----------

tmp_token = w.tokens.create(
    comment=f"sdk-{time.time_ns()}", lifetime_seconds=120
).token_value

configure_endpoint(
    ENDPOINT_NAME, tmp_token, gateway_request_data, DATABRICKS_HOST, "ai-gateway"
)

# COMMAND ----------

# MAGIC %md
# MAGIC The following shows an example of what the output error looks like when the rate limit is exceeded.

# COMMAND ----------

client = OpenAI(
    api_key=w.tokens.create(
        comment=f"sdk-{time.time_ns()}", lifetime_seconds=120
    ).token_value,
    base_url=f"{w.config.host}/serving-endpoints",
)

start_time = time.time()
for i in range(1, 12):
    client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"This is request {i}"},
        ],
        max_tokens=10,
    )
    print(f"Request {i} sent")
print(f"Total time: {time.time() - start_time:.2f} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Add another model
# MAGIC
# MAGIC At some point, you might want to A/B test models from different providers. You can add another OpenAI model to the configuration, like in the following example:

# COMMAND ----------

new_config = {
    "served_entities": [
        {
            "name": "gpt-4o-mini",
            "external_model": {
                "name": "gpt-4o-mini",
                "provider": "openai",
                "task": "llm/v1/chat",
                "openai_config": {
                    "openai_api_key": f"{{{{secrets/{SECRETS_SCOPE}/{SECRETS_KEY}}}}}",
                },
            },
        },
        {
            "name": "gpt-4o",
            "external_model": {
                "name": "gpt-4o",
                "provider": "openai",
                "task": "llm/v1/chat",
                "openai_config": {
                    "openai_api_key": f"{{{{secrets/{SECRETS_SCOPE}/{SECRETS_KEY}}}}}",
                },
            },
        },
    ],
    "traffic_config": {
        "routes": [
            {"served_model_name": "gpt-4o-mini", "traffic_percentage": 50},
            {"served_model_name": "gpt-4o", "traffic_percentage": 50},
        ]
    },
}

# COMMAND ----------

tmp_token = w.tokens.create(
    comment=f"sdk-{time.time_ns()}", lifetime_seconds=120
).token_value

configure_endpoint(ENDPOINT_NAME, tmp_token, new_config, DATABRICKS_HOST, "config")

# COMMAND ----------

# MAGIC %md
# MAGIC Now, traffic will be split between these two models (you can configure the proportion of traffic going to each model). This enables you to use the inference tables to evaluate the quality of each model and make an informed decision about switching from one model to another.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable fallback models for requests
# MAGIC
# MAGIC For requests on External Models, you can configure a fallback. 
# MAGIC
# MAGIC Enabling fallbacks ensures that if a request to one entity fails with a 429 or 5XX error, it will automatically failover to the next entity in the listed order, cycling back to the top if necessary. There is a maximum of 2 fallbacks allowed. Any External Model assigned 0% traffic functions exclusively as a fallback model. The first successful or last failed request attempt is recorded in both the usage tracking system table and the inference table.
# MAGIC
# MAGIC In the following example: 
# MAGIC
# MAGIC - The `traffic_config` field specifies that 50 percent of traffic goes to `external_model_1` and the other 50% of the traffic goes to `external_model_2`. 
# MAGIC - In the `ai_gateway` section, the `fallback_config` field specifies that fallbacks are enabled. 
# MAGIC - If a request fails when it is sent to `external_model_1` then the request is redirected to the next model listed in the traffic configuration, in this case, `external_model_2`.

# COMMAND ----------

endpoint_config = {
   "name": endpoint_name,
   "config": {
       # Define your external models as entities
       "served_entities": [
         external_model_1,
         external_model_2
   ],
       "traffic_config": {
         "routes": [
           {
		# 50% traffic goes to first external model
             "served_model_name": “external_model_1”,
             "traffic_percentage": 50
           },
           {
		# 50% traffic goes to second external model (fallback only)
             "served_model_name": “external_model_2”,
             "traffic_percentage": 50
           }
         ]
       }
   },
# Enable fallbacks (occurs in the order of served entities)
   "ai_gateway": {
     "fallback_config": {"enabled": True}
   }
}

