# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluate and improve a GenAI application
# MAGIC
# MAGIC
# MAGIC This notebook contains code from the "Evaluate and improve your application" in the Databricks documentation. ([AWS](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/evaluate-app) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/evaluate-app) | [GCP](https://docs.databricks.com/gcp/en/mlflow3/genai/eval-monitor/evaluate-app)).

# COMMAND ----------

# MAGIC %md
# MAGIC Refer Link: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/evaluate-app

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[databricks]>=3.1.0" openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Simulated CRM database
CRM_DATA = {
    "Acme Corp": {
        "contact_name": "Alice Chen",
        "recent_meeting": "Product demo on Monday, very interested in enterprise features. They asked about: advanced analytics, real-time dashboards, API integrations, custom reporting, multi-user support, SSO authentication, data export capabilities, and pricing for 500+ users",
        "support_tickets": ["Ticket #123: API latency issue (resolved last week)", "Ticket #124: Feature request for bulk import", "Ticket #125: Question about GDPR compliance"],
        "account_manager": "Sarah Johnson"
    },
    "TechStart": {
        "contact_name": "Bob Martinez",
        "recent_meeting": "Initial sales call last Thursday, requested pricing",
        "support_tickets": ["Ticket #456: Login issues (open - critical)", "Ticket #457: Performance degradation reported", "Ticket #458: Integration failing with their CRM"],
        "account_manager": "Mike Thompson"
    },
    "Global Retail": {
        "contact_name": "Carol Wang",
        "recent_meeting": "Quarterly review yesterday, happy with platform performance",
        "support_tickets": [],
        "account_manager": "Sarah Johnson"
    }
}

# COMMAND ----------

import mlflow
from openai import OpenAI
from mlflow.entities import Document
from typing import List, Dict

# Enable automatic tracing for OpenAI calls
mlflow.openai.autolog()

# Connect to a Databricks LLM via OpenAI using the same credentials as MLflow
# Alternatively, you can use your own OpenAI credentials here
mlflow_creds = mlflow.utils.databricks_utils.get_databricks_host_creds()
client = OpenAI(
    api_key=mlflow_creds.token,
    base_url=f"{mlflow_creds.host}/serving-endpoints"
)

# Simulated CRM database
CRM_DATA = {
    "Acme Corp": {
        "contact_name": "Alice Chen",
        "recent_meeting": "Product demo on Monday, very interested in enterprise features. They asked about: advanced analytics, real-time dashboards, API integrations, custom reporting, multi-user support, SSO authentication, data export capabilities, and pricing for 500+ users",
        "support_tickets": ["Ticket #123: API latency issue (resolved last week)", "Ticket #124: Feature request for bulk import", "Ticket #125: Question about GDPR compliance"],
        "account_manager": "Sarah Johnson"
    },
    "TechStart": {
        "contact_name": "Bob Martinez",
        "recent_meeting": "Initial sales call last Thursday, requested pricing",
        "support_tickets": ["Ticket #456: Login issues (open - critical)", "Ticket #457: Performance degradation reported", "Ticket #458: Integration failing with their CRM"],
        "account_manager": "Mike Thompson"
    },
    "Global Retail": {
        "contact_name": "Carol Wang",
        "recent_meeting": "Quarterly review yesterday, happy with platform performance",
        "support_tickets": [],
        "account_manager": "Sarah Johnson"
    }
}

# Use a retriever span to enable MLflow's predefined RetrievalGroundedness scorer to work
@mlflow.trace(span_type="RETRIEVER")
def retrieve_customer_info(customer_name: str) -> List[Document]:
    """Retrieve customer information from CRM database"""
    if customer_name in CRM_DATA:
        data = CRM_DATA[customer_name]
        return [
            Document(
                id=f"{customer_name}_meeting",
                page_content=f"Recent meeting: {data['recent_meeting']}",
                metadata={"type": "meeting_notes"}
            ),
            Document(
                id=f"{customer_name}_tickets",
                page_content=f"Support tickets: {', '.join(data['support_tickets']) if data['support_tickets'] else 'No open tickets'}",
                metadata={"type": "support_status"}
            ),
            Document(
                id=f"{customer_name}_contact",
                page_content=f"Contact: {data['contact_name']}, Account Manager: {data['account_manager']}",
                metadata={"type": "contact_info"}
            )
        ]
    return []

@mlflow.trace
def generate_sales_email(customer_name: str, user_instructions: str) -> Dict[str, str]:
    """Generate personalized sales email based on customer data & a sale's rep's instructions."""
    # Retrieve customer information
    customer_docs = retrieve_customer_info(customer_name)

    # Combine retrieved context
    context = "\n".join([doc.page_content for doc in customer_docs])

    # Generate email using retrieved context
    prompt = f"""You are a sales representative. Based on the customer information below,
    write a brief follow-up email that addresses their request.

    Customer Information:
    {context}

    User instructions: {user_instructions}

    Keep the email concise and personalized."""

    response = client.chat.completions.create(
        model="databricks-claude-sonnet-4-5", # This example uses a Databricks hosted LLM - you can replace this with any AI Gateway or Model Serving endpoint. If you provide your own OpenAI credentials, replace with a valid OpenAI model e.g., gpt-4o, etc.
        messages=[
            {"role": "system", "content": "You are a helpful sales assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )

    return {"email": response.choices[0].message.content}

# Test the application
result = generate_sales_email("Acme Corp", "Follow up after product demo")
print(result["email"])

# COMMAND ----------

# Simulate beta testing traffic with scenarios designed to fail guidelines
test_requests = [
    {"customer_name": "Acme Corp", "user_instructions": "Follow up after product demo"},
    {"customer_name": "TechStart", "user_instructions": "Check on support ticket status"},
    {"customer_name": "Global Retail", "user_instructions": "Send quarterly review summary"},
    {"customer_name": "Acme Corp", "user_instructions": "Write a very detailed email explaining all our product features, pricing tiers, implementation timeline, and support options"},
    {"customer_name": "TechStart", "user_instructions": "Send an enthusiastic thank you for their business!"},
    {"customer_name": "Global Retail", "user_instructions": "Send a follow-up email"},
    {"customer_name": "Acme Corp", "user_instructions": "Just check in to see how things are going"},
]

# Run requests and capture traces
print("Simulating production traffic...")
for req in test_requests:
    try:
        result = generate_sales_email(**req)
        print(f"✓ Generated email for {req['customer_name']}")
    except Exception as e:
        print(f"✗ Error for {req['customer_name']}: {e}")

# COMMAND ----------

import mlflow.genai.datasets
import time
from databricks.connect import DatabricksSession

# 1. Create an evaluation dataset

# Replace with a Unity Catalog schema where you have CREATE TABLE permission
uc_schema = "docs.default"
# This table will be created in the above UC schema
evaluation_dataset_table_name = "email_generation_eval"
full_table_name = f"{uc_schema}.{evaluation_dataset_table_name}"

if not spark.catalog.tableExists(full_table_name):
    eval_dataset = mlflow.genai.datasets.create_dataset(name=full_table_name)
    print(f"Created evaluation dataset: {full_table_name}")
else:
    eval_dataset = mlflow.genai.datasets.get_dataset(name=full_table_name)
    print(f"Retrieved existing evaluation dataset: {full_table_name}")

# 2. Search for the simulated production traces from step 2: get traces from the last 20 minutes with our trace name.
ten_minutes_ago = int((time.time() - 10 * 60) * 1000)

traces = mlflow.search_traces(
    filter_string=f"attributes.timestamp_ms > {ten_minutes_ago} AND "
                 f"attributes.status = 'OK' AND "
                 f"tags.`mlflow.traceName` = 'generate_sales_email'",
    order_by=["attributes.timestamp_ms DESC"]
)

print(f"Found {len(traces)} successful traces from beta test")

# 3. Add the traces to the evaluation dataset
eval_dataset = eval_dataset.merge_records(traces)
print(f"Added {len(traces)} records to evaluation dataset")

# Preview the dataset
eval_dataset_df = eval_dataset.to_df()
print(f"\nDataset preview:")
print(f"Total records: {len(eval_dataset_df)}")
print("\nSample record:")
sample = eval_dataset_df.iloc[0]
print(f"Inputs: {sample['inputs']}")

# COMMAND ----------

from mlflow.genai.scorers import (
    RetrievalGroundedness,
    RelevanceToQuery,
    Safety,
    Guidelines,
)

# Save the scorers as a variable so we can re-use them in step 7

email_scorers = [
        RetrievalGroundedness(),  # Checks if email content is grounded in retrieved data
        Guidelines(
            name="follows_instructions",
            guidelines="The generated email must follow the user_instructions in the request.",
        ),
        Guidelines(
            name="concise_communication",
            guidelines="The email MUST be concise and to the point. The email should communicate the key message efficiently without being overly brief or losing important context.",
        ),
        Guidelines(
            name="mentions_contact_name",
            guidelines="The email MUST explicitly mention the customer contact's first name (e.g., Alice, Bob, Carol) in the greeting. Generic greetings like 'Hello' or 'Dear Customer' are not acceptable.",
        ),
        Guidelines(
            name="professional_tone",
            guidelines="The email must be in a professional tone.",
        ),
        Guidelines(
            name="includes_next_steps",
            guidelines="The email MUST end with a specific, actionable next step that includes a concrete timeline.",
        ),
        RelevanceToQuery(),  # Checks if email addresses the user's request
        Safety(),  # Checks for harmful or inappropriate content
    ]

# Run evaluation with predefined scorers
eval_results_v1 = mlflow.genai.evaluate(
    data=eval_dataset_df,
    predict_fn=generate_sales_email,
    scorers=email_scorers,
)

# COMMAND ----------

eval_traces = mlflow.search_traces(run_id=eval_results_v1.run_id)

# eval_traces is a Pandas DataFrame that has the evaluated traces.  The column `assessments` includes each scorer's feedback.
eval_traces

# COMMAND ----------

display(eval_traces)

# COMMAND ----------

@mlflow.trace
def generate_sales_email_v2(customer_name: str, user_instructions: str) -> Dict[str, str]:
    """Generate personalized sales email based on customer data & a sale's rep's instructions."""
    # Retrieve customer information
    customer_docs = retrieve_customer_info(customer_name)

    if not customer_docs:
        return {"error": f"No customer data found for {customer_name}"}

    # Combine retrieved context
    context = "\n".join([doc.page_content for doc in customer_docs])

    # Generate email using retrieved context with better instruction following
    prompt = f"""You are a sales representative writing an email.

MOST IMPORTANT: Follow these specific user instructions exactly:
{user_instructions}

Customer context (only use what's relevant to the instructions):
{context}

Guidelines:
1. PRIORITIZE the user instructions above all else
2. Keep the email CONCISE - only include information directly relevant to the user's request
3. End with a specific, actionable next step that includes a concrete timeline (e.g., "I'll follow up with pricing by Friday" or "Let's schedule a 15-minute call this week")
4. Only reference customer information if it's directly relevant to the user's instructions

Write a brief, focused email that satisfies the user's exact request."""

    response = client.chat.completions.create(
        model="databricks-claude-sonnet-4-5",
        messages=[
            {"role": "system", "content": "You are a helpful sales assistant who writes concise, instruction-focused emails."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )

    return {"email": response.choices[0].message.content}

# Test the application
result = generate_sales_email("Acme Corp", "Follow up after product demo")
print(result["email"])

# COMMAND ----------

# Run evaluation of the new version with the same scorers as before
# We use start_run to name the evaluation run in the UI
with mlflow.start_run(run_name="v2"):
    eval_results_v2 = mlflow.genai.evaluate(
        data=eval_dataset_df, # same eval dataset
        predict_fn=generate_sales_email_v2, # new app version
        scorers=email_scorers, # same scorers as step 4
    )

# COMMAND ----------

import pandas as pd

# Fetch runs separately since mlflow.search_runs doesn't support IN or OR operators
run_v1_df = mlflow.search_runs(
    filter_string=f"run_id = '{eval_results_v1.run_id}'"
)
run_v2_df = mlflow.search_runs(
    filter_string=f"run_id = '{eval_results_v2.run_id}'"
)

# Extract metric columns (they end with /mean, not .aggregate_score)
# Skip the agent metrics (latency, token counts) for quality comparison
metric_cols = [col for col in run_v1_df.columns
               if col.startswith('metrics.') and col.endswith('/mean')
               and 'agent/' not in col]

# Create comparison table
comparison_data = []
for metric in metric_cols:
    metric_name = metric.replace('metrics.', '').replace('/mean', '')
    v1_score = run_v1_df[metric].iloc[0]
    v2_score = run_v2_df[metric].iloc[0]
    improvement = v2_score - v1_score

    comparison_data.append({
        'Metric': metric_name,
        'V1 Score': f"{v1_score:.3f}",
        'V2 Score': f"{v2_score:.3f}",
        'Improvement': f"{improvement:+.3f}",
        'Improved': '✓' if improvement >= 0 else '✗'
    })

comparison_df = pd.DataFrame(comparison_data)
display(comparison_df)

avg_v1 = run_v1_df[metric_cols].mean(axis=1).iloc[0]
avg_v2 = run_v2_df[metric_cols].mean(axis=1).iloc[0]
display(f"Overall average improvement: {(avg_v2 - avg_v1):+.3f} ({((avg_v2/avg_v1 - 1) * 100):+.1f}%)")

# COMMAND ----------

# Get detailed traces for both versions
traces_v1 = mlflow.search_traces(run_id=eval_results_v1.run_id)
traces_v2 = mlflow.search_traces(run_id=eval_results_v2.run_id)

# Create a merge key based on the input parameters
traces_v1['merge_key'] = traces_v1['request'].apply(
    lambda x: f"{x.get('customer_name', '')}|{x.get('user_instructions', '')}"
)
traces_v2['merge_key'] = traces_v2['request'].apply(
    lambda x: f"{x.get('customer_name', '')}|{x.get('user_instructions', '')}"
)

# Merge on the input data to compare same inputs
merged = traces_v1.merge(
    traces_v2,
    on='merge_key',
    suffixes=('_v1', '_v2')
)

display(f"Found {len(merged)} matching examples between v1 and v2")

# Find examples where specific metrics did NOT improve
regression_examples = []

for idx, row in merged.iterrows():
    v1_assessments = {a['assessment_name']: a for a in row['assessments_v1']}
    v2_assessments = {a['assessment_name']: a for a in row['assessments_v2']}

    # Check each scorer for regressions
    for scorer_name in ['follows_instructions', 'concise_communication', 'includes_next_steps', 'retrieval_groundedness']:
        v1_assessment = v1_assessments.get(scorer_name)
        v2_assessment = v2_assessments.get(scorer_name)

        if v1_assessment and v2_assessment:
            v1_val = v1_assessment['feedback']['value']
            v2_val = v2_assessment['feedback']['value']

            # Check if metric got worse (yes -> no)
            if v1_val == 'yes' and v2_val == 'no':
                regression_examples.append({
                    'index': idx,
                    'customer': row['request_v1']['customer_name'],
                    'instructions': row['request_v1']['user_instructions'],
                    'metric': scorer_name,
                    'v1_score': v1_val,
                    'v2_score': v2_val,
                    'v1_rationale': v1_assessment['rationale'],
                    'v2_rationale': v2_assessment['rationale'],
                    'v1_response': row['response_v1']['email'],
                    'v2_response': row['response_v2']['email']
                })

# Display regression examples
if regression_examples:
    display(f"\n=== Found {len(regression_examples)} metric regressions ===\n")

    # Group by metric
    by_metric = {}
    for ex in regression_examples:
        metric = ex['metric']
        if metric not in by_metric:
            by_metric[metric] = []
        by_metric[metric].append(ex)

    # Show examples for each regressed metric
    for metric, examples in by_metric.items():
        display(f"\n{'='*80}")
        display(f"METRIC REGRESSION: {metric}")
        display(f"{'='*80}")

        # Show the first example for this metric
        ex = examples[0]
        display(f"\nCustomer: {ex['customer']}")
        display(f"Instructions: {ex['instructions']}")
        display(f"\nV1 Score: ✓ (passed)")
        display(f"V1 Rationale: {ex['v1_rationale']}")
        display(f"\nV2 Score: ✗ (failed)")
        display(f"V2 Rationale: {ex['v2_rationale']}")

        display(f"\n--- V1 Response ---")
        display(ex['v1_response'][:800] + "..." if len(ex['v1_response']) > 800 else ex['v1_response'])

        display(f"\n--- V2 Response ---")
        display(ex['v2_response'][:800] + "..." if len(ex['v2_response']) > 800 else ex['v2_response'])

        if len(examples) > 1:
            display(f"\n(+{len(examples)-1} more examples with {metric} regression)")
else:
    display("No metric regressions found - V2 improved or maintained all metrics")

# COMMAND ----------

# DBTITLE 1,ro

