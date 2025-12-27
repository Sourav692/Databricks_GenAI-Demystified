# Databricks notebook source
# MAGIC %md
# MAGIC # Module 3: Running Evaluations with Built-in Judges
# MAGIC 
# MAGIC **Objective:** Evaluate agents using MLflow's built-in LLM judges
# MAGIC 
# MAGIC **What you'll learn:**
# MAGIC - Use built-in judges (Safety, Faithfulness, Answer Correctness)
# MAGIC - Run evaluations on datasets
# MAGIC - Interpret evaluation results
# MAGIC - Access aggregate metrics and individual scores
# MAGIC - Identify failure patterns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites Check

# COMMAND ----------

import mlflow
from packaging import version

assert version.parse(mlflow.__version__) >= version.parse("3.1.0"), \
    "Please run Notebook 01 first"

print(f"✅ MLflow {mlflow.__version__} ready")

# Set experiment
experiment_name = "/Users/sourav.banerjee@databricks.com/agent-evaluation-demo"
mlflow.set_experiment(experiment_name)
print(f"✅ Experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create a Simple Agent for Evaluation

# COMMAND ----------

import mlflow

# Enable autologging
mlflow.openai.autolog()

@mlflow.trace(span_type="RETRIEVER")
def retrieve_documents(query: str):
    """
    Simulated retrieval function
    Returns relevant documents based on query
    """
    # Simulated knowledge base
    knowledge_base = {
        "return": [
            {
                "content": "Returns accepted within 30 days. Original packaging required.",
                "doc_uri": "policies/returns.pdf"
            },
            {
                "content": "Refunds processed in 5-7 business days after receipt.",
                "doc_uri": "policies/refunds.pdf"
            }
        ],
        "shipping": [
            {
                "content": "Standard shipping: 3-5 days ($5.99). Express: 1-2 days ($14.99).",
                "doc_uri": "policies/shipping.pdf"
            },
            {
                "content": "Free shipping on orders over $50.",
                "doc_uri": "policies/shipping.pdf"
            }
        ],
        "warranty": [
            {
                "content": "1-year warranty covers manufacturing defects.",
                "doc_uri": "policies/warranty.pdf"
            }
        ]
    }
    
    # Simple keyword matching
    query_lower = query.lower()
    if "return" in query_lower:
        return knowledge_base["return"]
    elif "ship" in query_lower:
        return knowledge_base["shipping"]
    elif "warrant" in query_lower:
        return knowledge_base["warranty"]
    else:
        return knowledge_base["return"]  # Default

@mlflow.trace
def simple_agent(question: str):
    """
    Simple rule-based agent for demonstration
    In production, this would call an actual LLM
    """
    # Retrieve relevant documents
    documents = retrieve_documents(question)
    
    # Format context
    context = "\n".join([doc["content"] for doc in documents])
    
    # Generate response (simulated - in production use actual LLM)
    response = f"Based on our policies: {context}"
    
    return response

# Test the agent
test_question = "What is your return policy?"
test_answer = simple_agent(test_question)

print(f"Question: {test_question}")
print(f"Answer: {test_answer}")

trace_id = mlflow.get_last_active_trace_id()
print(f"\n✓ Trace ID: {trace_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load the Evaluation Dataset

# COMMAND ----------

# Load the dataset created in Notebook 02
CATALOG = "main"
SCHEMA = "default"
DATASET_NAME = "agent_eval_dataset"

dataset_full_name = f"{CATALOG}.{SCHEMA}.{DATASET_NAME}"

eval_dataset = mlflow.genai.datasets.load_dataset(
    name=dataset_full_name
)

print(f"✅ Loaded dataset: {dataset_full_name}")
print(f"   Version: {eval_dataset.version}")
print(f"   Records: {len(eval_dataset)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Set Up Built-in Judges

# COMMAND ----------

from mlflow.metrics.genai import (
    answer_correctness,
    faithfulness,
    safety,
    answer_relevance
)

# Create built-in judges
judges = [
    # Compares response against expected_facts and expected_response
    answer_correctness(),
    
    # Checks if response is grounded in retrieved context
    # REQUIRES span_type="RETRIEVER" in traces
    faithfulness(),
    
    # Checks for harmful or inappropriate content
    safety(),
    
    # Checks if response addresses the question
    answer_relevance()
]

print("Built-in judges configured:")
for judge in judges:
    print(f"  ✓ {judge.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Run Evaluation

# COMMAND ----------

# Run evaluation
# Note: This evaluates the agent function we created
print("Starting evaluation...")
print("This may take a few minutes depending on dataset size")

results = mlflow.genai.evaluate(
    model=simple_agent,  # Our agent function
    data=eval_dataset,   # Dataset to evaluate on
    model_type="databricks-agent",
    evaluators=judges
)

print("\n✅ Evaluation complete!")
print(f"   Run ID: {results.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: View Aggregate Metrics

# COMMAND ----------

print("="*60)
print("AGGREGATE METRICS")
print("="*60)

# Display all metrics
for metric_name, value in sorted(results.metrics.items()):
    if 'mean' in metric_name or 'variance' in metric_name:
        print(f"{metric_name}: {value:.3f}")

# Calculate overall quality score
key_metrics = [
    'answer_correctness/score/mean',
    'faithfulness/score/mean',
    'answer_relevance/score/mean'
]

scores = [results.metrics.get(m, 0) for m in key_metrics]
overall_score = sum(scores) / len(scores) if scores else 0

print(f"\n{'='*60}")
print(f"Overall Quality Score: {overall_score:.3f}")
print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Examine Individual Results

# COMMAND ----------

# Get the detailed results table
eval_table = results.tables["eval_results"]

print(f"Total test cases: {len(eval_table)}")
print(f"\nColumns in results:")
for col in eval_table.columns:
    print(f"  - {col}")

# Display first few results
print("\nSample Results:")
display(eval_table.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Identify Failures

# COMMAND ----------

# Find cases where answer_correctness is low
threshold = 0.5

failures = eval_table[eval_table["answer_correctness/score"] < threshold]

print(f"{'='*60}")
print(f"FAILURE ANALYSIS")
print(f"{'='*60}")
print(f"\nThreshold: {threshold}")
print(f"Failed cases: {len(failures)} out of {len(eval_table)}")
print(f"Failure rate: {len(failures)/len(eval_table)*100:.1f}%")

# Show failed cases
if len(failures) > 0:
    print(f"\nFailed Test Cases:")
    for idx, row in failures.iterrows():
        print(f"\n{'-'*60}")
        print(f"Case {idx + 1}:")
        print(f"  Question: {row['inputs']['question']}")
        print(f"  Agent Answer: {row['outputs'][:200]}...")
        print(f"  Score: {row['answer_correctness/score']:.2f}")
        print(f"  Why it failed:")
        print(f"    {row['answer_correctness/rationale'][:300]}...")
else:
    print("\n✅ No failures! All cases passed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Analyze by Category

# COMMAND ----------

import pandas as pd

# Load dataset to get metadata
spark_df = spark.table(dataset_full_name)
metadata_df = spark_df.select("inputs", "metadata").toPandas()

# Merge with results
results_with_meta = eval_table.merge(
    metadata_df,
    on='inputs',
    how='left'
)

# Extract category
results_with_meta['category'] = results_with_meta['metadata'].apply(
    lambda x: x.get('category', 'unknown') if isinstance(x, dict) else 'unknown'
)

# Calculate performance by category
category_performance = results_with_meta.groupby('category').agg({
    'answer_correctness/score': 'mean',
    'faithfulness/score': 'mean',
    'answer_relevance/score': 'mean'
}).round(3)

print("="*60)
print("PERFORMANCE BY CATEGORY")
print("="*60)
print(category_performance)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: View Results in MLflow UI

# COMMAND ----------

# Get URLs for viewing results
experiment = mlflow.get_experiment_by_name(experiment_name)

print("="*60)
print("VIEW RESULTS IN MLFLOW UI")
print("="*60)

print(f"\nExperiment URL:")
print(f"https://<your-databricks-workspace>/ml/experiments/{experiment.experiment_id}")

print(f"\nEvaluation Run URL:")
print(f"https://<your-databricks-workspace>/ml/experiments/{experiment.experiment_id}/runs/{results.run_id}")

print(f"\nWhat you can see in the UI:")
print("  ✓ Aggregate metrics dashboard")
print("  ✓ Individual test case results")
print("  ✓ Judge rationales and explanations")
print("  ✓ Traces for each evaluation")
print("  ✓ Comparison with other runs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Generate Evaluation Report

# COMMAND ----------

def generate_report(results, dataset_name):
    """Generate a text evaluation report"""
    
    eval_table = results.tables["eval_results"]
    
    report = f"""
{'='*60}
EVALUATION REPORT
{'='*60}

Dataset: {dataset_name}
Run ID: {results.run_id}
Test Cases: {len(eval_table)}

{'='*60}
OVERALL PERFORMANCE
{'='*60}

Answer Correctness: {results.metrics.get('answer_correctness/score/mean', 0):.3f}
Faithfulness:       {results.metrics.get('faithfulness/score/mean', 0):.3f}
Safety:             {results.metrics.get('safety/score/mean', 0):.3f}
Answer Relevance:   {results.metrics.get('answer_relevance/score/mean', 0):.3f}

{'='*60}
FAILURE ANALYSIS
{'='*60}

"""
    
    # Count failures
    failures = eval_table[eval_table["answer_correctness/score"] < 0.5]
    
    report += f"Failed cases: {len(failures)} / {len(eval_table)}\n"
    report += f"Failure rate: {len(failures)/len(eval_table)*100:.1f}%\n"
    
    report += f"\n{'='*60}\n"
    report += "RECOMMENDATIONS\n"
    report += f"{'='*60}\n\n"
    
    # Generate recommendations based on scores
    if results.metrics.get('answer_correctness/score/mean', 0) < 0.7:
        report += "⚠️  Low answer correctness - Review knowledge base coverage\n"
    
    if results.metrics.get('faithfulness/score/mean', 0) < 0.8:
        report += "⚠️  Hallucination concerns - Improve retrieval quality\n"
    
    if results.metrics.get('answer_relevance/score/mean', 0) < 0.7:
        report += "⚠️  Relevance issues - Check query understanding\n"
    
    if len(failures) == 0:
        report += "✅ No critical issues detected\n"
    
    report += f"\n{'='*60}\n"
    
    return report

# Generate and print report
report = generate_report(results, dataset_full_name)
print(report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC **What we accomplished:**
# MAGIC 1. ✅ Created a simple agent with proper tracing
# MAGIC 2. ✅ Configured built-in judges (correctness, faithfulness, safety)
# MAGIC 3. ✅ Ran evaluation on dataset
# MAGIC 4. ✅ Analyzed aggregate metrics
# MAGIC 5. ✅ Identified individual failures
# MAGIC 6. ✅ Analyzed performance by category
# MAGIC 7. ✅ Generated evaluation report
# MAGIC 
# MAGIC **Next Steps:**
# MAGIC - Proceed to Notebook 4: Custom Judges and Code-based Scorers
# MAGIC - Learn how to create custom evaluation logic
