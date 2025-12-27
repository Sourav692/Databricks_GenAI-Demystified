# Databricks notebook source
# MAGIC %md
# MAGIC # Module 5: Production Monitoring and Operations
# MAGIC 
# MAGIC **Objective:** Set up continuous quality monitoring for production agents
# MAGIC 
# MAGIC **What you'll learn:**
# MAGIC - Register scorers for monitoring
# MAGIC - Configure sampling strategies
# MAGIC - Start and stop monitoring
# MAGIC - Query monitoring results
# MAGIC - Generate monitoring reports

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites Check

# COMMAND ----------

import mlflow
from packaging import version

assert version.parse(mlflow.__version__) >= version.parse("3.1.0"), \
    "Please run Notebook 01 first"

print(f"✅ MLflow {mlflow.__version__} ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Set Up Production Experiment

# COMMAND ----------

# Create a separate experiment for production monitoring
prod_experiment_name = "/Users/sourav.banerjee@databricks.com/agent-production-monitoring"
mlflow.set_experiment(prod_experiment_name)

print(f"✅ Production experiment: {prod_experiment_name}")

experiment = mlflow.get_experiment_by_name(prod_experiment_name)
print(f"   Experiment ID: {experiment.experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Define Scorers for Monitoring

# COMMAND ----------

from mlflow.metrics.genai import (
    safety,
    faithfulness,
    answer_relevance
)

# Define which scorers to monitor
monitoring_scorers = [
    safety(),              # Critical: Check every interaction
    faithfulness(),        # High priority: Check for hallucinations
    answer_relevance()     # Medium priority: Check relevance
]

print("Scorers configured for monitoring:")
for scorer in monitoring_scorers:
    print(f"  ✓ {scorer.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Register Scorers

# COMMAND ----------

from mlflow.genai.monitoring import register_scorer

registered_scorers = []

for scorer in monitoring_scorers:
    try:
        registered = register_scorer(
            scorer=scorer,
            name=f"{scorer.name}_monitor"  # Add _monitor suffix
        )
        registered_scorers.append(registered)
        print(f"✅ Registered: {registered.name}")
    except Exception as e:
        print(f"⚠️  Error registering {scorer.name}: {e}")

print(f"\nTotal registered: {len(registered_scorers)} scorers")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Configure Sampling Strategies

# COMMAND ----------

from mlflow.genai.monitoring import ScorerSamplingConfig

# Different sampling rates for different priorities
sampling_configs = {
    "safety_monitor": ScorerSamplingConfig(
        sample_rate=1.0  # 100% - check every interaction
    ),
    "faithfulness_monitor": ScorerSamplingConfig(
        sample_rate=0.2  # 20% - sample for hallucinations
    ),
    "answer_relevance_monitor": ScorerSamplingConfig(
        sample_rate=0.1  # 10% - sample for quality
    )
}

print("Sampling configuration:")
for name, config in sampling_configs.items():
    print(f"  {name}: {config.sample_rate * 100}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Start Monitoring

# COMMAND ----------

active_monitors = []

for registered_scorer in registered_scorers:
    scorer_name = registered_scorer.name
    
    # Get appropriate sampling config
    sampling_config = sampling_configs.get(
        scorer_name,
        ScorerSamplingConfig(sample_rate=0.1)  # Default 10%
    )
    
    try:
        # Start monitoring
        active = registered_scorer.start(
            sampling_config=sampling_config
        )
        active_monitors.append(active)
        print(f"✅ Started: {scorer_name} at {sampling_config.sample_rate*100}%")
    except Exception as e:
        print(f"⚠️  Error starting {scorer_name}: {e}")

print(f"\n{'='*60}")
print("MONITORING STATUS")
print(f"{'='*60}")
print(f"Active monitors: {len(active_monitors)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Verify Monitoring Status

# COMMAND ----------

from mlflow.genai.monitoring import list_active_scorers

# List all active scorers
try:
    active_scorers = list_active_scorers(
        experiment_name=prod_experiment_name
    )
    
    print(f"{'='*60}")
    print("ACTIVE MONITORING SCORERS")
    print(f"{'='*60}")
    
    for scorer in active_scorers:
        print(f"\nScorer: {scorer.name}")
        print(f"  Status: {scorer.status}")
        print(f"  Sample Rate: {scorer.sampling_config.sample_rate * 100}%")
    
except Exception as e:
    print(f"Note: {e}")
    print("Monitoring may need to be activated via Databricks UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Simulate Production Traffic

# COMMAND ----------

import mlflow
import time

mlflow.openai.autolog()

@mlflow.trace(span_type="RETRIEVER")
def prod_retrieve(query):
    """Production retrieval function"""
    return [
        {"content": "Standard return policy content", "doc_uri": "returns.pdf"}
    ]

@mlflow.trace
def production_agent(question: str):
    """Simulated production agent"""
    docs = prod_retrieve(question)
    response = f"Response based on policy: {docs[0]['content']}"
    return response

# Simulate 10 production interactions
print("Simulating production traffic...")
for i in range(10):
    question = f"Test question {i+1}"
    answer = production_agent(question)
    trace_id = mlflow.get_last_active_trace_id()
    print(f"  Interaction {i+1}: Trace {trace_id[:8]}...")
    time.sleep(0.5)  # Brief pause between interactions

print("\n✅ Simulated 10 production interactions")
print("   Monitoring scorers should evaluate sampled traces")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Query Monitoring Results

# COMMAND ----------

from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name(prod_experiment_name)

# Search for recent traces
try:
    traces = client.search_traces(
        experiment_ids=[experiment.experiment_id],
        max_results=100
    )
    
    print(f"Found {len(traces)} traces in production experiment")
    
    # Extract scores
    scores_data = []
    for trace in traces:
        if hasattr(trace.data, 'feedback') and trace.data.feedback:
            trace_info = {
                'trace_id': trace.info.request_id[:8],
                'timestamp': trace.info.timestamp_ms
            }
            
            for scorer_name, feedback in trace.data.feedback.items():
                trace_info[f'{scorer_name}_score'] = feedback.get('value')
            
            scores_data.append(trace_info)
    
    if scores_data:
        scores_df = pd.DataFrame(scores_data)
        print(f"\nScored traces: {len(scores_df)}")
        display(scores_df.head(10))
    else:
        print("\nNo scored traces yet. Monitoring may take a few minutes to start.")
        
except Exception as e:
    print(f"Note: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Generate Monitoring Report

# COMMAND ----------

def generate_monitoring_report(experiment_name):
    """
    Generate a monitoring report for recent activity
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    # Get traces
    traces = client.search_traces(
        experiment_ids=[experiment.experiment_id],
        max_results=1000
    )
    
    report = f"""
{'='*60}
PRODUCTION MONITORING REPORT
{'='*60}

Experiment: {experiment_name}
Report Generated: {pd.Timestamp.now()}

{'='*60}
VOLUME METRICS
{'='*60}

Total Interactions: {len(traces)}
"""
    
    # Count scored traces
    scored_traces = [
        t for t in traces 
        if hasattr(t.data, 'feedback') and t.data.feedback
    ]
    
    report += f"Scored Interactions: {len(scored_traces)}\n"
    report += f"Scoring Rate: {len(scored_traces)/len(traces)*100:.1f}%\n" if traces else "N/A\n"
    
    # Calculate scores if available
    if scored_traces:
        report += f"\n{'='*60}\n"
        report += "QUALITY METRICS\n"
        report += f"{'='*60}\n\n"
        
        # Aggregate by scorer
        scorer_stats = {}
        for trace in scored_traces:
            for scorer_name, feedback in trace.data.feedback.items():
                if scorer_name not in scorer_stats:
                    scorer_stats[scorer_name] = []
                scorer_stats[scorer_name].append(feedback.get('value'))
        
        for scorer_name, scores in scorer_stats.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            report += f"{scorer_name}:\n"
            report += f"  Evaluated: {len(scores)} traces\n"
            report += f"  Average: {avg_score:.3f}\n\n"
    
    report += f"{'='*60}\n"
    
    return report

# Generate report
report = generate_monitoring_report(prod_experiment_name)
print(report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Monitoring Best Practices

# COMMAND ----------

print("="*60)
print("PRODUCTION MONITORING BEST PRACTICES")
print("="*60)

print("""
1. SAMPLING STRATEGY
   ✓ Critical scorers (safety): 100%
   ✓ Important scorers (faithfulness): 20%
   ✓ Quality scorers (relevance): 10%
   ✓ Cost optimization: Lower rates for expensive judges

2. ALERT THRESHOLDS
   ✓ Safety violations: Immediate alert
   ✓ Faithfulness < 0.8: Daily review
   ✓ Relevance < 0.7: Weekly review

3. MONITORING LIMITS
   ✓ Maximum 20 active scorers per experiment
   ✓ Balance coverage with cost

4. REGULAR REVIEWS
   ✓ Daily: Check alerts
   ✓ Weekly: Trend analysis
   ✓ Monthly: Deep dive and optimization

5. COST MANAGEMENT
   ✓ Monitor evaluation costs
   ✓ Adjust sampling rates as needed
   ✓ Use code-based scorers for cheap checks
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Update or Stop Monitoring (Optional)

# COMMAND ----------

# Example: Update sampling rate
# Uncomment to use

# from mlflow.genai.monitoring import get_scorer

# # Get a specific scorer
# scorer = get_scorer(
#     experiment_name=prod_experiment_name,
#     name="faithfulness_monitor"
# )

# # Update sampling rate
# new_sampling = ScorerSamplingConfig(sample_rate=0.5)  # Increase to 50%
# updated_scorer = scorer.update(sampling_config=new_sampling)

# print(f"✅ Updated sampling rate to {new_sampling.sample_rate * 100}%")

# # To stop monitoring
# # stopped_scorer = scorer.stop()
# # print("✅ Monitoring stopped")

print("Note: Uncomment code above to update or stop monitoring")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC **What we accomplished:**
# MAGIC 1. ✅ Created production monitoring experiment
# MAGIC 2. ✅ Registered scorers for continuous monitoring
# MAGIC 3. ✅ Configured tiered sampling strategies
# MAGIC 4. ✅ Started monitoring with different sample rates
# MAGIC 5. ✅ Simulated production traffic
# MAGIC 6. ✅ Queried monitoring results
# MAGIC 7. ✅ Generated monitoring reports
# MAGIC 
# MAGIC **Key Concepts:**
# MAGIC - Use higher sampling (100%) for critical scorers
# MAGIC - Use lower sampling (5-20%) for cost optimization
# MAGIC - Monitor the monitoring system itself
# MAGIC - Generate regular quality reports
# MAGIC 
# MAGIC **Production Checklist:**
# MAGIC - [ ] Separate monitoring experiment created
# MAGIC - [ ] Critical scorers at appropriate sampling
# MAGIC - [ ] Alert thresholds defined
# MAGIC - [ ] Regular review schedule established
# MAGIC - [ ] Cost monitoring in place
