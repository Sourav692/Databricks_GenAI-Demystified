# Databricks notebook source
# MAGIC %md
# MAGIC # Module 4: Custom Judges and Code-Based Scorers
# MAGIC 
# MAGIC **Objective:** Create custom evaluation logic for domain-specific requirements
# MAGIC 
# MAGIC **What you'll learn:**
# MAGIC - Create custom LLM judges with make_judge()
# MAGIC - Use template variables (inputs, outputs, context, trace)
# MAGIC - Create code-based scorers with @scorer decorator
# MAGIC - Combine custom and built-in evaluators

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites Check

# COMMAND ----------

import mlflow
from packaging import version

assert version.parse(mlflow.__version__) >= version.parse("3.1.0"), \
    "Please run Notebook 01 first"

print(f"✅ MLflow {mlflow.__version__} ready")

experiment_name = "/Users/sourav.banerjee@databricks.com/agent-evaluation-demo"
mlflow.set_experiment(experiment_name)
print(f"✅ Experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create a Custom LLM Judge

# COMMAND ----------

from mlflow.metrics.genai import make_judge

# Create a custom judge for tone and professionalism
tone_judge = make_judge(
    name="customer_tone_quality",
    criteria="""
    You are evaluating customer service responses for tone and professionalism.
    
    User Question: {{ inputs }}
    Agent Response: {{ outputs }}
    
    Evaluate the response on a scale of 1-5 based on:
    1. Professional language (no slang or overly casual tone)
    2. Appropriate empathy for the customer's situation
    3. Clear and helpful communication
    4. Respectful and courteous tone
    
    Scoring Guide:
    5 = Excellent: Professional, empathetic, clear, and helpful
    4 = Good: Professional with minor tone issues
    3 = Acceptable: Professional but lacks empathy or clarity
    2 = Poor: Unprofessional or inappropriate tone
    1 = Unacceptable: Rude, dismissive, or completely inappropriate
    
    Provide your score (1-5) and explain your reasoning.
    """
)

print("✅ Custom tone judge created")
print(f"   Name: {tone_judge.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Another Custom Judge (Context-Aware)

# COMMAND ----------

# Judge that considers retrieved context
context_judge = make_judge(
    name="context_utilization",
    criteria="""
    Evaluate how effectively the agent used the retrieved context.
    
    User Question: {{ inputs }}
    Retrieved Context: {{ context }}
    Agent Response: {{ outputs }}
    
    Assess on a scale of 1-5:
    1. Did the agent use information from the retrieved documents?
    2. Did it miss important information available in the context?
    3. Did it add information not present in the context (hallucination)?
    
    Scoring:
    5 = Perfectly utilized all relevant context
    4 = Good use, minor omissions
    3 = Partial use, some key info missed
    2 = Minimal context utilization
    1 = Ignored context or hallucinated
    
    Explain which documents were used and what was missed.
    """
)

print("✅ Context utilization judge created")
print(f"   Name: {context_judge.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Code-Based Scorers

# COMMAND ----------

from mlflow.metrics.genai import scorer

@scorer
def response_length_checker(inputs, outputs):
    """
    Check if response length is appropriate
    Too short = incomplete, too long = verbose
    """
    # Get response text
    response = outputs if isinstance(outputs, str) else str(outputs)
    
    # Count words
    word_count = len(response.split())
    
    # Define acceptable range
    min_words = 20
    max_words = 150
    
    # Calculate score
    if min_words <= word_count <= max_words:
        score = 1.0
        rationale = f"✓ Appropriate length: {word_count} words"
    elif word_count < min_words:
        score = 0.0
        rationale = f"✗ Too short: {word_count} words (min: {min_words})"
    else:
        score = 0.0
        rationale = f"✗ Too verbose: {word_count} words (max: {max_words})"
    
    return {
        "score": score,
        "rationale": rationale,
        "word_count": word_count
    }

print("✅ Length checker scorer created")

# COMMAND ----------

@scorer
def required_keywords_scorer(inputs, outputs):
    """
    Check if response contains required keywords based on question type
    """
    question = inputs.get("question", "").lower() if isinstance(inputs, dict) else str(inputs).lower()
    response = outputs.lower() if isinstance(outputs, str) else str(outputs).lower()
    
    # Define required keywords by topic
    keyword_rules = {
        "return": ["return", "policy", "days"],
        "ship": ["shipping", "days", "business"],
        "warrant": ["warranty", "defect", "year"]
    }
    
    # Determine which rule applies
    required_keywords = []
    for topic, keywords in keyword_rules.items():
        if topic in question:
            required_keywords = keywords
            break
    
    if not required_keywords:
        return {
            "score": None,
            "rationale": "No specific keyword requirements for this question type"
        }
    
    # Check presence
    present = [kw for kw in required_keywords if kw in response]
    missing = [kw for kw in required_keywords if kw not in response]
    
    score = len(present) / len(required_keywords)
    
    return {
        "score": score,
        "rationale": f"Keywords present: {present}, missing: {missing}",
        "keywords_present": len(present),
        "keywords_total": len(required_keywords)
    }

print("✅ Keyword checker scorer created")

# COMMAND ----------

@scorer
def response_time_scorer(inputs, outputs, trace):
    """
    Check response time from trace
    """
    if trace is None:
        return {
            "score": None,
            "rationale": "No trace available for timing analysis"
        }
    
    # Calculate duration in seconds
    duration_ms = trace.info.end_time_ms - trace.info.start_time_ms
    duration_seconds = duration_ms / 1000.0
    
    # Define threshold
    threshold_seconds = 3.0
    
    # Calculate score
    if duration_seconds <= threshold_seconds:
        score = 1.0
        rationale = f"✓ Fast response: {duration_seconds:.2f}s"
    else:
        # Degrade score linearly
        score = max(0.0, 1.0 - (duration_seconds - threshold_seconds) / threshold_seconds)
        rationale = f"⚠ Slow response: {duration_seconds:.2f}s (threshold: {threshold_seconds}s)"
    
    return {
        "score": score,
        "rationale": rationale,
        "duration_seconds": duration_seconds
    }

print("✅ Response time scorer created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Combine All Evaluators

# COMMAND ----------

from mlflow.metrics.genai import answer_correctness, safety

# Combine built-in and custom evaluators
all_evaluators = [
    # Built-in judges
    answer_correctness(),
    safety(),
    
    # Custom LLM judges
    tone_judge,
    context_judge,
    
    # Code-based scorers
    response_length_checker,
    required_keywords_scorer,
    response_time_scorer
]

print("Complete evaluation suite:")
for evaluator in all_evaluators:
    eval_name = evaluator.name if hasattr(evaluator, 'name') else evaluator.__name__
    print(f"  ✓ {eval_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Create Test Agent

# COMMAND ----------

import mlflow

mlflow.openai.autolog()

@mlflow.trace(span_type="RETRIEVER")
def retrieve_docs(query: str):
    """Simulated retrieval"""
    return [
        {
            "content": "Returns accepted within 30 days with original packaging.",
            "doc_uri": "policies/returns.pdf"
        }
    ]

@mlflow.trace
def test_agent(question: str):
    """Simple test agent"""
    docs = retrieve_docs(question)
    context = docs[0]["content"]
    
    # Simulated response
    response = f"According to our policy: {context}"
    return response

# Test
test_response = test_agent("What is your return policy?")
print(f"Test response: {test_response}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Load Dataset

# COMMAND ----------

CATALOG = "main"
SCHEMA = "default"
DATASET_NAME = "agent_eval_dataset"

dataset_full_name = f"{CATALOG}.{SCHEMA}.{DATASET_NAME}"

eval_dataset = mlflow.genai.datasets.load_dataset(
    name=dataset_full_name
)

print(f"✅ Loaded dataset: {len(eval_dataset)} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Run Evaluation with Custom Evaluators

# COMMAND ----------

print("Starting evaluation with custom evaluators...")
print("This may take several minutes...")

results = mlflow.genai.evaluate(
    model=test_agent,
    data=eval_dataset,
    model_type="databricks-agent",
    evaluators=all_evaluators
)

print("\n✅ Evaluation complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: View Custom Scorer Results

# COMMAND ----------

eval_table = results.tables["eval_results"]

print("="*60)
print("CUSTOM EVALUATOR RESULTS")
print("="*60)

# Show all available metrics
print("\nAvailable metrics:")
for col in eval_table.columns:
    if 'score' in col or 'rationale' in col:
        print(f"  - {col}")

# Display aggregate metrics
print("\n" + "="*60)
print("AGGREGATE SCORES")
print("="*60)

for metric_name, value in sorted(results.metrics.items()):
    if 'mean' in metric_name:
        print(f"{metric_name}: {value:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Analyze Custom Judge Results

# COMMAND ----------

# Analyze tone quality
if "customer_tone_quality/score" in eval_table.columns:
    tone_scores = eval_table["customer_tone_quality/score"]
    
    print("="*60)
    print("TONE QUALITY ANALYSIS")
    print("="*60)
    print(f"Average Score: {tone_scores.mean():.2f} / 5.0")
    print(f"Min Score: {tone_scores.min():.2f}")
    print(f"Max Score: {tone_scores.max():.2f}")
    
    # Show cases with poor tone
    poor_tone = eval_table[eval_table["customer_tone_quality/score"] <= 2]
    if len(poor_tone) > 0:
        print(f"\n⚠️  {len(poor_tone)} cases with poor tone detected")
        for idx, row in poor_tone.head(2).iterrows():
            print(f"\nCase {idx}:")
            print(f"  Question: {row['inputs']['question']}")
            print(f"  Score: {row['customer_tone_quality/score']}")
            print(f"  Rationale: {row['customer_tone_quality/rationale'][:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Analyze Code-Based Scorer Results

# COMMAND ----------

# Analyze length checker results
if "response_length_checker/score" in eval_table.columns:
    length_scores = eval_table["response_length_checker/score"]
    
    print("="*60)
    print("RESPONSE LENGTH ANALYSIS")
    print("="*60)
    print(f"Pass Rate: {length_scores.mean() * 100:.1f}%")
    
    # Show length violations
    length_violations = eval_table[eval_table["response_length_checker/score"] == 0]
    if len(length_violations) > 0:
        print(f"\n⚠️  {len(length_violations)} length violations")
        for idx, row in length_violations.head(2).iterrows():
            print(f"\nCase {idx}:")
            print(f"  {row['response_length_checker/rationale']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Compare Built-in vs Custom Evaluators

# COMMAND ----------

import pandas as pd

# Create comparison DataFrame
comparison = pd.DataFrame({
    'Evaluator Type': ['Built-in', 'Built-in', 'Custom LLM', 'Custom LLM', 
                       'Code-based', 'Code-based', 'Code-based'],
    'Name': ['answer_correctness', 'safety', 'customer_tone_quality', 
             'context_utilization', 'response_length_checker', 
             'required_keywords_scorer', 'response_time_scorer'],
    'What it Evaluates': [
        'Factual correctness',
        'Safety/harmful content',
        'Tone and professionalism',
        'Context utilization',
        'Response length',
        'Required keywords',
        'Response speed'
    ]
})

print("="*60)
print("EVALUATOR COMPARISON")
print("="*60)
display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC **What we accomplished:**
# MAGIC 1. ✅ Created custom LLM judges with natural language criteria
# MAGIC 2. ✅ Used template variables (inputs, outputs, context)
# MAGIC 3. ✅ Created code-based scorers for deterministic checks
# MAGIC 4. ✅ Combined built-in and custom evaluators
# MAGIC 5. ✅ Analyzed results from different evaluator types
# MAGIC 6. ✅ Understood when to use each evaluator type
# MAGIC 
# MAGIC **Key Takeaways:**
# MAGIC - Custom LLM judges: For subjective, nuanced evaluation
# MAGIC - Code-based scorers: For deterministic, fast checks
# MAGIC - Combine both: For comprehensive evaluation
# MAGIC 
# MAGIC **Next Steps:**
# MAGIC - Proceed to Notebook 5: Production Monitoring
# MAGIC - Learn how to monitor quality in production
