# Databricks notebook source
# MAGIC %md
# MAGIC # Module 2: Building Evaluation Datasets
# MAGIC
# MAGIC **Objective:** Create and manage evaluation datasets in Unity Catalog
# MAGIC
# MAGIC **What you'll learn:**
# MAGIC - Create evaluation datasets in Unity Catalog
# MAGIC - Add records from manual entry
# MAGIC - Generate synthetic test cases
# MAGIC - Understand dataset schema (inputs, expectations, metadata)
# MAGIC - Work with reserved keys for built-in judges

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites Check

# COMMAND ----------

import mlflow
from packaging import version

# Verify MLflow version
assert version.parse(mlflow.__version__) >= version.parse("3.1.0"), \
    "Please run Notebook 01 first to install MLflow 3.1+"

print(f"✅ MLflow {mlflow.__version__} ready")

# Set experiment
experiment_name = "/Users/sourav.banerjee@databricks.com/agent-evaluation-demo"
mlflow.set_experiment(experiment_name)
print(f"✅ Experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Define Unity Catalog Location
# MAGIC
# MAGIC **Note:** Replace with your actual catalog and schema

# COMMAND ----------

# Define your Unity Catalog location
# Format: catalog.schema.table_name
CATALOG = "main"  # Replace with your catalog
SCHEMA = "default"  # Replace with your schema
DATASET_NAME = "agent_eval_dataset"

dataset_full_name = f"{CATALOG}.{SCHEMA}.{DATASET_NAME}"

print(f"Dataset location: {dataset_full_name}")
print(f"\nNote: Ensure you have CREATE TABLE permissions in {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Evaluation Dataset

# COMMAND ----------

import mlflow.genai.datasets
from requests import HTTPError

# Create a new evaluation dataset in Unity Catalog
# If it already exists, load the existing one
try:
    eval_dataset = mlflow.genai.datasets.get_dataset(
        name=dataset_full_name
    )
    print(f"✅ Dataset created: {eval_dataset.name}")
    # print(f"   Version: {eval_dataset.version}")
except HTTPError as e:
    if "TABLE_ALREADY_EXISTS" in str(e):
        print(f"ℹ️  Dataset already exists, loading existing dataset...")
        eval_dataset = mlflow.genai.datasets.get_dataset(
            name=dataset_full_name
        )
        print(f"✅ Loaded existing dataset: {eval_dataset.name}")
        print(f"   Current version: {eval_dataset.version}")
        print(f"   Current records: {len(eval_dataset)}")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Manual Entry - Create Test Cases

# COMMAND ----------

import pandas as pd

# Create manual test cases
# This demonstrates the correct schema structure
manual_test_cases = pd.DataFrame([
    {
        # INPUTS: What the user provides to the agent
        "inputs": {
            "question": "What's your return policy for electronics?"
        },
        
        # EXPECTATIONS: Ground truth for evaluation
        "expectations": {
            # expected_facts: Used by Answer Correctness judge
            "expected_facts": [
                "30-day return window",
                "Original packaging required",
                "Receipt or order number needed",
                "Refunds processed in 5-7 business days"
            ],
            
            # expected_response: Ideal complete answer
            "expected_response": 
                "Electronics can be returned within 30 days of purchase. "
                "The item must be in its original packaging, and you'll need "
                "your receipt or order number. Refunds are typically processed "
                "within 5-7 business days after we receive the return.",
            
            # guidelines: Used by ExpectationsGuidelines judge
            "guidelines":
                "Response must mention the 30-day window, packaging requirement, "
                "and refund processing time. Tone should be helpful and clear."
        },
        
        # METADATA: Organizational information
        "metadata": {
            "category": "returns",
            "difficulty": "easy",
            "source": "manual",
            "priority": "high"
        }
    },
    {
        "inputs": {
            "question": "Can I return a customized laptop?"
        },
        "expectations": {
            "expected_facts": [
                "Customized items are typically non-returnable",
                "Exceptions for defective items",
                "Contact support for special cases"
            ],
            "guidelines":
                "Must clearly state non-returnable policy for customized items. "
                "Must mention exception for defects. Tone should be empathetic "
                "but firm about policy."
        },
        "metadata": {
            "category": "returns",
            "difficulty": "medium",
            "source": "manual",
            "priority": "high"
        }
    },
    {
        "inputs": {
            "question": "How long does shipping take?"
        },
        "expectations": {
            "expected_facts": [
                "Standard shipping: 3-5 business days",
                "Express shipping: 1-2 business days",
                "Free shipping on orders over $50"
            ],
            "expected_response":
                "We offer two shipping options: Standard shipping takes 3-5 business days "
                "and costs $5.99, while Express shipping takes 1-2 business days and costs "
                "$14.99. Orders over $50 qualify for free standard shipping.",
            "guidelines":
                "Must clearly differentiate between shipping options. "
                "Must mention costs and timeframes for each option."
        },
        "metadata": {
            "category": "shipping",
            "difficulty": "easy",
            "source": "manual",
            "priority": "medium"
        }
    }
])

print(f"Created {len(manual_test_cases)} manual test cases")
print("\nSchema structure:")
print("  - inputs: What the user asks")
print("  - expectations: Ground truth for judges")
print("  - metadata: Organization and context")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Add Manual Cases to Dataset

# COMMAND ----------

# Add manual test cases to the dataset
eval_dataset.merge_records(
    records=manual_test_cases
)

print(f"✅ Added {len(manual_test_cases)} manual test cases")
print(f"   Dataset now has {len(eval_dataset.to_df())} total records")

# COMMAND ----------

display(eval_dataset.to_df())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Synthetic Data Generation (Optional)
# MAGIC
# MAGIC **Note:** This requires documents to generate from. 
# MAGIC We'll create sample documents for demonstration.

# COMMAND ----------

# Create sample documents for synthetic generation
sample_documents = pd.DataFrame([
    {
        "content": """
        RETURN POLICY
        
        We accept returns within 30 days of purchase for most items.
        Items must be in original packaging and unused condition.
        A receipt or order confirmation is required.
        
        Refunds are processed within 5-7 business days after we receive the item.
        Original shipping costs are non-refundable.
        
        EXCEPTIONS:
        - Customized or personalized items cannot be returned
        - Software and digital products are non-returnable once opened
        - Items marked as "Final Sale" cannot be returned
        """,
        "doc_uri": "policies/return_policy.pdf"
    },
    {
        "content": """
        SHIPPING INFORMATION
        
        Standard Shipping: 3-5 business days - $5.99
        Express Shipping: 1-2 business days - $14.99
        
        FREE STANDARD SHIPPING on orders over $50
        
        We ship to all 50 US states.
        International shipping available to select countries.
        
        Tracking numbers are provided for all shipments.
        Orders placed before 2 PM EST ship same day.
        """,
        "doc_uri": "policies/shipping_info.pdf"
    },
    {
        "content": """
        WARRANTY INFORMATION
        
        All electronics include a 1-year limited warranty.
        
        The warranty covers:
        - Manufacturing defects
        - Hardware malfunctions under normal use
        
        The warranty does NOT cover:
        - Accidental damage
        - Water damage
        - Unauthorized modifications
        
        Extended warranty options available at checkout.
        """,
        "doc_uri": "policies/warranty.pdf"
    }
])

print(f"Created {len(sample_documents)} sample documents for synthetic generation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Generate Synthetic Test Cases

# COMMAND ----------

# Note: Uncomment and use this if you want to generate synthetic data
# This may take a few minutes and requires LLM access

from databricks.agents.evals import generate_evals_df

synthetic_cases = generate_evals_df(
    docs=sample_documents,
    agent_description="""
    This is a customer support chatbot for an e-commerce company.
    It helps customers with questions about returns, shipping, warranties,
    and general product information.
    """,
    question_guidelines="""
    Generate questions that:
    - Are natural and conversational
    - Vary in complexity from simple to multi-part questions
    - Reflect real customer concerns
    - Are specific and realistic
    """,
    num_evals=10  # Generate 10 synthetic test cases
)

print(f"Generated {len(synthetic_cases)} synthetic test cases")

# Add synthetic cases to dataset
eval_dataset.merge_records(
    records=synthetic_cases
)

print("Note: Synthetic generation completed")
print("      This requires LLM access and may incur costs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: View Dataset Contents

# COMMAND ----------

# Load dataset as Spark DataFrame
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.table(dataset_full_name)

print(f"Dataset: {dataset_full_name}")
print(f"Total records: {df.count()}")

# Show schema
print("\nDataset Schema:")
df.printSchema()

# Show sample records
print("\nSample Records:")
df.show(3, truncate=False, vertical=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Analyze Dataset by Category

# COMMAND ----------

# Convert to Pandas for easier analysis
pdf = df.toPandas()

# Extract category from metadata
pdf['category'] = pdf['metadata'].apply(lambda x: x.get('category', 'unknown'))
pdf['difficulty'] = pdf['metadata'].apply(lambda x: x.get('difficulty', 'unknown'))
pdf['source'] = pdf['metadata'].apply(lambda x: x.get('source', 'unknown'))

print("Dataset Composition:")
print("\nBy Category:")
print(pdf['category'].value_counts())

print("\nBy Difficulty:")
print(pdf['difficulty'].value_counts())

print("\nBy Source:")
print(pdf['source'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Examine Reserved Keys

# COMMAND ----------

# Show examples of reserved keys in expectations
print("="*60)
print("RESERVED KEYS IN EXPECTATIONS FIELD")
print("="*60)

print("\nThese keys are recognized by built-in judges:")
print("\n1. expected_facts (list)")
print("   - Used by: Answer Correctness judge")
print("   - Purpose: Key facts that must appear in response")

print("\n2. expected_response (string)")
print("   - Used by: Answer Correctness judge")
print("   - Purpose: Complete ideal response")

print("\n3. guidelines (string)")
print("   - Used by: ExpectationsGuidelines judge")
print("   - Purpose: Natural language evaluation criteria")

print("\n4. expected_retrieved_context (list of dicts)")
print("   - Used by: Document Recall judge")
print("   - Purpose: Documents that should be retrieved")

# Show actual example
print("\n" + "="*60)
print("EXAMPLE FROM DATASET:")
print("="*60)

sample_row = pdf.iloc[0]
print(f"\nQuestion: {sample_row['inputs']['question']}")
print(f"\nExpectations:")
for key, value in sample_row['expectations'].items():
    print(f"\n  {key}:")
    if isinstance(value, list):
        for item in value:
            print(f"    - {item}")
    else:
        print(f"    {value[:100]}..." if len(str(value)) > 100 else f"    {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Dataset Versioning

# COMMAND ----------

# Load current version
current_dataset = mlflow.genai.datasets.load_dataset(
    name=dataset_full_name
)

print(f"Current dataset version: {current_dataset.version}")
print(f"Total records: {len(current_dataset)}")

# Note: Every time you add records with merge_records(),
# a new version is automatically created

print("\nVersioning features:")
print("  ✓ Every change creates a new version")
print("  ✓ Can load any historical version")
print("  ✓ Full lineage tracking")
print("  ✓ Unity Catalog governance")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Best Practices Summary

# COMMAND ----------

print("="*60)
print("EVALUATION DATASET BEST PRACTICES")
print("="*60)

print("""
1. SCHEMA STRUCTURE
   ✓ inputs: User's question/input (dict)
   ✓ expectations: Ground truth (dict with reserved keys)
   ✓ metadata: Organization info (dict)

2. RESERVED KEYS (in expectations)
   ✓ expected_facts: List of key facts
   ✓ expected_response: Complete ideal answer
   ✓ guidelines: Natural language criteria
   ✓ expected_retrieved_context: Expected documents

3. DATA SOURCES
   ✓ Manual: Critical test cases
   ✓ Production traces: Real-world examples
   ✓ Synthetic: Broad coverage

4. METADATA TAGS
   ✓ category: Topic classification
   ✓ difficulty: easy/medium/hard
   ✓ source: Where it came from
   ✓ priority: Business importance

5. DATASET COMPOSITION
   ✓ 40% easy cases
   ✓ 40% medium cases
   ✓ 20% hard cases
   
6. QUALITY CONTROL
   ✓ Review synthetic data samples
   ✓ Validate schema compliance
   ✓ Regular dataset updates
   ✓ Version control
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What we accomplished:**
# MAGIC 1. ✅ Created evaluation dataset in Unity Catalog
# MAGIC 2. ✅ Added manual test cases with proper schema
# MAGIC 3. ✅ Understood reserved keys for built-in judges
# MAGIC 4. ✅ Learned about synthetic data generation
# MAGIC 5. ✅ Analyzed dataset composition
# MAGIC 6. ✅ Understood versioning and governance
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Proceed to Notebook 3: Running Evaluations with Built-in Judges
# MAGIC - Learn how to evaluate your agent using this dataset
