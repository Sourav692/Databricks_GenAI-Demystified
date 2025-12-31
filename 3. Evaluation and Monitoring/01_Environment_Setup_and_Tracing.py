# Databricks notebook source
# MAGIC %md
# MAGIC # Module 1: Environment Setup and Agent Tracing
# MAGIC
# MAGIC **Objective:** Set up MLflow 3 environment and learn basic tracing concepts
# MAGIC
# MAGIC **What you'll learn:**
# MAGIC - Install and verify MLflow 3.1+
# MAGIC - Configure MLflow tracking
# MAGIC - Enable autologging
# MAGIC - Instrument functions with @mlflow.trace
# MAGIC - Use span_type for RAG systems
# MAGIC - Capture and view trace IDs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Required Libraries

# COMMAND ----------

# Install MLflow 3.1+ with Databricks extras and OpenAI library
%pip install -q --upgrade "mlflow[databricks]>=3.1.0" openai

# Restart Python kernel to load new packages
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Verify Installation

# COMMAND ----------

import mlflow
from packaging import version

print("="*60)
print("INSTALLATION VERIFICATION")
print("="*60)

# Check MLflow version
required_version = version.parse("3.1.0")
current_version = version.parse(mlflow.__version__)

print(f"\nMLflow Version: {mlflow.__version__}")
if current_version >= required_version:
    print("✅ MLflow version is compatible")
else:
    print(f"❌ Please upgrade MLflow to 3.1.0 or higher")

# Check OpenAI library
try:
    import openai
    print(f"\n✅ OpenAI library installed: {openai.__version__}")
except ImportError:
    print("\n❌ OpenAI library not found")

print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Configure MLflow Tracking

# COMMAND ----------

# Set tracking URI to Databricks
mlflow.set_tracking_uri("databricks")

# Create or set an experiment
# Replace with your desired experiment path
experiment_name = "/Users/sourav.banerjee@databricks.com/agent-evaluation-demo"
mlflow.set_experiment(experiment_name)

print(f"✓ Tracking URI set to: {mlflow.get_tracking_uri()}")
print(f"✓ Experiment set to: {experiment_name}")

# Get experiment details
experiment = mlflow.get_experiment_by_name(experiment_name)
print(f"\nExperiment ID: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Enable Autologging

# COMMAND ----------

# Enable autologging for OpenAI/Databricks Foundation Models
mlflow.openai.autolog()

print("✓ Autologging enabled")
print("  All LLM calls will be automatically traced")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Create a Simple Function with Tracing

# COMMAND ----------

import mlflow

# Simple function with basic tracing
@mlflow.trace
def simple_function(text):
    """
    A simple function to demonstrate basic tracing
    This will create a span capturing inputs and outputs
    """
    # Simulate some processing
    result = f"Processed: {text.upper()}"
    return result

# Test the function
output = simple_function("hello world")
print(f"Output: {output}")

# Get the trace ID of the last execution
trace_id = mlflow.get_last_active_trace_id()
print(f"\n✓ Trace ID: {trace_id}")
print("  View this trace in the MLflow UI Traces tab")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Simulated Retrieval Function with span_type

# COMMAND ----------

@mlflow.trace(span_type="RETRIEVER")
def retrieve_documents(query: str):
    """
    Simulates document retrieval from a knowledge base
    
    The span_type="RETRIEVER" is CRITICAL for RAG judges to work:
    - Faithfulness judge needs this to find retrieved context
    - Retrieval Precision judge needs this
    - Document Recall judge needs this
    """
    # Simulated retrieval results
    # In production, this would call a vector database
    documents = [
        {
            "content": "Our return policy allows returns within 30 days of purchase.",
            "doc_uri": "policies/returns.pdf",
            "score": 0.92
        },
        {
            "content": "Items must be in original packaging with receipt.",
            "doc_uri": "policies/returns.pdf",
            "score": 0.85
        },
        {
            "content": "Refunds are processed within 5-7 business days.",
            "doc_uri": "policies/refunds.pdf",
            "score": 0.78
        }
    ]
    
    print(f"Retrieved {len(documents)} documents for query: '{query}'")
    return documents

# Test the retrieval function
query = "What is your return policy?"
docs = retrieve_documents(query)

for i, doc in enumerate(docs, 1):
    print(f"\nDocument {i}:")
    print(f"  Content: {doc['content'][:60]}...")
    print(f"  Source: {doc['doc_uri']}")
    print(f"  Score: {doc['score']}")

# Get trace ID
trace_id = mlflow.get_last_active_trace_id()
print(f"\n✓ Trace ID: {trace_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Complete RAG Agent Simulation

# COMMAND ----------

@mlflow.trace(span_type="RETRIEVER")
def search_knowledge_base(query: str):
    """
    Search function marked with RETRIEVER span type
    """
    documents = [
        {
            "content": "Product X costs $299 and includes free shipping.",
            "doc_uri": "products/product-x.pdf"
        },
        {
            "content": "Product X has a 1-year warranty covering defects.",
            "doc_uri": "warranty/product-x-warranty.pdf"
        }
    ]
    return documents

@mlflow.trace
def format_context(documents):
    """
    Format retrieved documents into a context string
    """
    context = "\n\n".join([
        f"Source: {doc['doc_uri']}\n{doc['content']}"
        for doc in documents
    ])
    return context

@mlflow.trace
def generate_response(question: str, context: str):
    """
    Simulates LLM response generation
    In production, this would call an actual LLM
    """
    # Simulated response based on context
    response = f"Based on the documentation: {context[:100]}..."
    return response

@mlflow.trace
def rag_agent(question: str):
    """
    Main RAG agent function
    Orchestrates retrieval and generation
    """
    print(f"Processing question: {question}")
    
    # Step 1: Retrieve documents (creates RETRIEVER span)
    documents = search_knowledge_base(question)
    print(f"  ✓ Retrieved {len(documents)} documents")
    
    # Step 2: Format context
    context = format_context(documents)
    print(f"  ✓ Formatted context")
    
    # Step 3: Generate response
    response = generate_response(question, context)
    print(f"  ✓ Generated response")
    
    return response

# Test the complete RAG agent
question = "How much does Product X cost?"
answer = rag_agent(question)

print(f"\nFinal Answer: {answer}")

# Get trace ID
trace_id = mlflow.get_last_active_trace_id()
print(f"\n✓ Complete trace captured")
print(f"  Trace ID: {trace_id}")
print(f"  This trace includes:")
print(f"    - Main agent span")
print(f"    - RETRIEVER span (for RAG judges)")
print(f"    - Context formatting span")
print(f"    - Response generation span")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Verify Trace Structure

# COMMAND ----------

from mlflow.tracking import MlflowClient

# Create MLflow client
client = MlflowClient()

# Get the last trace
trace_id = mlflow.get_last_active_trace_id()
trace = client.get_trace(trace_id)

print("="*60)
print("TRACE STRUCTURE")
print("="*60)

print(f"\nTrace ID: {trace.info.request_id}")
print(f"Total Spans: {len(trace.data.spans)}")

# List all spans
print("\nSpans in this trace:")
for i, span in enumerate(trace.data.spans, 1):
    print(f"\n{i}. Span: {span.name}")
    print(f"   Type: {span.span_type if hasattr(span, 'span_type') and span.span_type else 'None'}")
    print(f"   Duration: {span.end_time_ms - span.start_time_ms if hasattr(span, 'end_time_ms') else 'N/A'} ms")

# Check for RETRIEVER spans (critical for RAG judges)
retriever_spans = [
    span for span in trace.data.spans 
    if hasattr(span, 'span_type') and span.span_type == "RETRIEVER"
]

print(f"\n{'='*60}")
if retriever_spans:
    print(f"✅ Found {len(retriever_spans)} RETRIEVER span(s)")
    print("   RAG judges (faithfulness, retrieval_precision) will work!")
else:
    print("❌ No RETRIEVER spans found")
    print("   Add span_type='RETRIEVER' to your retrieval functions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Complete Setup Verification

# COMMAND ----------

from packaging import version

def verify_setup():
    """
    Complete verification of environment setup
    """
    print("="*60)
    print("COMPLETE SETUP VERIFICATION")
    print("="*60)
    
    checks = []
    
    # 1. MLflow version
    version_ok = version.parse(mlflow.__version__) >= version.parse("3.1.0")
    checks.append(("MLflow >= 3.1.0", version_ok))
    
    # 2. OpenAI installed
    try:
        import openai
        openai_ok = True
    except ImportError:
        openai_ok = False
    checks.append(("OpenAI library", openai_ok))
    
    # 3. Tracking configured
    tracking_ok = mlflow.get_tracking_uri() == "databricks"
    checks.append(("Tracking URI (databricks)", tracking_ok))
    
    # 4. Experiment set
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        exp_ok = exp is not None
    except:
        exp_ok = False
    checks.append(("Experiment configured", exp_ok))
    
    # 5. Test tracing
    @mlflow.trace
    def test_trace():
        return "test"
    
    test_trace()
    trace_id = mlflow.get_last_active_trace_id()
    trace_ok = trace_id is not None
    checks.append(("Tracing functional", trace_ok))
    
    # Print results
    print("\nSetup Status:")
    for check_name, status in checks:
        icon = "✅" if status else "❌"
        print(f"  {icon} {check_name}")
    
    # Overall
    all_ok = all(status for _, status in checks)
    print("\n" + "="*60)
    if all_ok:
        print("✅ ALL CHECKS PASSED")
        print("   Ready to proceed to evaluation!")
    else:
        print("❌ SOME CHECKS FAILED")
        print("   Please address issues above")
    print("="*60)
    
    return all_ok

# Run verification
verify_setup()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What we accomplished:**
# MAGIC 1. ✅ Installed MLflow 3.1+ with required dependencies
# MAGIC 2. ✅ Configured MLflow tracking for Databricks
# MAGIC 3. ✅ Enabled autologging for LLM calls
# MAGIC 4. ✅ Created traced functions with @mlflow.trace
# MAGIC 5. ✅ Used span_type="RETRIEVER" for RAG systems
# MAGIC 6. ✅ Verified complete trace structure
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Proceed to Notebook 2: Building Evaluation Datasets
# MAGIC - Learn how to create test cases from traces, manual entry, and synthetic generation

# COMMAND ----------


