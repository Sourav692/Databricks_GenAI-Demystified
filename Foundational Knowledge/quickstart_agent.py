
import json
import uuid
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, DatabricksFunctionClient
from typing import Any, Optional

import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext

# Get an OpenAI client configured to talk to Databricks model serving endpoints
# We'll use this to query an LLM in our agent
openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()

# The snippet below tries to pick the first LLM API available in your Databricks workspace
# from a set of candidates. You can override and simplify it
# to just specify LLM_ENDPOINT_NAME.
LLM_ENDPOINT_NAME = None

def is_endpoint_available(endpoint_name):
  try:
    client = WorkspaceClient().serving_endpoints.get_open_ai_client()
    client.chat.completions.create(model=endpoint_name, messages=[{"role": "user", "content": "What is AI?"}])
    return True
  except Exception:
    return False
  
for candidate_endpoint_name in ["databricks-claude-3-7-sonnet", "databricks-meta-llama-3-3-70b-instruct"]:
    if is_endpoint_available(candidate_endpoint_name):
      LLM_ENDPOINT_NAME = candidate_endpoint_name
assert LLM_ENDPOINT_NAME is not None, "Please specify LLM_ENDPOINT_NAME"

# Enable automatic tracing of LLM calls
mlflow.openai.autolog()

# Load Databricks built-in tools (a stateless Python code interpreter tool)
client = DatabricksFunctionClient()
builtin_tools = UCFunctionToolkit(function_names=["system.ai.python_exec"], client=client).tools
for tool in builtin_tools:
    del tool["function"]["strict"]

def call_tool(tool_name, parameters):
    if tool_name == "system__ai__python_exec":
        return DatabricksFunctionClient().execute_function("system.ai.python_exec", parameters=parameters)
    raise ValueError(f"Unknown tool: {tool_name}")

def run_agent(prompt):
    """
    Send a user prompt to the LLM, and return a list of LLM response messages
    The LLM is allowed to call the code interpreter tool if needed, to respond to the user
    """
    result_msgs = []
    response = openai_client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=[{"role": "user", "content": prompt}],
        tools=builtin_tools,
    )
    msg = response.choices[0].message
    result_msgs.append(msg.to_dict())

    # If the model executed a tool, call it
    if msg.tool_calls:
        call = msg.tool_calls[0]
        tool_result = call_tool(call.function.name, json.loads(call.function.arguments))
        result_msgs.append({"role": "tool", "content": tool_result.value, "name": call.function.name, "tool_call_id": call.id})
    return result_msgs

class QuickstartAgent(ChatAgent):
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        prompt = messages[-1].content
        raw_msgs = run_agent(prompt)
        out = []
        for m in raw_msgs:
            out.append(ChatAgentMessage(
                id=uuid.uuid4().hex,
                **m
            ))

        return ChatAgentResponse(messages=out)

AGENT = QuickstartAgent()
mlflow.models.set_model(AGENT)
