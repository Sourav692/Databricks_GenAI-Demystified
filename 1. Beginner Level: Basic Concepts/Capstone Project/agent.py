
from getpass import getpass
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from databricks_langchain.uc_ai import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)

import mlflow
from langchain_core.tools import tool
from databricks_langchain import ChatDatabricks

from langgraph.graph import StateGraph, START, END
from typing import Annotated, Literal,Any, Optional, Sequence, Union
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolNode,tools_condition
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph.graph import CompiledGraph
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from mlflow.models import ModelConfig

mlflow.langchain.autolog()

# class State(TypedDict):
#     messages: Annotated[list, add_messages]

def create_tool_calling_agent(
    llm: LanguageModelLike,
    tools: list
) -> CompiledGraph:
    
    llm_with_tools = llm.bind_tools(tools=tools)

    def routing_logic(state: ChatAgentState):
        last_message = state["messages"][-1]
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    preprocessor = RunnableLambda(lambda state: state["messages"])

    model_runnable = preprocessor | llm_with_tools

    # Augmented LLM with Tools Node function
    def tool_calling_llm(
        state: ChatAgentState,
        config: RunnableConfig):

        response = model_runnable.invoke(state, config)
        return {"messages": [response]}
    
    # Build the graph
    builder = StateGraph(ChatAgentState)
    builder.add_node("tool_calling_llm", RunnableLambda(tool_calling_llm))
    builder.add_node("tools", ChatAgentToolNode(tools=tools))
    builder.add_edge(START, "tool_calling_llm")

    # # Conditional Edge
    # builder.add_conditional_edges(
    #     "tool_calling_llm",
    #     # If the latest message (result) from LLM is a tool call -> tools_condition routes to tools
    #     # If the latest message (result) from LLM is a not a tool call -> tools_condition routes to END
    #     tools_condition,
    #     ["tools", END]
    # )

    builder.add_conditional_edges(
        "tool_calling_llm",
        routing_logic,
        {
            "continue": "tools",
            "end": END,
        },
    )
    builder.add_edge("tools", "tool_calling_llm") # this is the key feedback loop
    agent = builder.compile()

    return agent

class DocsAgent(ChatAgent):
    def __init__(self, config, tools):
        # Load config
        # When this agent is deployed to Model Serving, the configuration loaded here is replaced with the config passed to mlflow.pyfunc.log_model(model_config=...)
        self.config = ModelConfig(development_config=config)
        self.tools = tools
        self.agent = self._build_agent_from_config()

    def _build_agent_from_config(self):
        llm = ChatDatabricks(
            endpoint=self.config.get("endpoint_name"),
            temperature=self.config.get("temperature"),
            max_tokens=self.config.get("max_tokens"),
        )
        agent = create_tool_calling_agent(
            llm,
            tools=self.tools
        )
        return agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        # ChatAgent has a built-in helper method to help convert framework-specific messages, like langchain BaseMessage to a python dictionary
        request = {"messages": self._convert_messages_to_dict(messages)}

        output = self.agent.invoke(request)
        # Here 'output' is already a ChatAgentResponse, but to make the ChatAgent signature explicit for this demonstration we are returning a new instance
        return ChatAgentResponse(**output)
    

catalog = "agentic_ai"
schema = "databricks"

# TODO: Replace with your model serving endpoint
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

baseline_config = {
    "endpoint_name": LLM_ENDPOINT,
    "temperature": 0.01,
    "max_tokens": 1000
}

uc_client = DatabricksFunctionClient()
set_uc_function_client(uc_client)
uc_tool_names = [f"{catalog}.{schema}.search_web"]
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools=[*uc_toolkit.tools]

AGENT = DocsAgent(baseline_config, tools)
mlflow.models.set_model(AGENT)
