from typing import Any, Optional, Sequence, Union

import mlflow
import pandas as pd
from databricks_langchain import ChatDatabricks
from databricks_langchain.uc_ai import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.models import ModelConfig
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from sklearn.feature_extraction.text import TfidfVectorizer

databricks_docs_url = "https://raw.githubusercontent.com/databricks/genai-cookbook/refs/heads/main/quick_start_demo/chunked_databricks_docs_filtered.jsonl"
parsed_docs_df = pd.read_json(databricks_docs_url, lines=True)

documents = parsed_docs_df
doc_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = doc_vectorizer.fit_transform(documents["content"])


@tool
@mlflow.trace(name="LittleIndex", span_type=mlflow.entities.SpanType.RETRIEVER)
def find_relevant_documents(query: str, top_n: int = 5) -> list[dict[str, Any]]:
    """gets relevant documents for the query"""
    query_tfidf = doc_vectorizer.transform([query])
    similarities = (tfidf_matrix @ query_tfidf.T).toarray().flatten()
    ranked_docs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    result = []
    for idx, score in ranked_docs[:top_n]:
        row = documents.iloc[idx]
        content = row["content"]
        doc_entry = {
            "page_content": content,
            "metadata": {
                "doc_uri": row["doc_uri"],
                "score": score,
            },
        }
        result.append(doc_entry)
    return result


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    agent_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    def routing_logic(state: ChatAgentState):
        last_message = state["messages"][-1]
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if agent_prompt:
        system_message = {"role": "system", "content": agent_prompt}
        preprocessor = RunnableLambda(
            lambda state: [system_message] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        routing_logic,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


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
            tools=self.tools,
            agent_prompt=self.config.get("system_prompt"),
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
    

# TODO fill in your catalog and schema name
catalog = "agentic_ai"
schema = "databricks"

# TODO: Replace with your model serving endpoint
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

baseline_config = {
    "endpoint_name": LLM_ENDPOINT,
    "temperature": 0.01,
    "max_tokens": 1000,
    "system_prompt": """You are a helpful assistant that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.

    You answer questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.
    """,
}

tools = [find_relevant_documents]
uc_client = DatabricksFunctionClient()
set_uc_function_client(uc_client)
uc_toolkit = UCFunctionToolkit(function_names=[f"{catalog}.{schema}.*"])
tools.extend(uc_toolkit.tools)


AGENT = DocsAgent(baseline_config, tools)
mlflow.models.set_model(AGENT)
