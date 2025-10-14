import json
import warnings
from typing import Any, Generator

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from openai import OpenAI

# TODO: Replace with your model serving endpoint
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"

# TODO: Update with your system prompt
SYSTEM_PROMPT = """
You are a helpful assistant that provides brief, clear responses.
"""


class SimpleChatAgent(ResponsesAgent):
    """
    Simple chat agent that calls an LLM using the Databricks OpenAI client API.

    You can replace this with your own agent.
    The decorators @mlflow.trace tell MLflow Tracing to track calls to the agent.
    """

    def __init__(self):
        self.workspace_client = WorkspaceClient()
        self.client: OpenAI = self.workspace_client.serving_endpoints.get_open_ai_client()
        self.llm_endpoint = LLM_ENDPOINT_NAME
        self.SYSTEM_PROMPT = SYSTEM_PROMPT

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in self.client.chat.completions.create(
                model=self.llm_endpoint,
                messages=self.prep_msgs_for_cc_llm(messages),
                stream=True,
            ):
                yield chunk.to_dict()

    # With autologging, you do not need @mlflow.trace here, but you can add it to override the span type.
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    # With autologging, you do not need @mlflow.trace here, but you can add it to override the span type.
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            i.model_dump() for i in request.input
        ]
        yield from self.output_to_responses_items_stream(chunks=self.call_llm(messages))


mlflow.openai.autolog()
AGENT = SimpleChatAgent()
mlflow.models.set_model(AGENT)
