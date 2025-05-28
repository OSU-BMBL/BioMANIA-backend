import json
import os
from unittest.mock import MagicMock, patch

import pytest

from biochatter.llm_connect import GptConversation
from biochatter.rag_agent import RagAgent, RagAgentModeEnum
from biochatter.vectorstore_agent import Document

def conversation_factory():
    conversation = GptConversation(
        model_name="gpt-4o",
        prompts={},
        correct=False,
    )
    conversation.set_api_key(os.getenv("OPENAI_API_KEY"), user="test")

    return conversation


def test_rag_agent_invalid_mode():
    with pytest.raises(ValueError) as excinfo:
        RagAgent(
            mode="invalid_mode",
            model_name="test_model",
            connection_args={},
        )
    assert "Invalid mode. Choose either 'kg', 'vectorstore', 'api_blast', 'api_oncokb' or 'api_scanpy'." in str(excinfo.value)


def conversation_factory():
    # Mock conversation factory
    return MagicMock()

@patch("biochatter.api_agent.python.scanpy.agent.ScanpyFetcher")
@patch("biochatter.api_agent.python.scanpy.agent.ScanpyInterpreter")
@patch("biochatter.api_agent.python.scanpy.agent.ScanpyQueryBuilder")
def test_rag_agent_api_scanpy_mode(
    mock_fetcher,
    mock_interpreter,
    mock_builder,
):
    rag_agent = RagAgent(
        mode=RagAgentModeEnum.API_SCANPI,
        model_name="gpt-4o",
        use_prompt=True,
        conversation_factory=conversation_factory,
    )
    assert rag_agent.mode == RagAgentModeEnum.API_SCANPI
    question = "What is the sequence of the gene?"
    response = rag_agent.generate_responses(question)
    assert response is not None
    
