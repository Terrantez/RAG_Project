"""
Implementation of RAG agent using llama-index
"""
import os

from llama_index.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.agent import OpenAIAgent, ReActAgent

from utils import get_latest_news, download_latest_news
from llama_index_base import get_query_index
from models import get_llama_index_cpu

index = get_query_index()

documents = get_latest_news()


def news_reloader() -> str:
    """
    Reload the latest news after downloading them.
    :return: str
    """
    index.refresh_ref_docs(get_latest_news())
    return "News reloaded successfully."


def news_reader(query: str) -> str:
    """
    Read information about the latest news on a subject.
    :param query: A detailed plain text question as input to the tool.
    :return: Latest news information
    """
    return str(index.as_query_engine().query(query))


# ------------------- Defining the tools the agent has access to -------------------

tools = [
    # QueryEngineTool(
    #     query_engine=index.as_query_engine(),
    #     metadata=ToolMetadata(
    #         name="news_reader",
    #         description=(
    #             "Provides information about the latest news on a subject."
    #             "Use a detailed plain text question as input to the tool."
    #         ),
    #     ),
    # ),
    FunctionTool.from_defaults(news_reader),
    FunctionTool.from_defaults(download_latest_news),
    FunctionTool.from_defaults(news_reloader)
]

# ------------------- Implementation of agent using Local LLM -------------------

# llm = get_llama_index_cpu()
# os.environ["OPENAI_API_KEY"] = "key"
#
# agent = ReActAgent.from_tools(
#     tools,
#     # llm=llm,
#     verbose=True,
#     context="You are a news expert and are able to read the latest news on a subject using a tool. "
#             "Only talk about the news from your tool, not from your knowledge."
#             "If there are none, you can download them using a tool. "
#             "After downloading you MUST reload the news."
#             "Finally you MUST use the read the news again."
# )
#
# while True:
#     response = agent.stream_chat(input())
#     response.print_response_stream()

# ------------- Implementation og agent using OpenAI LLM Through API (Requires API key as in environ) ------------

os.environ["OPENAI_API_KEY"] = "key"

agent = OpenAIAgent.from_tools(tools, verbose=True)

while True:
    response = agent.stream_chat(input())
    response.print_response_stream()

