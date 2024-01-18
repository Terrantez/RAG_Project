"""
Implementation of RAG agent using langchain
"""
import os

from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from llama_index_base import get_query_index

import utils
import models

index = get_query_index()


def news_reloader() -> str:
    """
    Reload the latest news after downloading them.
    :return: str
    """
    index.refresh_ref_docs(utils.get_latest_news())
    return "News reloaded successfully."


def news_reader(query: str) -> str:
    return str(index.as_query_engine().query(query))


# ------------------- Defining the tools the agent has access to -------------------
llama_index_tool = Tool(
    name=f"news_reader",
    func=lambda q: news_reader(q),
    description=f"Useful for when you need to read or summarize the latest news about a specific subject."
                f"Receives a detailed plain text question as input.",
    return_direct=True
)

download_news_tool = Tool(
    name=f"news_downloader",
    func=lambda subject: str(utils.download_latest_news(subject)),
    description=f"useful for when you need to download news about a specific subject."
                f"Receives a single word as input, the subject of the news. ",
    return_direct=True
)

tools = [
    llama_index_tool,
    download_news_tool
]

# ------------------- System prompt downloaded from langchain hub -------------------

prompt = hub.pull("hwchase17/react-chat")

# ------------------- Example of a custom ReACT system prompt -------------------

prompt_template = PromptTemplate(
    input_variables=['agent_scratchpad', 'chat_history', 'input', 'tool_names', 'tools'],
    template='Your name is "Assistant", a large language model trained by OpenAI.\n\n'
             'Your job is to comply with the user {input}.\n'
             'You have access to several tools that can help you.\n'
             'Make sure to think about what the user asked. Only use a tool if you need to.\n'
             'If you do not need to use a tool, move to the Final Answer".\n\n'
             'You have access to the following tools:\n\n'
             '{tools}\n\n'
             'When you need to use a tool please use the following format:\n\n'
             '```\nThought: I need to use a tool\n'
             'Action: the action to take, should be one of: [{tool_names}]\n'
             'Action Input: the input to the action\n'
             'Observation: the result of the action\n```\n\n'
             'You may perform multiple actions and use multiple tools sequentially.\n'
             'If you do not need to use a tool, move to the Final Answer".\n\n'
             'When you think that your answer is complete,'
             'you MUST respond with the format:\n\n'
             '```\nThought: I do not need to use a tool\n'
             'Final Answer: [your response here]\n```\n\n'
             'You will answer the following questions as best you can.\n' 
             'Begin!\n\n'
             'Previous conversation history:\n'
             '{chat_history}\n\n'
             '{agent_scratchpad}'
)

# ------------------- Implementation of agent using Local LLM -------------------

# langchain_llm = models.get_langchain_llm_cpu()
# os.environ["OPENAI_API_KEY"] = "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"
#
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
#
# agent = create_react_agent(prompt=prompt_template, tools=tools, llm=llm)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#
# agent_executor.invoke(
#     {
#         "input": input(),
#         "chat_history": "",
#     }
# )

# ------------- Implementation og agent using OpenAI LLM Through API (Requires API key as in environ) ------------

os.environ["OPENAI_API_KEY"] = "key"

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt_open_ai = hub.pull("hwchase17/openai-functions-agent")

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt_open_ai)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": input()})
while True:
    for step in agent_executor.stream({"input": input()}):
        print(step)
