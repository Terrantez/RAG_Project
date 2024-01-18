import logging
import sys

from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index import ServiceContext, VectorStoreIndex

import models
import utils

# ------------------- Enable debugging messages -------------------

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# ------------------- Down custom embedding model from HuggingFace -------------------

embed_model = HuggingFaceEmbeddings(model_name="WhereIsAI/UAE-Large-V1")
llama_llm = models.get_llama_index_llm_gpu()

# ------------------- Prepare LLM for document interaction -------------------

service_context = ServiceContext.from_defaults(
    llm=llama_llm,
    chunk_size=512,
    chunk_overlap=64,
    embed_model=embed_model
)

# ------------------- Load documents from memory -------------------

documents = utils.get_latest_news()

# ------------------- Create vector store index -------------------

index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context
)


def get_query_index():
    """
    Return the query engine for the index
    :return:
    """
    return index


# ------------------- Simple chat engine to talk with the documents -------------------

# chat_engine = index.as_chat_engine(chat_mode="context")
# while True:
#     response = chat_engine.stream_chat(input())
#     response.print_response_stream()

# response = index.as_query_engine().query("What is the latest news about the apple?")
# print(response.response)
#
# print("Source nodes:")
# [print("node:", node.get_content()) for node in response.source_nodes]
