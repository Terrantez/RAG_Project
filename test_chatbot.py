import os

import pytest

from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.evaluation import DatasetGenerator

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric, SummarizationMetric

from llama_index_base import get_query_index
from utils import get_latest_news

os.environ["OPENAI_API_KEY"] = "sk-key"

llm = OpenAI(model="gpt-4", temperature=0.0)
service_context = ServiceContext.from_defaults(llm=llm)
data_generator = DatasetGenerator.from_documents(get_latest_news(), service_context=service_context)

eval_questions = data_generator.generate_questions_from_nodes(5)

engine = get_query_index().as_query_engine()

responses = [engine.query(question) for question in eval_questions]

test_cases = [
    LLMTestCase(
        input=eval_questions[i],
        actual_output=responses[i].response,
        context=[node.get_content() for node in responses[i].source_nodes],
        retrieval_context=[node.get_content() for node in responses[i].source_nodes])
    for i in range(len(eval_questions))
]
dataset = EvaluationDataset(test_cases=test_cases)


def test_chatbot():
    evaluate(test_cases, [
        SummarizationMetric(threshold=0.5),
        FaithfulnessMetric(threshold=0.5, include_reason=True),
        AnswerRelevancyMetric(threshold=0.5, include_reason=True),
        ContextualRelevancyMetric(threshold=0.5, include_reason=True),
    ])
