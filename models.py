"""
Functions to load local LLMs using LLamaCPP
There are two different loaders, one from the LangChain package and another from the LLamaIndex package
Functionality is the same, but they are wrapped differently and are not compatible with each other
"""
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain_community.llms import LlamaCpp as LangCPP


def get_llama_index_llm_gpu():
    return LlamaCPP(
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path="C:/Users/DPWar/Desktop/Univ/Doutoramento/PLEI/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        temperature=0.1,
        max_new_tokens=1000,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": -1, "f16_kv": True},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True
    )


def get_llama_index_cpu():
    return LlamaCPP(
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path="C:/Users/DPWar/Desktop/Univ/Doutoramento/PLEI/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        temperature=0.1,
        max_new_tokens=1000,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True
    )


def get_langchain_llm_gpu():
    return LangCPP(
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path="C:/Users/DPWar/Desktop/Univ/Doutoramento/PLEI/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        temperature=0.1,
        max_tokens=256,
        n_gpu_layers=-1,
        n_ctx=4096,
        f16_kv=True,
        verbose=True
    )


def get_langchain_llm_cpu():
    return LangCPP(
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path="C:/Users/DPWar/Desktop/Univ/Doutoramento/PLEI/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        temperature=0.1,
        max_tokens=256,
        n_gpu_layers=0,
        n_ctx=4096,
        verbose=True
    )
