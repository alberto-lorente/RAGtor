from typing import List, Tuple, Dict

import ollama
from langchain_core.documents import Document
from .config import SUMMARY_OLLAMA_MODEL, QUERY_OLLAMA_MODEL, EMBEDDINGS_OLLAMA_MODEL, VLM_OLLAMA_MODEL



def ollama_generate_text(model: str, 
                        formatted_prompt: str) -> Tuple[str, List[Dict]]: # the response and the list of chat messages

    if model == SUMMARY_OLLAMA_MODEL:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Your task is to generate a summary of the given text."
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ]

    elif model == QUERY_OLLAMA_MODEL:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Your task is to answer the users question/s."
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ]

    else:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ]


    response = ollama.chat(model=model,
                    messages=messages)

    return response.message.content, messages



def format_context_string(responses: List[Document]) -> str:

    key_points_str = ""
    for i, response in enumerate(responses):
        content = response.page_content
        source = response.metadata["chunk_source"]
        key_points_str += f"\n{str(i+1)}. Source: {source}\nInformation: {content}\n"

    return key_points_str
