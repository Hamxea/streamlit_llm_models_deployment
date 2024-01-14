import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import numpy as np
import os


def text_embedding(text, model_name='sentence-transformers/all-MiniLM-L6-v2')-> None:
    model = SentenceTransformer(model_name)
    return model.encode(text)


def generate_context(query):
    vector = text_embedding(query).tolist()

    result = collection.query(
        query_embeddings=vector,
        n_results=15,
        include=['documents']
    )

    res = "\n".join(str(item) for item in result['documents'][0])
    return res


def chat_completion(system_prompt, user_prompt,length=1000):
    final_prompt=f"""<s>[INST]<<SYS>>
    {system_prompt}
    <</SYS>>

    {user_prompt} [/INST]"""
    return client.text_generation(prompt=final_prompt,max_new_tokens = length).strip()