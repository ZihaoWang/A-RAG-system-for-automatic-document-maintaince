import os

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tiktoken
import utils

def get_embeddings():
    embd = OpenAIEmbeddings()
    #query_result = embd.embed_query(question)
    #db = FAISS.from_documents()

def run(data_path):
    preprocessor = utils.DataPreProcessor(data_path)
    splitted_data = preprocessor.splitted_data
    print(len(docs))


if __name__ == "__main__":
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    with open("secrets.txt", "r") as f_src:
        for line in f_src:
            key, secret = line.strip().split("=")
            os.environ[key] = secret

    data_path = "./data/scraped_data.jsonl"
    run(data_path)


