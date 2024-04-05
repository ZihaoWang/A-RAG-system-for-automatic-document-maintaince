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
    args = init_env_args_logging()
    data_loader = DataLoader(args)
    doc_emb = DocEncoder(data_loader, args)

    retriever = doc_emb.get_retriever()
    retrieved_docs = retriever.get_relevant_documents("blockchain data")
    print(retrieved_docs)

