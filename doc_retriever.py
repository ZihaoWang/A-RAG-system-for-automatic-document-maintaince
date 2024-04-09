import logging
from typing import Any
from collections.abc import Iterable
from argparse import Namespace
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore, LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

from args import get_args
from data_handler import DataHandler
from utils import *

class DocRetriever(object):
    def __init__(self, data_handler: DataHandler, args: Namespace):
        self.args = args
        self._docs = data_handler.get_data()

        if args.emb_provider == "hugging_face":
            model_kwargs = {"device": args.device}
            self.emb_model = HuggingFaceEmbeddings(model_name=args.emb_model_name,
                    model_kwargs = model_kwargs)
        elif args.emb_provider == "openai":
            model_kwargs = {}
            self.emb_model = OpenAIEmbeddings(model=args.emb_model_name,
                    model_kwargs = model_kwargs)
        else:
            raise(f"Unsupport embedding provider: {args.emb_provider}")


        if args.retriever == "multi_vector_retriever":
            self.__init_multivec_retriever()
        elif args.retriever == "threshold_retriever":
            self.__init_threshold_retriever()
        else:
            self.__init_parent_document_retriever()

    def get_relevant_documents(self, query: str) -> Iterable[Document]:
        cand_docs = self.retriever.get_relevant_documents(query)
        final_ids = set()
        final_docs = []
        for cand_doc in cand_docs:
            doc_id = int(cand_doc.metadata["doc_id"])
            if doc_id not in final_ids:
                final_docs.append(self._docs[doc_id])
                final_ids.add(doc_id)
            else:
                continue

        return final_docs

    def get_vectorstore(self) -> VectorStore:
        return self.vectorstore

    def __init_parent_document_retriever(self):
        logging.info("Initialize parent document retriever.")

        separators = RecursiveCharacterTextSplitter.get_separators_for_language("python")
        separators += [".", "?", "!"]
        if self.args.parent_chunk_size == -1:
            parent_splitter = None
        else:
            parent_splitter = RecursiveCharacterTextSplitter(separators=separators,
                    keep_separator = True,
                    chunk_size=self.args.parent_chunk_size,
                    chunk_overlap=self.args.chunk_overlap_size)
        child_splitter = RecursiveCharacterTextSplitter(separators=separators,
                keep_separator = True,
                chunk_size=self.args.child_chunk_size,
                chunk_overlap=self.args.chunk_overlap_size)

        checking_keys = ["parent_chunk_size",
                "child_chunk_size",
                "chunk_overlap_size",
                "emb_model_name"]
        saving_root, new_root = get_saving_root(checking_keys, self.args)
        self.vectorstore = Chroma(embedding_function=self.emb_model,
                collection_metadata={"hnsw:space": "cosine"},
                persist_directory=saving_root + "vectorstore/")
        self.docstore = create_kv_docstore(LocalFileStore(saving_root + "docstore/"))

        search_kwargs = {"k": self.args.search_return_k}
        if self.args.search_type == "mmr":
            search_kwargs["fetch_k"] = self.args.search_fetch_k
        self.retriever = ParentDocumentRetriever(vectorstore=self.vectorstore,
                docstore=self.docstore,
                search_type=self.args.search_type,
                search_kwargs=search_kwargs,
                parent_splitter=parent_splitter,
                child_splitter=child_splitter)

        if new_root:
            logging.info("Loading documents into the new vectorstore.")
            ids = [doc.metadata["doc_id"] for doc in self._docs]
            batch_openai_request(self.retriever.add_documents,
                    (self._docs, ids),
                    len(self._docs),
                    False,
                    self.args)

    def __init_threshold_retriever(self):
        logging.info("Initialize threshold retriever.")

        separators = RecursiveCharacterTextSplitter.get_separators_for_language("python")
        separators += [".", "?", "!"]
        splitter = RecursiveCharacterTextSplitter(separators=separators,
                keep_separator = True,
                chunk_size=self.args.chunk_size,
                chunk_overlap=self.args.chunk_overlap_size)
        splitted_docs = splitter.split_documents(self._docs)

        checking_keys = ["child_chunk_size",
                "chunk_overlap_size",
                "emb_model_name"]
        saving_root, new_root = get_saving_root(checking_keys, self.args)
        self.vectorstore = Chroma(embedding_function=self.emb_model,
                collection_metadata={"hnsw:space": "cosine"},
                persist_directory=saving_root + "vectorstore/")

        if new_root:
            logging.info("Loading documents into the new vectorstore.")
            batch_openai_request(self.vectorstore.add_documents,
                    (splitted_docs,),
                    len(splitted_docs),
                    False,
                    self.args)

        self.retriever = self.vectorstore.as_retriever(search_type=self.args.search_type,
                search_kwargs={"score_threshold": self.args.score_threshold,
                    "k": self.args.search_return_k})


    def __init_multivec_retriever(self):
        separators = RecursiveCharacterTextSplitter.get_separators_for_language("python")
        separators += [".", "?", "!"]

        checking_keys = ["multi_vec_chunk_size",
                "chunk_overlap_size",
                "emb_model_name"]
        saving_root, new_root = get_saving_root(checking_keys, self.args)

if __name__ == "__main__":
    args = init_env_args_logging()
    data_handler = DataHandler(args)
    doc_retriever = DocRetriever(data_handler, args)

    vectorstore = doc_retriever.get_vectorstore()
    data = vectorstore.get(where={"doc_id": "0"}, include=["embeddings", "metadatas", "documents"])
    print(len(data["ids"]), len(data["documents"]), len(data["embeddings"]), len(data["metadatas"]), len(data["embeddings"][0]))
    idx_child_segment = 1
    print(data["metadatas"][idx_child_segment])
    print(data["documents"][idx_child_segment])
    print(data["embeddings"][idx_child_segment][:5])

    retrieved_docs = doc_retriever.get_relevant_documents("blockchain data")
    print(len(retrieved_docs))
    if len(retrieved_docs) > 0:
        print(retrieved_docs[0])
        print(retrieved_docs[-1])

