import os
import logging
from collections.abc import Iterable
from argparse import Namespace
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore, LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

from args import get_args
from data_loader import DataLoader
from utils import *

class DocEncoder(object):
    def __init__(self, data_loader: DataLoader, args: Namespace):
        self.args = args
        self._docs = data_loader.get_data()

        model_kwargs = {"device": args.device}
        if args.emb_provider == "hf":
            self.emb_model = HuggingFaceEmbeddings(model_name=args.emb_model_name,
                    model_kwargs = model_kwargs)
        elif args.emb_provider == "openai":
            self.emb_model = OpenAIEmbeddings(model_name=args.emb_model_name,
                    model_kwargs = model_kwargs)
        else:
            raise(f"Unsupport embedding provider: {args.emb_provider}")


        if args.retriever == "multi_vector_retriever":
            self.__init_multivec_retriever()
        else:
            self.__init_parent_document_retriever()

    def __get_saving_root(self, checking_keys: Iterable[str]):
        os.makedirs(args.retriever_saving_root, exist_ok = True)
        all_saving_dirs = os.listdir(args.retriever_saving_root)
        saving_root = None
        new_root = None

        for d in all_saving_dirs:
            saving_root = args.retriever_saving_root + d + "/"
            with open(saving_root + "config.pickle", "rb") as f_src:
                saved_config = pickle.load(f_src)
            if self.__same_config(saved_config, checking_keys):
                logging.info("Using the existing vectorstore with the config:")
                for k in checking_keys:
                    logging.info(f"\t\t{k}: {saved_config[k]}")
                new_root = False
                break
        else:
            saving_root = args.retriever_saving_root + str(len(all_saving_dirs)) + "/"
            os.makedirs(saving_root, exist_ok = True)

            logging.info(f"Creating a vectorstore at {saving_root} with a new config:")
            saved_config = {}
            for k in checking_keys:
                v = getattr(self.args, k)
                logging.info(f"\t\t{k}: {v}")
                saved_config[k] = v
        
            new_root = True
            with open(saving_root + "config.pickle", "wb") as f_dst:
                pickle.dump(saved_config, f_dst, pickle.HIGHEST_PROTOCOL)
    
        return saving_root, new_root

    def __same_config(self, saved_config, checking_keys):
        if len(checking_keys) != len(saved_config):
            #saved_keys = list(saved_config.keys())
            #logging.warn(f"The running config is different as the saved config:")
            #logging.warn(f"Keys in the running config: {checking_keys}")
            #logging.warn(f"Keys in the saved config: {saved_keys}")
            return False 
    
        for k in checking_keys:
            running_arg = getattr(self.args, k)
            saved_arg = saved_config[k]
            if saved_arg != running_arg:
                #logging.warn(f"Arg in the running config is different as the arg in the saved config:")
                #logging.warn(f"In the running config, {k}: {running_arg}")
                #logging.warn(f"In the saved config, {k}: {saved_arg}")
                return False 
    
        return True

    def get_retriever(self) -> BaseRetriever:
        return self.retriever

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
        saving_root, new_root = self.__get_saving_root(checking_keys)
        vectorstore = Chroma(embedding_function=self.emb_model, persist_directory=saving_root + "vectorstore/")
        docstore = create_kv_docstore(LocalFileStore(saving_root + "docstore/"))

        self.retriever = ParentDocumentRetriever(vectorstore=vectorstore,
                docstore=docstore,
                parent_splitter=parent_splitter,
                child_splitter=child_splitter)

        if new_root:
            logging.info("Loading documents into the new vectorstore.")
            ids = [doc.metadata["doc_id"] for doc in self._docs]
            self.retriever.add_documents(self._docs, ids)

        print(len(list(docstore.yield_keys())))
        data = vectorstore.get(where={"doc_id": "0"}, include=["embeddings", "metadatas", "documents"])
        print(len(data["ids"]), len(data["documents"]), len(data["embeddings"]), len(data["metadatas"]), len(data["embeddings"][0]))
        print(data["metadatas"][1])
        print(data["documents"][1])
        print(data["embeddings"][1][:5])
        exit()
        #self.vectorstore.add_documents(self._docs)

    def __init_multivec_retriever(self):
        separators = RecursiveCharacterTextSplitter.get_separators_for_language("python")
        separators += [".", "?", "!"]

        checking_keys = ["multi_vec_chunk_size",
                "chunk_overlap_size",
                "emb_model_name"]
        saving_root = self.__get_saving_root(checking_keys)

if __name__ == "__main__":
    args = init_env_args_logging()
    data_loader = DataLoader(args)
    doc_emb = DocEncoder(data_loader, args)

    retriever = doc_emb.get_retriever()
    retrieved_docs = retriever.get_relevant_documents("blockchain data")
    print(retrieved_docs)
