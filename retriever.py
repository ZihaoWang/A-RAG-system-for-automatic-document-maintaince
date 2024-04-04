import os
import logging
import json
import re
import uuid
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
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from args import get_args
from utils import *

class DataPreProcessor(object):
    def __init__(self, args: Namespace):
        self.args = args

        self.__load_data(args.data_path)

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
            checking_keys = ["multi_vec_chunk_size",
                    "chunk_overlap_size",
                    "emb_model_name"]
            self.retriever = load_model(checking_keys)
            if self.retriever is None:
                self.__init_multivec_retriever(checking_keys)
        else:
            checking_keys = ["parent_chunk_size",
                    "child_chunk_size",
                    "chunk_overlap_size",
                    "emb_model_name"]
            saved_emb = load_model(checking_keys)
            if saved_emb is None:
                self.__init_parent_document_retriever(checking_keys)
            else:
                store = InMemoryByteStore()
                store.mset(saved_emb)
                self.retriever = ParentDocumentRetriever(vectorstore=vectorstore,
                        docstore=parent_storage,
                        parent_splitter=parent_splitter,
                        child_splitter=child_splitter)

    def get_retriever(self) -> BaseRetriever:
        return self.retriever

    def __init_parent_document_retriever(self, checking_keys):
        logging.info("Initialize parent document retriever.")

        separators = RecursiveCharacterTextSplitter.get_separators_for_language("python")
        separators += [".", "?", "!"]
        if self.args.parent_chunk_size == -1:
            parent_splitter = RecursiveCharacterTextSplitter(separators=separators,
                    keep_separator = True)
        else:
            parent_splitter = RecursiveCharacterTextSplitter(separators=separators,
                    keep_separator = True,
                    chunk_size=self.args.parent_chunk_size,
                    chunk_overlap=self.args.chunk_overlap_size)
        child_splitter = RecursiveCharacterTextSplitter(separators=separators,
                keep_separator = True,
                chunk_size=self.args.child_chunk_size,
                chunk_overlap=self.args.chunk_overlap_size)
        vectorstore = Chroma(embedding_function=self.emb_model, persist_directory = self.args.child_saving_root)

        parent_storage = create_kv_docstore(LocalFileStore(self.args.parent_saving_root))
        self.retriever = ParentDocumentRetriever(vectorstore=vectorstore,
                docstore=parent_storage,
                parent_splitter=parent_splitter,
                child_splitter=child_splitter)

        print(4)
        print(len(self._raw_docs))
        self.retriever.add_documents(self._raw_docs)

        print(5)
        save_model(self.retriever, checking_keys)

    def __init_multivec_retriever(self, checking_keys):
        separators = RecursiveCharacterTextSplitter.get_separators_for_language("python")
        separators += [".", "?", "!"]

        save_model(self.retriever, checking_keys)

    def __load_data(self, data_path: str):
        self._raw_docs = []
        with open(data_path, "r") as f_src:
            for i, line in enumerate(f_src):
                content = json.loads(line)["content"]
                content = self.__clean_content(content)
                content = Document(page_content=content)
                self._raw_docs.append(content)
                if i > 10:
                    break


    def __clean_content(self, content):
        # remove urls: https://gist.github.com/gruber/8891611
        # url_regex = r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"
        # content = re.sub(url_regex, "", content)

        return content

if __name__ == "__main__":
    import os
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    with open("secrets.txt", "r") as f_src:
        for line in f_src:
            key, secret = line.strip().split("=")
            os.environ[key] = secret

    args = get_args()
    log_path = os.path.join(args.log_root, "rag.log")
    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_path
    )
    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info(f"Saving the log at: {log_path}")

    preprocessor = DataPreProcessor(args)

    retriever = preprocessor.get_retriever()
    retrieved_docs = retriever.get_relevant_documents("blockchain data")
    print(retrieved_docs)
