import os
import logging
import json
import re
from operator import itemgetter
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
            ids = [doc.metadata["doc_id"] for doc in self._raw_docs]
            self.retriever.add_documents(self._raw_docs, ids)

        print(len(list(docstore.yield_keys())))
        data = vectorstore.get(where={"doc_id": "0"}, include=["embeddings", "metadatas", "documents"])
        print(len(data["ids"]), len(data["documents"]), len(data["embeddings"]), len(data["metadatas"]), len(data["embeddings"][0]))
        print(data["metadatas"][1])
        print(data["documents"][1])
        print(data["embeddings"][1][:5])
        exit()
        #self.vectorstore.add_documents(self._raw_docs)

    def __init_multivec_retriever(self):
        separators = RecursiveCharacterTextSplitter.get_separators_for_language("python")
        separators += [".", "?", "!"]

        checking_keys = ["multi_vec_chunk_size",
                "chunk_overlap_size",
                "emb_model_name"]
        saving_root = self.__get_saving_root(checking_keys)

    def __load_data(self, data_path: str):
        self._raw_docs = []
        with open(data_path, "r") as f_src:
            for i, line in enumerate(f_src):
                content = json.loads(line)["content"]
                content = self.__clean_content(content)
                content = Document(page_content=content, metadata={"doc_id": str(i)})
                self._raw_docs.append(content)


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
