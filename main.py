import os
import json
from copy import deepcopy
from argparse import Namespace
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI

from data_handler import DataHandler
from doc_retriever import DocRetriever
from prompt import PromptGenerator
from utils import *

def run(args: Namespace):
    data_handler = DataHandler(args)
    doc_retriever = DocRetriever(data_handler, args)
    queries = read_query(args.query_path)
    prompt_gen = PromptGenerator(args)
    llm = ChatOpenAI(model_name=args.llm_model_name)
    updated_docs = deepcopy(data_handler.get_data())

    for i_query, query in enumerate(queries):
        doc_list = doc_retriever.get_relevant_documents(query)
        logging.info(f"Query: {query}\n")
        for doc in doc_list:
            doc_id = int(doc.metadata["doc_id"])
            context = updated_docs[doc_id].page_content
    
            prompt = prompt_gen.get_IO_prompt(query, context)
            response = llm.invoke(prompt)

            response_content = response.content
            logging.info(f"In document {doc_id}:\n\n")
            valid_response = None
            try:
                diff_str, updated_text = response_content.split(args.llm_response_separator)
                valid_response = True
                if diff_str[:8] == "```json\n":
                    diff_str = diff_str[8:]
                    valid_response = False
                diff_list = json.loads(diff_str.strip())

                updated_text = updated_text.strip()
                if updated_text[-3:] == "```":
                    updated_text = updated_text[:-3]
                    valid_response = False
            except Exception as ex:
                logging.warning(f"The response from LLM is incorrectly formatted! The updated context will not be saved.\n Exception:{ex}\nResponse:\n{response_content}\n\n")
                valid_response = None

            if valid_response is not None:
                diff_list = diff_list["differences"]
                updated_docs[doc_id].page_content = updated_text

                for i_diff, diff in enumerate(diff_list):
                    before, after = diff["before"], diff["after"]
                    logging.info(f"Difference {i_diff}\nBefore updating: {before}\nAfter updating: {after}\n\n")

                logging.info(f"Updated Text:\n{updated_text}\n\n")

            separator = "Conclusion: the above response is "
            if valid_response is None:
                separator += "problematic, cannot be resolved."
            elif valid_response:
                separator += "successful."
            else:
                separator += "invalid, but we have resolved."
            separator += "\n--------------------------------------------------\n\n"
            logging.info(separator)
    updated_data_path = args.result_root + "updated_scraped_data.json"
    data_handler.save_updated_data(updated_data_path, updated_docs)


if __name__ == "__main__":
    args = init_env_args_logging()
    run(args)

