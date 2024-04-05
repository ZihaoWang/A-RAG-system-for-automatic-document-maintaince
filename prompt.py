import logging
from argparse import Namespace
from langchain import PromptTemplate

from args import get_args
from utils import *

class PromptGenerator(object):
    def __init__(self, args: Namespace):
        pass

    def get_IO_prompt(self, query: str, context: str):
        prompt_template = PromptTemplate.from_template('''
        You are a helpful assistant that must try your best effort to update the information in the CONTEXT given the QUERY.
        You should ALWAYS following this guidelines:
        1. Find all the places in the CONTEXT that should be updated according to the QUERY, including antonyms and synonyms of the keywords in the QUERY.
        2. For each difference, firstly, output the first ten words and the next ten words of the position in the context that should be updated according to the QUERY. 
        Then, in a new line, output what you have updated according to the QUERY.

        QUERY: {query}

        CONTEXT: {context}
        ''')

        prompt_template.format(

if __name__ == "__main__":
    args = init_env_args_logging()
    prompt = PromptGenerator(args)
