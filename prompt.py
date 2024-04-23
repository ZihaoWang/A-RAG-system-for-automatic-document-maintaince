import logging
from argparse import Namespace
from langchain_core.prompts import PromptTemplate

from args import get_args
from utils import *

class PromptGenerator(object):
    def __init__(self, args: Namespace):
        self.args = args

    def get_IO_prompt(self, query: str, context: str) -> str:
        prompt_template = PromptTemplate.from_template('''
        You are a helpful assistant that must try your best effort to update the information in the CONTEXT given the QUERY.
        You should ALWAYS following this guidelines:
        1. Find all the places in the CONTEXT that should be updated according to the QUERY, including antonyms and synonyms of the keywords in the QUERY.
        2. Your response should be two parts. The first part is a JSON file which contains a list of all the differences as the value of the key "differences". Then, output "{separator}" in a new line. Next, output the updated CONTEXT given the QUERY as the second part on another new line.
        3. For each difference in the JSON file, firstly, save the first ten words and the next ten words of the position in the context that should be updated according to the QUERY as the value of the key "before". 
        Then, save what you have updated given the QUERY as the value of the key "after".

        QUERY: {query}

        CONTEXT: {context}
        ''')

        prompt = prompt_template.format(query=query, context=context, separator=self.args.llm_response_separator)
        return prompt
        

if __name__ == "__main__":
    args = init_env_args_logging()
    prompt_gen = PromptGenerator(args)
    prompt = prompt_gen.get_IO_prompt("Here is query.", "Here is context.")
    print(prompt)
    print(type(prompt))
