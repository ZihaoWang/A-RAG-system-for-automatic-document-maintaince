# RAG-Query-Processor
A simple RAG framework using LangChain.

# The Task

Information changes quickly, and it's hard to keep knowledge current. Some team members are responsible for maintaining accurate documentation, but it's tough for them to identify all the places needing updates, especially with each new product change. So, we're creating a RAG solution that can automatically update documentation based on natural language queries.

Imagine there's a new document update. Now, archiving isn't possible anymore; instead, queries can only be deleted. With our solution, a team member can simply enter the instruction *We removed the ability to archive queries, and instead added the ability to completely delete them. Update all relevant knowledge.* Then, the RAG Solution will automatically update all the relevant information in the documentation.

# Installation and Running
1. Build the Docker image:

    * docker build -t rag img --rm .

    * docker run -it --name rag app --rm rag img

2. Change the OpenAI and LangChain keys in the secrets.txt

3. Check the existing queries in the queries.txt and see if you want to add more. Each line is a query.

4. Run the framework, such as:

* python main.py --emb model name all-MiniLM-L6-v2 python main.py --emb model name text-embedding-3-small --retriever parent document retriever --search fetch k 40

5. The updated documents will be stored under the result/, and the log file rag.log is under the log/.

6. For other parameters in the configuration, check the config.py.

# Important Arguments In The Configuration

I list some of the arguments that are important to the final performance. More details can be viewed in the args.py.

* --retriever: Can be Threshold Retriever or ParentDocument Retriever. The Threshold Retriever returns all documents whose similarities with the query are above a certain threshold, and the ParentDocument Retriever always returns Top-k similar documents.

* --search type: For ParentDocument Retriever, it can retrieve documents with ”mmr” using reranking or ”similarity” computing only the cosine similarity. For Threshold Retriever, it can only be ”similarity score threshold” that retrieves documents above a similarity threshold set in the argument --score threshold.

* --chunk size: The size of chunks within each document. A smaller chunk size enables more subtle retrieval.

* --search fetch k: Number of documents sent to the re-ranking process of the MMR. A larger number increases the diversity of retrieval results.

* --emb model name: Embedding models for documents. Can be OpenAI (”text-embedding3-small”, ”text-embedding-3-large”) or HuggingFace (”all-MiniLM-L6-v2”) embedding models.

* --llm model name: OpenAI chat model for text generation, can be gpt-3.5-turbo or gpt-4.

# Implementation Details

Given the user’s natural language queries and the provided documents, the project is implemented with a (Retrieval Augmented Generation) RAG framework. Specifically, the initial framework processes each query with the following steps:

1. The framework retrieves embeddings for each provided document from an LLM.

2. Given a query, the framework retrieves the top-K relevant documents from all.

3. The framework considers each retrieved document (each line in the original JSON file) as the context of the query and formulates a prompt by filling the query and context into a pre-defined template.

4. The framework sends the prompt to an LLM, and the response from the LLM consists of the updated document and differences.

5. The original documents and their updates are saved into the JSON file.

Based on this initial framework, the following optimizations have been performed:

* To save querying to LLMs, the embeddings and corresponding configurations are stored locally. Each time, the framework decides whether to reuse the local embeddings by checking the current configurations with the stored ones. If both configurations are the same, the framework can reuse the local embeddings and does not need to query from LLMs.

* The framework automatically generates a log, which consists of the current configuration, queries, responses from LLMs, updated documents, and other messages.

* To prevent exceeding the requests per minute (RPM) from OpenAI, the framework can automatically control the frequency of requests.

* The framework supports OpenAI and HuggingFace embeddings, and GPUs can be used for HuggingFace embeddings.

# Simple Experiments

For experiments, I use three different queries:

1. *We removed the ability to archive queries and instead added the ability to completely delete them. Update all relevant documents.*

2. *We have increased the default limit of 250,000 datapoints to 400,000. Update all relevant documents.*

3. *We do not support TrinoSQL anymore. Update all relevant documents.*

I compared different configurations of the framework listed in the previous section. All the logs of different configurations can be viewed under the directory log/. The following configuration is considered as the baseline: retriever = threshold retriever, search type = similarity score threshold, chunk size = 400, search fetch k = 20, emb model name = text-embedding-3-small, llm model name = gpt-4.

After manually analyzing the framework’s performance under different configurations, here are several interesting results:

* For the Query 1, most models successfully update the ”archive” into the ”delete.” Moreover, there are two highlights:
    1. The framework (such as the baseline configuration) can also recognize the word ”unarchive” and update the information correctly.
    2. Some codes can also be updated. For example:

	– Before: POST/api/v1/query/query id/archive

	– After: POST/api/v1/query/query id/delete

	– Before: defarchive query(self, query id : int)− > bool : more codes.

	– After: Method to archive queries has been removed.

* The Query 2 is difficult for document retrieval, as the keyword ”250,000” is a number that is difficult for embedding models to recognize. However, as I decreased the chunk size to 200, the framework can retrieve the correct document with this information.

* For the Query 3, the framework tends to replace all the ”TrinoSQL” with the ”DuneSQL.” Although LLMs find relevant information about ”DuneSQL” in the context, I do not ask the framework to do so, and thus, it is regarded as a hallucination.

* For the chat model, GPT-4 always performs better than GPT3.5, as GPT3.5 sometimes just copy-and-paste words from prompts to their updates.

* Surprisingly, when using HuggingFace embeddings with the OpenAI chat model, the results are still as good as the baseline. This result suggests that the retrieval and text generation can be two independent steps by using different LLMs, which provides us more space for optimization.
