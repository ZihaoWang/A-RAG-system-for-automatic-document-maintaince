import argparse

def get_args():
    parser = argparse.ArgumentParser(description = "Awesome RAG")

    parser.add_argument("--log_root", default = "./log/", type = str, help = "Directory of logs")
    parser.add_argument("--data_root", default = "./data/", type = str, help = "Directory of data")
    parser.add_argument("--result_root", default = "./result/", type = str, help = "Directory of results")
    parser.add_argument("--tmp_root", default = "./tmp/", type = str, help = "Directory of temporary files.")
    parser.add_argument("--query_path", default = "./queries.txt", type = str, help = "Patg of the query file, each line is a query.")
    parser.add_argument("--llm_response_separator", default = "####--separator--####", type = str, help = "Separator between differences and updated text..")

    parser.add_argument("--retriever", type = str, default = "threshold_retriever", choices = ["threshold_retriever", "parent_document_retriever", "multi_vector_retriever"], help = "Use the Threshold, ParentDocument or Multi-vector LangChain Retriever.")
    parser.add_argument("--parent_chunk_size", type = int, default = -1, choices = [-1, 1000], help = "The chunk size of each document, -1 means no chunk, for ParentDocument LangChain Retriever.")
    parser.add_argument("--child_chunk_size", type = int, default = 400, choices = [100, 200, 400], help = "The size of child chunks within each document, for ParentDocument LangChain Retriever.")
    parser.add_argument("--chunk_size", type = int, default = 400, choices = [100, 200, 400],ehelp = "The size of chunks within each document, for Threshold and Multi-vector LangChain Retriever.")
    parser.add_argument("--chunk_overlap_size", type = int, default = 50, choices = [20, 50, 100], help = "Overlapping chars between chunks.")
    #parser.add_argument("--", type = str, choices = [], help = "")


    parser.add_argument("--llm_model_name", type = str, default = "gpt-4", choices = ["gpt-4", "gpt-3.5-turbo"], help = "Use OpenAI chat model: gpt-3.5-turbo, gpt-4.")
    parser.add_argument("--emb_model_name", type = str, default = "text-embedding-3-small", choices = ["all-MiniLM-L6-v2", "text-embedding-3-small", "text-embedding-3-large"], help = "Use HuggingFace embedding model: all-MiniLM-L6-v2, or OpenAI embedding model: text-embedding-3-small, text-embedding-3-large.")
    parser.add_argument("--search_type", default = "mmr", type = str, choices = ["mmr", "similarity", "similarity_score_threshold"], help = "The type of search that the Retriever should perform.")
    parser.add_argument("--score_threshold", default = "0.5", type = float, help = "Minimum relevance threshold for similarity_score_threshold")
    parser.add_argument("--search_fetch_k", default = "20", type = int, choices = [5, 10, 20, 40], help = "Number of documents passed to the search function.")
    parser.add_argument("--search_return_k", default = "3", type = int, choices = [1, 3, 5, 10, 20, 40], help = "Number of documents to return after searching.")

    parser.add_argument("--idx_gpu", default = -1, type = int, help = "which cuda device to use (-1 for cpu training)")
    parser.add_argument("--openai_user_tier", default = 1, type = int, help = "OpenAI user tier")

    args = parser.parse_args()

    args.data_path = args.data_root + "scraped_data.jsonl"
    args.device = "cpu" if args.idx_gpu == -1 else f"cuda:{args.idx_gpu}"

    if args.retriever == "threshold_retriever":
        args.search_type = "similarity_score_threshold"
        args.search_return_k = 40

    if args.emb_model_name in ["all-MiniLM-L6-v2"]:
        args.emb_provider = "hugging_face"
    else:
        args.emb_provider = "openai"

    args.retriever_saving_root = args.tmp_root + f"{args.retriever}/{args.emb_provider}/{args.emb_model_name}/"

    return args

if __name__ == "__main__":
    args = get_args()
    print(args.llm_model_name)
