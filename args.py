import argparse

def get_args():
    parser = argparse.ArgumentParser(description = "Awesome RAG")

    parser.add_argument("--tag", default = "", help = "adding extra tags to save_dir")
    parser.add_argument("--log_root", default = "./log/", type = str, help = "directory of logs")
    parser.add_argument("--data_root", default = "./data/", type = str, help = "directory of logs")
    parser.add_argument("--tmp_root", default = "./tmp/", type = str, help = "directory of temporary files.")

    parser.add_argument("--retriever", type = str, default = "parent_document_retriever", choices = ["parent_document_retriever", "multi_vector_retriever"], help = "Use the ParentDocument or Multi-vector LangChain Retriever.")
    parser.add_argument("--parent_chunk_size", type = int, default = -1, choices = [-1, 1000], help = "The chunk size of each document, -1 means no chunk, for ParentDocument LangChain Retriever.")
    parser.add_argument("--child_chunk_size", type = int, default = 200, choices = [100, 200, 400], help = "The size of child chunks within each document, for ParentDocument LangChain Retriever.")
    parser.add_argument("--multi_vec_chunk_size", type = int, default = 400, choices = [100, 200, 400], help = "The size of chunks within each document, for Multi-vector LangChain Retriever.")
    parser.add_argument("--chunk_overlap_size", type = int, default = 50, choices = [20, 50, 100], help = "Overlapping chars between chunks.")
    #parser.add_argument("--", type = str, choices = [], help = "")


    parser.add_argument("--emb_model_name", type = str, default = "all-MiniLM-L6-v2", choices = ["all-MiniLM-L6-v2", "gpt-3.5-turbo-16k"], help = "Use HuggingFace embedding model: all-MiniLM-L6-v2, or OpenAI embedding model: gpt-3.5-turbo-16k.")
    parser.add_argument("--search_type", default = "mmr", type = str, choices = ["mmr", "similarity"], help = "The type of search that the Retriever should perform.")
    parser.add_argument("--search_fetch_k", default = "20", type = int, choices = [10, 20, 40], help = "Number of documents passed to the search function.")
    parser.add_argument("--search_return_k", default = "5", type = int, choices = [1, 5, 10], help = "Number of documents to return after searching.")

    parser.add_argument("--idx_gpu", default = -1, type = int, help = "which cuda device to use (-1 for cpu training)")

    args = parser.parse_args()

    args.data_path = args.data_root + "scraped_data.jsonl"
    args.device = "cpu" if args.idx_gpu == -1 else f"cuda:{args.idx_gpu}"

    if args.emb_model_name in ["all-MiniLM-L6-v2"]:
        args.emb_provider = "hf"
    else:
        args.emb_provider = "openai"

    args.retriever_saving_root = args.tmp_root + f"{args.retriever}/{args.emb_provider}/{args.emb_model_name}/"

    return args

if __name__ == "__main__":
    args = get_args()
    print(args.emb_model_name)
