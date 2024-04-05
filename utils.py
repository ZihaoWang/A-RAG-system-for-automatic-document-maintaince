import os
import logging

from args import get_args

def init_env_args_logging():
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

    return args


