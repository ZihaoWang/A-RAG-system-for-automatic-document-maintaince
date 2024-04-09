import os
import logging
import time
import pickle
from typing import Callable, TypeVar, Any, Union
from argparse import Namespace
from collections.abc import Iterable
from openai import RateLimitError

from args import get_args

T = TypeVar("T")
U = TypeVar("U")

def batch_openai_request(fn: Callable, fn_input: Iterable[U], length: int, has_return: bool, args: Namespace) -> Union[Iterable[T], None]:
    max_retries = 6
    delay_increment = 60

    batch_size = min(30 if args.openai_user_tier < 4 else 80, length)
    logging.info(f"In {fn.__name__}, batch size = {batch_size}.")

    for i_batch in range(0, length, batch_size):
        batch = []
        for each_input in fn_input:
            if isinstance(each_input, Iterable):
                batch.append(each_input[i_batch : i_batch + batch_size])
            else:
                batch.append(each_input)
        retries = 0
        results = [] if has_return else None
        while retries <= max_retries:
            try:
                if has_return:
                    results.append(fn(*batch))
                else:
                    fn(*batch)
                break

            except RateLimitError as rate_limit_error:
                delay = (retries + 1) * delay_increment
                logging.warning(f"{rate_limit_error}. Retrying in {delay} seconds ...")
                time.sleep(delay)
                retries += 1

                if retries > max_retries:
                    logging.error(f"Max retries reached for batch {i_batch}.")
                    raise

            except Exception as ex:
                logging.error(f"Error happens when batch processing {fn.__name__}: {ex}")
                raise

    return results

def init_env_args_logging() -> Namespace:
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
        filename=log_path,
        filemode="w"
    )
    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info(f"Saving the log at: {log_path}\nParameters:\n--------------------------------\n")

    summary_dict = vars(args)
    for k, v in summary_dict.items():
        logging.info(f"{k} = {v}\n")

    logging.info("End of parameters.\n------------------------------------\n")

    for root in ["tmp_root", "log_root", "result_root"]:
        os.makedirs(summary_dict[root], exist_ok = True)

    return args

def read_query(query_path: str) -> Iterable[str]:
    queries = []
    with open(query_path, "r") as f_src:
        for line in f_src:
            line = line.strip()
            queries.append(line)

    return queries

def get_saving_root(checking_keys: Iterable[str], args: Namespace) -> (str, bool):
    os.makedirs(args.retriever_saving_root, exist_ok = True)
    all_saving_dirs = os.listdir(args.retriever_saving_root)
    saving_root = None
    new_root = None

    for d in all_saving_dirs:
        saving_root = args.retriever_saving_root + d + "/"
        with open(saving_root + "config.pickle", "rb") as f_src:
            saved_config = pickle.load(f_src)
        if same_config(saved_config, checking_keys, args):
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
            v = getattr(args, k)
            logging.info(f"\t\t{k}: {v}")
            saved_config[k] = v
    
        new_root = True
        with open(saving_root + "config.pickle", "wb") as f_dst:
            pickle.dump(saved_config, f_dst, pickle.HIGHEST_PROTOCOL)

    return saving_root, new_root

def same_config(saved_config: dict[str, Any], checking_keys: dict[str, Any], args: Namespace) -> bool:
    for k in checking_keys:
        running_arg = getattr(args, k)
        saved_arg = saved_config[k]
        if saved_arg != running_arg:
            logging.info(f"Arg in the running config is different as the arg in the saved config:")
            logging.info(f"In the running config, {k}: {running_arg}")
            logging.info(f"In the saved config, {k}: {saved_arg}")
            return False 

    return True


