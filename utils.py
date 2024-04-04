import os
import logging
import pickle

def save_model(self, obj, save_path: str, checking_keys: Iterable[str]):
    logging.info(f"Saving model at {save_path} with the following config:")
    saved_config = {}
    for k in checking_keys:
        v = getattr(self.args, k)
        logging.info(f"\t\t{k}: {v}")
        saved_config[k] = v

    os.makedirs("/".join(save_path.split("/")[:-1]))
    with open(save_path, "wb") as f_dst:
        pickle.dump((obj, saved_config), f_dst, pickle.HIGHEST_PROTOCOL)

def load_model(self, load_path: str, checking_keys: Iterable[str]):
    if not os.path.exists(load_path):
        logging.info(f"Retriever cache does not exist at {load_path}")
        return None

    logging.info(f"Loading model at {save_path}.")
    with open(load_path, "rb") as f_src:
        obj, saved_config = pickle.load(f_src)

    if len(checking_keys) != len(saved_config):
        saved_keys = list(saved_config.keys())
        logging.warn(f"The running config is different as the saved config:")
        logging.warn(f"Keys in the running config: {checking_keys}")
        logging.warn(f"Keys in the saved config: {saved_keys}")
        return None

    for k in checking_keys:
        running_arg = getattr(self.args, k)
        saved_arg = saved_config[k]
        if saved_arg != running_arg:
            logging.warn(f"Arg in the running config is different as the arg in the saved config:")
            logging.warn(f"In the running config, {k}: {running_arg}")
            logging.warn(f"In the saved config, {k}: {saved_arg}")
            return None

    return obj

