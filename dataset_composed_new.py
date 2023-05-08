"""
Construct a randomly sampled subset of the composed dataset.
"""
import os
from pathlib import Path
import argparse
import logging
import random
from sklearn.model_selection import train_test_split

from compose import COMPOSE_OPERATORS, compose_and, compose_or
from utils import load_from_file, save_to_file, deserialize_props_str


def load_base_dataset(base_fpath, logger):
    base_dataset = load_from_file(base_fpath)
    base_data, base_meta = [], []
    base_ltls = set()
    for pattern_type, props_str, utt, ltl in base_dataset:
        props = deserialize_props_str(props_str)
        base_data.append((utt, ltl))
        base_meta.append((pattern_type, props))
        base_ltls.add(ltl)
    logger.info(f"Base dataset nsamples: {len(base_data)}, {len(base_meta)}")
    logger.info(f"Base dataset nformulas: {len(base_ltls)}")
    return base_data, base_meta


def sample_composed_dataset(compose_operators, base_fpath, nsamples, seed, composed_dpath, logger):
    base_data, base_meta = load_base_dataset(base_fpath, logger)
    based_dataset = {"data": base_data, "meta": base_meta}
    base_fpath = os.path.join(composed_dpath, f"base_{Path(base_fpath).stem}.pkl")
    save_to_file(based_dataset, base_fpath)

    composed_data, composed_meta = [], []
    random.seed(seed)
    for sample_id in range(nsamples):
        logger.info(f"Composing sample {sample_id}")

        compose_operator = random.sample(compose_operators, 1)[0]
        data, meta = zip(*random.sample(list(zip(base_data, base_meta)), 2))
        utts_base, ltls_base = zip(*data)

        if compose_operator == "and":
            utt_composed, ltl_composed = compose_and(utts_base, ltls_base)
        elif compose_operator == "or":
            utt_composed, ltl_composed = compose_or(utts_base, ltls_base)
        else:
            raise ValueError(f"ERROR: operator not supported: {compose_operator}.")

        composed_data.append((utt_composed, ltl_composed))
        composed_meta.append(meta)

    composed_dataset = {"data": composed_data, "meta": composed_meta}
    composed_fpath = os.path.join(composed_dpath, f"composed_nsamples{nsamples}_seed{seed}_{Path(base_fpath).stem}.pkl")
    save_to_file(composed_dataset, composed_fpath)
    return base_fpath, composed_fpath


def construct_split_datasets(base_fpath, composed_fpath, split_ratio, seed, logger):
    """
    Construct train and test split.
    """
    comosed_dataset = load_from_file(composed_fpath)
    composed_data, composed_meta = comosed_dataset["data"], comosed_dataset["meta"]

    train_dataset, test_dataset = train_test_split(list(zip(composed_data, composed_meta)), test_size=split_ratio, random_state=seed)
    train_data, train_meta = zip(*train_dataset)
    test_data, test_meta = zip(*test_dataset)

    base_dataset = load_from_file(base_fpath)
    base_data, base_meta = base_dataset["data"], base_dataset["meta"]
    train_data = tuple(base_data) + train_data
    train_meta = tuple(base_meta) + train_meta

    split_dataset = {"train_iter": train_data, "train_meta": train_meta,
                     "valid_iter": test_data, "valid_meta": test_meta,
                     "info": {"split_ratio": split_ratio, "seed": seed}}
    split_fpath = os.path.join(os.path.dirname(composed_fpath), f"split_raito{split_ratio}_seed{seed}_{os.path.basename(composed_fpath)}")
    save_to_file(split_dataset, split_fpath)

    logger.info(f"Training set size: {len(train_data)}, {len(train_meta)}")
    logger.info(f"Testin set size: {len(test_data)}, {len(test_meta)}")
    return split_fpath


def construct_holdout_datasets(split_fpath):
    """
    Construct utterance and formula holdout datasets.
    """
    split_dataset = load_from_file(split_fpath)
    train_data, train_meta = split_dataset["train_iter"], split_dataset["train_meta"]
    test_data, test_meta = split_dataset["valid_iter"], split_dataset["valid_meta"]

    utt_data, utt_meta, formula_data, formula_meta = [], [], [], []
    utt_ltls, formula_ltls = set(), set()
    for (utt, ltl), meta in zip(test_data, test_meta):
        if ltl in train_data:  # TODO: use Spot equivalent check
            utt_data.append((utt, ltl))
            utt_meta.append(meta)
            utt_ltls.add(ltl)
        else:
            formula_data.append((utt, ltl))
            formula_meta.append(meta)
            formula_ltls.add(ltl)

    utt_fpath = os.path.join(os.path.dirname(split_fpath), f"utt_{os.path.basename(split_fpath)}")
    save_to_file({"train_iter": train_data, "train_meta": train_meta, "valid_iter": utt_data, "valid_meta": utt_meta,
                  "info": split_dataset["info"]}, utt_fpath)
    formula_fpath = os.path.join(os.path.dirname(split_fpath), f"formula_{os.path.basename(split_fpath)}")
    save_to_file({"train_iter": train_data, "train_meta": train_meta, "valid_iter": formula_data, "valid_meta": formula_meta,
                  "info": split_dataset["info"]}, formula_fpath)

    logger.info(f"Utterance holdout: {utt_fpath}")
    logger.info(f"Utterance holdout nsamples: {len(utt_data)}, {len(utt_meta)}")
    logger.info(f"Utterance holdout nformulas: {len(utt_ltls)}")
    logger.info(f"Formula holdout: {formula_fpath}")
    logger.info(f"Formula holdout nsamples: {len(formula_data)}, {len(formula_meta)}")
    logger.info(f"Formula holdout nformulas: {len(formula_ltls)}")
    return utt_fpath, formula_fpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_fpath", type=str, default="data/symbolic_batch12_perm.csv", help="base dataset.")
    parser.add_argument("--composed_dpath", type=str, default="data/composed", help="direction to save composed dataset.")
    parser.add_argument("--nclauses", type=int, default=2, help="number of clauses in composed formula.")
    parser.add_argument("--nsamples", type=int, default=10, choices=[10, None], help="number of samples in composed dataset. None to construct entire composed dataset.")
    parser.add_argument("--split_ratio", type=float, default=0.6, help="train test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(f"{os.path.dirname(args.base_fpath)}/log_composed_{Path(args.base_fpath).stem}.log", mode='w'),
                            logging.StreamHandler()
                        ]
                        )
    logger = logging.getLogger()

    base_fpath, composed_fpath = sample_composed_dataset(COMPOSE_OPERATORS, args.base_fpath, args.nsamples, args.seed, args.composed_dpath, logger)
    split_fpath = construct_split_datasets(base_fpath, composed_fpath, args.split_ratio, args.seed, logger)
    # utt_fpath, formula_fpath = construct_holdout_datasets(split_fpath)
