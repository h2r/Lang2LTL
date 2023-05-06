import os
from pathlib import Path
import argparse
import logging
import random
from sklearn.model_selection import train_test_split

from utils import load_from_file, save_to_file, deserialize_props_str


COMPOSE_OPERATORS = ["and", "or"]


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
    """
    Construct a randomly sampled subset of the composed dataset.
    """
    base_data, base_meta = load_base_dataset(base_fpath, logger)
    based_dataset = {"data": base_data, "meta": base_meta}
    base_fpath = os.path.join(composed_dpath, f"base_{Path(base_fpath).stem}.pkl")
    save_to_file(based_dataset, base_fpath)

    composed_data, composed_meta = [], []
    random.seed(seed)
    for sample_id in range(nsamples):
        logger.info(f"Composing sample {sample_id}")
        compose_operator = random.sample(compose_operators, 1)

        data, meta = zip(*random.sample(list(zip(base_data, base_meta)), 2))
        utts_base, ltls_base = zip(*data)

        if compose_operator == "and":
            utt_composed, ltl_composed = compose_and(utts_base, ltls_base)
        elif compose_operator == "or":
            utt_composed, ltl_composed = compose_or(utts_base, ltls_base)
        else:
            raise ValueError(f"ERROR: operator not yet supported: {compose_operator}.")

        composed_data.append((utt_composed, ltl_composed))
        composed_meta.append(meta)

    composed_dataset = {"data": composed_data, "meta": composed_meta}
    composed_fpath = os.path.join(composed_dpath, f"sample_composed_{Path(base_fpath).stem}.pkl")
    save_to_file(composed_dataset, composed_fpath)
    return base_fpath, composed_fpath


def construct_split_dataset(composed_fpath, split_ratio, seed, logger):
    """
    Construct composed datasets for zero shot transfer, utterance and formula holdout.
    """
    comosed_dataset = load_from_file(composed_fpath)
    composed_data, composed_meta = comosed_dataset["data"], comosed_dataset["meta"]
    train_dataset, valid_dataset = train_test_split(list(zip(composed_data, composed_meta)), test_size=split_ratio, random_state=seed)

    utt_data, utt_meta, formula_data, formula_meta = [], [], [], []
    utt_ltls, formula_ltls = set(), set()
    for (utt, ltl), meta in valid_dataset:
        if ltl in train_dataset:
            utt_data.append((utt, ltl))
            utt_meta.append(meta)
            utt_ltls.add(ltl)
        else:
            formula_data.append((utt, ltl))
            formula_meta.append(meta)
            formula_ltls.add(ltl)

    logger.info(f"Utterance holdout nsamples: {len(utt_data)}, {len(utt_meta)}")
    logger.info(f"Utterance holdout nformulas: {len(utt_ltls)}")
    logger.info(f"Formula holdout nsamples: {len(formula_data)}, {len(formula_meta)}")
    logger.info(f"Formula holdout nformulas: {len(formula_ltls)}")


def compose_and(utts, formulas):
    utt_composed = f"{utts[0]}, in addition {utts[1]}"
    formula_composed = f"& {formulas[0]} {formulas[1]}"
    return utt_composed, formula_composed


def compose_or(utts, formulas):
    utt_composed = f"Either {utts[0]}, or {utts[1]}"
    formula_composed = f"| {formulas[0]} {formulas[1]}"
    return utt_composed, formula_composed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_fpath", type=str, default="data/symbolic_batch12_perm.csv", help="base dataset.")
    parser.add_argument("--composed_dpath", type=str, default="data", help="root direction path of composed dataset.")
    parser.add_argument("--nclauses", type=int, default=2, help="number of clauses in composed formula.")
    parser.add_argument("--nsamples", type=int, default=2, choices=[2, None], help="number of samples in composed dataset. None to construct entire composed dataset.")
    parser.add_argument("--split_ratio", type=float, default=0.4, help="train test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(f"{os.path.dirname(args.base_fpath)}/log_composed_{Path(args.base_fpath).stem}.log", mode='w'),
                            logging.StreamHandler()
                        ]
                        )
    logger = logging.getLogger()

    base_fpath, composed_fpath = sample_composed_dataset(COMPOSE_OPERATORS, args.base_fpath, args.nsamples, args.seed, args.composed_dpath, logger)

    construct_split_dataset(composed_fpath, args.split_ratio, args.seed, logger)
