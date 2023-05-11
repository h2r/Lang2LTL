"""
Construct a randomly sampled subset of the composed dataset.
"""
import os
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

from compose import COMPOSE_OPERATORS, compose_and, compose_or
from utils import load_from_file, save_to_file, deserialize_props_str


def load_base_dataset(base_fpath, logger):
    base_dataset = load_from_file(base_fpath)
    meta2pairs, base_data, base_meta, base_ltls = defaultdict(list), [], [], set()
    for pattern_type, props_str, utt, ltl in base_dataset:
        props = tuple(deserialize_props_str(props_str))
        meta2pairs[(pattern_type, props)].append((utt, ltl))  # meta not ltl as key due to repeated ltl strs, visit 1 == low restricted avoidance 1
        base_data.append((utt, ltl))
        base_meta.append((pattern_type, props))
        base_ltls.add(ltl)
    base_size = sum([len(pairs) for pairs in meta2pairs.values()])
    assert base_size == len(base_data) == len(base_meta), f"base size not match {base_size} {len(base_data)} {len(base_meta)}"
    logger.info(f"Base dataset nsamples: {base_size} {len(base_data)} {len(base_meta)}")
    logger.info(f"Base dataset nformulas: {len(meta2pairs)}; repeats: {len(meta2pairs)-len(base_ltls)}")
    return meta2pairs, base_data, base_meta


def split_utts(meta2pairs, split_ratio, seed, logger):
    """
    Split utterances per LTL into train test to avoid utterance leakage from training to test set.
    :param meta2pairs: dict of unique formula id (pattern_type, props) to utt-ltl pairs
    :param split_ratio: ratio of splitting utterances per LTL into train test
    """
    meta2pairs_train, meta2pairs_test = {}, {}
    for meta, pairs in meta2pairs.items():
        assert len(set([ltl for _, ltl in pairs])) == 1, f"{meta} more than 1 ltl {[ltl for _, ltl in pairs]}"
        pairs_train, pairs_test = train_test_split(pairs, test_size=split_ratio, random_state=seed)
        meta2pairs_train[meta] = pairs_train
        meta2pairs_test[meta] = pairs_test
        logger.info(f"{meta}: train nutts {len(pairs_train)}; test nutts {len(pairs_test)}\n{pairs_train[0][0]}\n")
    return meta2pairs_train, meta2pairs_test


def sample_composed_dataset(meta2pairs, compose_operators, nsamples, seed, logger):
    base_data, base_meta = [], []
    for meta, pairs in meta2pairs.items():
        base_data.extend(pairs)
        base_meta.extend([meta] * len(pairs))
        assert len(base_data) == len(base_meta), f"base data size not match meta {base_data} {base_meta}"

    random.seed(seed)
    composed_data, composed_meta = [], []
    for sample_id in tqdm(range(nsamples)):
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

        type_composed = '-'.join([compose_operator] + [f"{pattern_type}_{len(props)}" for pattern_type, props in meta])
        _, props = zip(*meta)
        props_composed = sum(props, ())
        meta_composed = (type_composed, props_composed)  # "and-visit_2-global_avoidance_1", ("a", "b", "c")

        composed_data.append((utt_composed, ltl_composed))
        composed_meta.append(meta_composed)
    return composed_data, composed_meta


def costruct_composed_dataset(compose_operators, base_fpath, utt_split_ratio, test_split_ratio, nsamples, seed, composed_dpath, logger):
    meta2pairs, base_data, base_meta = load_base_dataset(base_fpath, logger)

    meta2pairs_train, meta2pairs_test = split_utts(meta2pairs, utt_split_ratio, seed, logger)

    nsamples_train = int(nsamples * (1 - test_split_ratio))
    composed_train, meta_train = sample_composed_dataset(meta2pairs_train, compose_operators, nsamples_train, seed, logger)
    composed_dataset = {"train_iter": base_data+composed_train, "train_meta": base_meta+meta_train}
    composed_fpath = os.path.join(composed_dpath, f"split-sample_nsamples{nsamples}_raito{utt_split_ratio}-{test_split_ratio}_seed{args.seed}_{Path(base_fpath).stem}.pkl")
    save_to_file(composed_dataset, composed_fpath)

    nsamples_test = nsamples - nsamples_train
    composed_test, meta_test = sample_composed_dataset(meta2pairs_test, compose_operators, nsamples_test, seed, logger)
    composed_dataset = load_from_file(composed_fpath)
    composed_dataset["valid_iter"], composed_dataset["valid_meta"] = composed_test, meta_test
    save_to_file(composed_dataset, composed_fpath)
    logger.info(f"Composed dataset: {composed_fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_fpath", type=str, default="data/symbolic_batch12_perm.csv", help="base dataset.")
    parser.add_argument("--composed_dpath", type=str, default="data/composed", help="dir to save composed dataset.")
    parser.add_argument("--nclauses", type=int, default=2, help="number of clauses in composed formula.")
    parser.add_argument("--nsamples", type=int, default=1000, help="number of samples in composed dataset. None to construct entire composed dataset.")
    parser.add_argument("--utt_split_ratio", type=float, default=0.3, help="ratio to split utts per ltl into train test.")
    parser.add_argument("--test_split_ratio", type=float, default=0.6, help="train test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    args = parser.parse_args()

    log_fpath = f"{args.composed_dpath}/split-sample_nsamples{args.nsamples}_raito{args.utt_split_ratio}-{args.test_split_ratio}_seed{args.seed}_{Path(args.base_fpath).stem}.log"
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(log_fpath, mode='w'),
                            logging.StreamHandler()
                        ]
                        )
    logger = logging.getLogger()

    costruct_composed_dataset(COMPOSE_OPERATORS, args.base_fpath, args.utt_split_ratio, args.test_split_ratio, args.nsamples, args.seed, args.composed_dpath, logger)
