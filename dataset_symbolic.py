"""
Generate utterance-language symbolic dataset, where propositions are letters.
"""
import argparse
import os
from pathlib import Path
import random
import string
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold

from utils import load_from_file, save_to_file, substitute_single_letter
from formula_sampler import PROPS, ALL_TYPES, sample_formulas


def create_symbolic_dataset(load_fpath, perm_props):
    """
    Generate dataset for training symbolic translation module.
    :param load_fpath: path to csv file storing Google form responses. Assume fields are t, user, ltl type, nprops, utt.
    :param perm_props: True if permute propositions in utterances and LTL formulas.
    """
    data = load_from_file(load_fpath)

    csv_symbolic = [["pattern_type", "props", "utterance", "ltl_formula"]]
    for _, _, pattern_type, nprops, utt in data:
        utt = utt.translate(str.maketrans('', '', string.punctuation))  # remove punctuations for substitution
        pattern_type = "_".join(pattern_type.lower().split())
        ltls, props_perm = sample_formulas(pattern_type, int(nprops), False)  # sample ltls w/ all possible permutations
        if perm_props:  # all possible permutations of propositions
            for ltl, prop_perm in zip(ltls, props_perm):
                sub_map = {prop_old: prop_new for prop_old, prop_new in zip(PROPS[:int(nprops)], prop_perm)}
                utt_perm = substitute_single_letter(utt, sub_map)  # sub props in utt w/ permutation corres to ltl
                csv_symbolic.append([pattern_type, prop_perm, utt_perm.lower().strip(), ltl.strip().replace('\r', '')])
        else:  # propositions only appear in ascending order
            csv_symbolic.append([pattern_type, PROPS[:int(nprops)], utt.lower().strip(), ltls[0].strip().replace('\r', '')])

    save_fpath = 'data/symbolic_perm_batch1.csv' if perm_props else 'data/symbolic_no_perm_batch1.csv'
    save_to_file(csv_symbolic, save_fpath)

    pairs = load_from_file(save_fpath)
    # for pattern_type, props, utt, ltl in pairs:
    #     print(f"{pattern_type}, {props}, {utt}")
    #     print(spot.formula(ltl))
    print(f"Total number of utt, ltl pairs: {len(pairs)}\n")
    return save_fpath


def construct_split_dataset(data_fpath, holdout_type, filter_types, test_size, seed):
    """
    Random sample for type, formula and utterance holdout.
    :param data_fpath: path to symbolic dataset
    :param holdout_type: type of holdout test. choices are ltl_type, ltl_formula, utt.
    :param filter_types: LTL types to filter out.
    :param split_fpath: path to pickle file that stores train, test split
    :param test_size: percentage or number of samples to holdout for testing.
    :param seed: random seed
    """
    print(f"Generating train, test split for holdout type: {holdout_type}")
    dataset = load_from_file(data_fpath)
    train_iter, train_meta, valid_iter, valid_meta = [], [], [], []  # meta data is (pattern_type, nprops) pairs

    if holdout_type == "ltl_type":  # hold out specified pattern types
        all_types = [typ for typ in ALL_TYPES if typ not in filter_types]
        random.seed(seed)
        holdout_types = random.sample(all_types, test_size)
        for pattern_type, props, utt, ltl in dataset:
            props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
            if pattern_type in holdout_types:
                valid_iter.append((utt, ltl))
                valid_meta.append((pattern_type, len(props)))
            elif pattern_type in all_types:
                train_iter.append((utt, ltl))
                train_meta.append((pattern_type, len(props)))
    elif holdout_type == "ltl_formula":  # hold out specified (pattern type, nprops) pairs
        all_formulas = []
        for pattern_type, props, _, _ in dataset:
            props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
            formula = (pattern_type, len(props))
            if pattern_type not in filter_types and formula not in all_formulas:
                all_formulas.append(formula)
        random.seed(seed)
        holdout_formulas = random.sample(all_formulas, int(len(all_formulas)*test_size))
        for pattern_type, props, utt, ltl in dataset:
            props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
            formula = (pattern_type, len(props))
            if formula in holdout_formulas:
                valid_iter.append((utt, ltl))
                valid_meta.append((pattern_type, len(props)))
            elif formula in all_formulas:
                train_iter.append((utt, ltl))
                train_meta.append((pattern_type, len(props)))
    elif holdout_type == "utt":  # hold out a specified ratio of utts for every (pattern type, nprops) pair
        meta2data = defaultdict(list)
        for pattern_type, props, utt, ltl in dataset:
            props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
            if pattern_type not in filter_types:
                meta2data[(pattern_type, len(props))].append((utt, ltl))
        for meta, data in meta2data.items():
            if len(data) == 1:
                train_iter.append(data[0])
                train_meta.append(meta)
            else:
                train_dataset, valid_dataset = train_test_split(data, test_size=test_size, random_state=seed)
                for utt, ltl in train_dataset:
                    train_iter.append((utt, ltl))
                    train_meta.append(meta)
                for utt, ltl in valid_dataset:
                    valid_iter.append((utt, ltl))
                    valid_meta.append(meta)
    else:
        raise ValueError(f"ERROR unrecognized holdout type: {holdout_type}.")

    split_dataset = {
        "train_iter": train_iter, "train_meta": train_meta, "valid_iter": valid_iter, "valid_meta": valid_meta,
        "holdout_type": holdout_type,
        "filter_types": filter_types,
        "test_size": test_size,
        "seed": seed
    }
    dataset_name = Path(data_fpath).stem
    split_fpath = f"data/holdout_splits/split_{dataset_name}_{holdout_type}_{test_size}_{seed}.pkl"
    save_to_file(split_dataset, split_fpath)

    # Testing by loading the saved split dataset back
    train_iter, train_meta, valid_iter, valid_meta = load_split_dataset(split_fpath)
    print(f"Training set: {set(train_meta)}")
    print(f"Validation set: {set(valid_meta)}")
    print(f"Number of training samples: {len(train_iter)}")
    print(f"Number of validation samples: {len(valid_iter)}\n")


def construct_split_dataset_new(data_fpath, holdout_type, filter_types, size, firstn, seed):
    """
    K-fold cross validation for type and formula holdout. Random sample for utterance holdout.
    :param data_fpath: path to symbolic dataset
    :param holdout_type: type of holdout test. choices are ltl_type, ltl_formula, utt.
    :param filter_types: LTL types to filter out.
    :param split_fpath: path to pickle file that stores train, test split.
    :param size: size of each fold for type and formula holdout; percentage utterances to holdout for utterance holdout.
    :param firstn: use first n training samples of each formula for utt, formula, type holdout.
    :param seed: random seed.
    """
    print(f"Generating train, test split for holdout type: {holdout_type}")
    dataset = load_from_file(data_fpath)

    if holdout_type == "ltl_type":  # hold out specified pattern types
        all_types = [typ for typ in ALL_TYPES if typ not in filter_types]
        kf = KFold(n_splits=len(all_types)//size, random_state=seed, shuffle=True)
        for fold_idx, (train_indices, holdout_indices) in enumerate(kf.split(all_types)):
            holdout_types = [all_types[idx] for idx in holdout_indices]
            train_iter, train_meta, valid_iter, valid_meta = [], [], [], []  # meta data is (pattern_type, nprops) pairs
            formua2count = defaultdict(int)
            for pattern_type, props, utt, ltl in dataset:
                props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
                if pattern_type in holdout_types:
                    valid_iter.append((utt, ltl))
                    valid_meta.append((pattern_type, len(props)))
                elif pattern_type in all_types:
                    if firstn:
                        formula = (pattern_type, len(props))
                        if formua2count[formula] < firstn:
                            train_iter.append((utt, ltl))
                            train_meta.append((pattern_type, len(props)))
                            formua2count[formula] += 1
                    else:
                        train_iter.append((utt, ltl))
                        train_meta.append((pattern_type, len(props)))
            dataset_name = Path(data_fpath).stem
            if firstn:
                split_fpath = f"data/holdout_splits_new/{dataset_name}_{holdout_type}_{size}_{seed}_fold{fold_idx}_first{firstn}.pkl"
            else:
                split_fpath = f"data/holdout_splits_new/{dataset_name}_{holdout_type}_{size}_{seed}_fold{fold_idx}.pkl"
            save_split_dataset(split_fpath, train_iter, train_meta, valid_iter, valid_meta,
                               holdout_type, filter_types, size, seed)
    elif holdout_type == "ltl_formula":  # hold out specified (pattern type, nprops) pairs
        all_formulas = []
        for pattern_type, props, _, _ in dataset:
            props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
            formula = (pattern_type, len(props))
            if pattern_type not in filter_types and formula not in all_formulas:
                all_formulas.append(formula)
        kf = KFold(n_splits=len(all_formulas)//size, random_state=seed, shuffle=True)
        for fold_idx, (train_indices, holdout_indices) in enumerate(kf.split(all_formulas)):
            holdout_formulas = [all_formulas[idx] for idx in holdout_indices]
            train_iter, train_meta, valid_iter, valid_meta = [], [], [], []  # meta data is (pattern_type, nprops) pairs
            formula2count = defaultdict(int)
            for pattern_type, props, utt, ltl in dataset:
                props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
                formula = (pattern_type, len(props))
                if formula in holdout_formulas:
                    valid_iter.append((utt, ltl))
                    valid_meta.append((pattern_type, len(props)))
                elif formula in all_formulas:
                    if firstn:
                        if formula2count[formula] < firstn:
                            train_iter.append((utt, ltl))
                            train_meta.append((pattern_type, len(props)))
                            formula2count[formula] += 1
                    else:
                        train_iter.append((utt, ltl))
                        train_meta.append((pattern_type, len(props)))
            dataset_name = Path(data_fpath).stem
            if firstn:
                split_fpath = f"data/holdout_splits_new/{dataset_name}_{holdout_type}_{size}_{seed}_fold{fold_idx}_first{firstn}.pkl"
            else:
                split_fpath = f"data/holdout_splits_new/{dataset_name}_{holdout_type}_{size}_{seed}_fold{fold_idx}.pkl"
            save_split_dataset(split_fpath, train_iter, train_meta, valid_iter, valid_meta,
                               holdout_type, filter_types, size, seed)
    elif holdout_type == "utt":  # hold out a specified ratio of utts for every (pattern type, nprops) pair
        train_iter, train_meta, valid_iter, valid_meta = [], [], [], []  # meta data is (pattern_type, nprops) pairs
        meta2data = defaultdict(list)
        for pattern_type, props, utt, ltl in dataset:
            props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
            if pattern_type not in filter_types:
                meta2data[(pattern_type, len(props))].append((utt, ltl))
        for meta, data in meta2data.items():
            if len(data) == 1:
                train_iter.append(data[0])
                train_meta.append(meta)
            else:
                train_dataset, valid_dataset = train_test_split(data, test_size=size, random_state=seed)
                if firstn:
                    train_dataset = train_dataset[:firstn]
                for utt, ltl in train_dataset:
                    train_iter.append((utt, ltl))
                    train_meta.append(meta)
                for utt, ltl in valid_dataset:
                    valid_iter.append((utt, ltl))
                    valid_meta.append(meta)
        dataset_name = Path(data_fpath).stem
        if firstn:
            split_fpath = f"data/holdout_splits_new/{dataset_name}_{holdout_type}_{size}_{seed}_first{firstn}.pkl"
        else:
            split_fpath = f"data/holdout_splits_new/{dataset_name}_{holdout_type}_{size}_{seed}.pkl"
        save_split_dataset(split_fpath, train_iter, train_meta, valid_iter, valid_meta,
                           holdout_type, filter_types, size, seed)
    else:
        raise ValueError(f"ERROR unrecognized holdout type: {holdout_type}.")


def save_split_dataset(split_fpath, train_iter, train_meta, valid_iter, valid_meta, holdout_type, filter_types, size, seed):
    split_dataset = {
        "train_iter": train_iter, "train_meta": train_meta, "valid_iter": valid_iter, "valid_meta": valid_meta,
        "holdout_type": holdout_type,
        "filter_types": filter_types,
        "size": size,
        "seed": seed
    }
    save_to_file(split_dataset, split_fpath)


def load_split_dataset(split_fpath):
    dataset = load_from_file(split_fpath)
    return dataset["train_iter"], dataset["train_meta"], dataset["valid_iter"], dataset["valid_meta"]


def generate_prompts_from_split_dataset(split_fpath, nexamples, seed):
    """
    :param split_fpath: path to pickle file containing train, test split for a holdout type
    :param nexamples: number of examples for 1 formula
    :return:
    """
    train_iter, train_meta, _, _ = load_split_dataset(split_fpath)

    meta2data = defaultdict(list)
    for idx, ((utt, ltl), (pattern_type, nprop)) in enumerate(zip(train_iter, train_meta)):
        meta2data[(pattern_type, nprop)].append((utt, ltl))
    sorted(meta2data.items(), key=lambda kv: kv[0])

    prompt = "Your task is to translate English utterances into linear temporal logic (LTL) formulas.\n\n"
    for (pattern_type, nprop), data in meta2data.items():
        random.seed(seed)
        examples = random.sample(data, nexamples)
        for utt, ltl in examples:
            prompt += f"Utterance: {utt}\nLTL: {ltl}\n\n"
            print(f"{pattern_type} | {nprop}\n{utt}\n{ltl}\n")
    prompt += "Utterance:"

    split_dataset_name = Path(split_fpath).stem
    prompt_fpath = f"data/symbolic_prompts_new/prompt_{nexamples}_{split_dataset_name}.txt"
    save_to_file(prompt, prompt_fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", type=str, default="data/aggregated_responses_batch1.csv", help="fpath to aggregated Google form responses.")
    parser.add_argument("--perm", action="store_true", help="construct symbolic dataset with permuted propositions.")
    parser.add_argument("--seeds_split", action="store", type=int, nargs="+", default=42, help="random seed for train, test split.")
    parser.add_argument("--firstn", type=int, default=None, help="first n training samples.")
    parser.add_argument("--nexamples", action="store", type=int, nargs="+", default=1, help="number of examples in prompt.")
    parser.add_argument("--seed_prompt", type=int, default=None, help="random seed for choosing examples.")
    args = parser.parse_args()

    # Construct train, test split for 3 types of holdout for symbolic translation
    symbolic_fpath = create_symbolic_dataset(args.data_fpath, args.perm)

    filter_types = ["fair_visit"]
    seeds = args.seeds_split if isinstance(args.seeds_split, list) else [args.seeds_split]
    for seed in seeds:
        construct_split_dataset_new(symbolic_fpath, holdout_type="ltl_type", filter_types=filter_types, size=1, firstn=args.firstn, seed=seed)
        construct_split_dataset_new(symbolic_fpath, holdout_type="ltl_formula", filter_types=filter_types, size=5, firstn=args.firstn, seed=seed)
        construct_split_dataset_new(symbolic_fpath, holdout_type="utt", filter_types=filter_types, size=0.2, firstn=args.firstn, seed=seed)

    # Generate prompts for off-the-shelf GPT-3
    split_dpath = os.path.join("data", "holdout_splits_new")
    split_fpaths = [os.path.join(split_dpath, fname) for fname in os.listdir(split_dpath) if "pkl" in fname]
    for split_fpath in split_fpaths:
        for n in args.nexamples:
            generate_prompts_from_split_dataset(split_fpath, n, args.seed_prompt)