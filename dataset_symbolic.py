"""
Generate utterance-language symbolic dataset, where propositions are letters, and train, test splits.
"""
import argparse
import os
from pathlib import Path
from pprint import pprint
import random
import string
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold

from utils import load_from_file, save_to_file, append_ids_to_path, remove_id_from_path, deserialize_props_str, substitute_single_letter
from formula_sampler import PROPS, FEASIBLE_TYPES, FILTER_TYPES, sample_formulas


def merge_batches(batch_fpaths):
    """
    Merge aggregated Google form responses from multiple csv files into 1.
    Assume all csv files have same field names in same order.
    :param batch_fpaths: aggregated responses from multiple batches of data collection.
    :return: fpath of merged aggregated responses.
    """
    data_aux = load_from_file(batch_fpaths[0], noheader=False)
    field = data_aux[0]
    data_merged = [field]

    for batch_fpath in batch_fpaths:
        print(f"{batch_fpath}\nNumber of responses: {len(load_from_file(batch_fpath))}\n")
        data_merged.extend(load_from_file(batch_fpath))

    postfix = "".join([f"{i}" for i in range(1, len(batch_fpaths) + 1)])
    data_fpath = f"{os.path.commonprefix(batch_fpaths)}{postfix}.csv"
    save_to_file(data_merged, data_fpath)

    return data_fpath


def create_symbolic_dataset(load_fpath, save_fpath, filter_types, update_dataset, remove_pun, perm_props=False):
    """
    Generate non-permuted symbolic dataset for training symbolic translation module.
    :param load_fpath: path to csv file storing Google form responses. Assume fields are t, user, ltl type, nprops, utt.
    :param save_fpath: path to save symbolic dataset.
    :param filter_types: LTL types to filter out.
    :param update_dataset: True if update existing symbolic dataset.
    :param remove_pun: True if remove all punctuations in utt.
    :param perm_props: True if permute props in utterances and LTL formulas.
    """
    if update_dataset or not os.path.isfile(save_fpath):
        data = load_from_file(load_fpath)

        csv_symbolic = [["pattern_type", "props", "utterance", "ltl_formula"]]
        for _, _, pattern_type, nprops, utt in data:
            if remove_pun:
                utt = utt.translate(str.maketrans('', '', string.punctuation))  # remove punctuations for substitution
            pattern_type = "_".join(pattern_type.lower().split())
            if pattern_type not in filter_types:
                ltls, props_perm = sample_formulas(pattern_type, int(nprops), False)  # sample ltls w/ all possible perms
                if perm_props:  # all possible permutations of propositions
                    for ltl, prop_perm in zip(ltls, props_perm):
                        sub_map = {prop_old: prop_new for prop_old, prop_new in zip(PROPS[:int(nprops)], prop_perm)}
                        utt_perm = substitute_single_letter(utt, sub_map)  # sub props in utt w/ permutation corres to ltl
                        csv_symbolic.append([pattern_type, prop_perm, utt_perm.lower(), ltl.strip().replace('\r', '')])
                else:  # propositions only appear in ascending order
                    csv_symbolic.append([pattern_type, PROPS[:int(nprops)], utt.lower(), ltls[0].strip().replace('\r', '')])

        save_to_file(csv_symbolic, save_fpath)


def analyze_symbolic_dataset(data_fpath):
    data = load_from_file(data_fpath, noheader=True)
    print(f"{data_fpath}\nTotal number of utt, ltl pairs in: {len(data)}")

    counter = defaultdict(lambda: defaultdict(int))
    for ltl_type, props, utt, ltl in data:
        # print(f"{ltl_type}, {props}, {utt}\n{spot.formula(ltl)}")
        props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
        counter[ltl_type][len(props)] += 1

    analysis = [["LTL Template Type", "Number of Propositions", "Number of Utterances"]]
    for ltl_type, nprops2count in counter.items():
        for nprops, count in nprops2count.items():
            analysis.append([ltl_type, nprops, count])

    analysis = sorted(analysis)
    pprint(analysis)
    print("\n")

    analysis_fpath = os.path.join("data", f"analysis_{Path(data_fpath).stem}.csv")
    save_to_file(analysis, analysis_fpath)


def construct_split_dataset(data_fpath, split_dpath, holdout_type, feasible_types, filter_types, perm_props, size, seed, firstn):
    """
    K-fold cross validation for type and formula holdout. Random sample for utterance holdout.
    If perm_props=True, assume data_fpath non-permuted for utt holdout, permuted for formula, type holdouts.
    :param data_fpath: path to symbolic dataset.
    :param split_dpath: directory to save train, test split.
    :param holdout_type: type of holdout test. choices are ltl_type, ltl_formula, utt.
    :param feasible_types: all LTL types except filter types
    :param filter_types: LTL types to filter out.
    :param perm_props: True if permute propositions in utterances and their corresponding LTL formula.
    :param size: size of each fold for type and formula holdout; ratio of utterances to holdout for utterance holdout.
    :param seed: random seed for train, test split.
    :param firstn: use first n training samples of each formula for utt, formula, type holdout.
    """
    print(f"Generating train, test split for holdout type: {holdout_type}; seed: {seed}")
    dataset = load_from_file(data_fpath)
    dataset_name = f"{'_'.join(Path(data_fpath).stem.split('_')[:2])}_perm" if perm_props else Path(data_fpath).stem  # remove noperm identifier from dataset name if perm_props=True

    if holdout_type == "ltl_type":  # hold out specified pattern types
        kf = KFold(n_splits=len(feasible_types)//size, random_state=seed, shuffle=True)
        for fold_idx, (train_indices, holdout_indices) in enumerate(kf.split(feasible_types)):
            holdout_types = [feasible_types[idx] for idx in holdout_indices]
            train_iter, train_meta, valid_iter, valid_meta = [], [], [], []  # meta data is (pattern_type, nprops) pair
            formula2count = defaultdict(int)
            for pattern_type, props_str, utt, ltl in dataset:
                props = deserialize_props_str(props_str)
                if pattern_type in holdout_types:
                    valid_iter.append((utt, ltl))
                    valid_meta.append((pattern_type, props))
                elif pattern_type in feasible_types:
                    if firstn:
                        formula = (pattern_type, len(props))
                        if formula2count[formula] < firstn:
                            train_iter.append((utt, ltl))
                            train_meta.append((pattern_type, props))
                            formula2count[formula] += 1
                    else:
                        train_iter.append((utt, ltl))
                        train_meta.append((pattern_type, props))
            dataset_name = Path(data_fpath).stem
            if firstn:
                split_fpath = f"{split_dpath}/{dataset_name}_{holdout_type}_{size}_{seed}_fold{fold_idx}_first{firstn}.pkl"
            else:
                split_fpath = f"{split_dpath}/{dataset_name}_{holdout_type}_{size}_{seed}_fold{fold_idx}.pkl"
            save_split_dataset(split_fpath, train_iter, train_meta, valid_iter, valid_meta,
                               size, seed, holdout_type, holdout_types)
    elif holdout_type == "ltl_formula":  # hold out specified (pattern type, nprops) pairs
        all_formulas = []
        for pattern_type, props_str, _, _ in dataset:
            props = deserialize_props_str(props_str)
            formula = (pattern_type, len(props))
            if pattern_type not in filter_types and formula not in all_formulas:
                all_formulas.append(formula)
        kf = KFold(n_splits=len(all_formulas)//size, random_state=seed, shuffle=True)
        for fold_idx, (train_indices, holdout_indices) in enumerate(kf.split(all_formulas)):
            holdout_formulas = [all_formulas[idx] for idx in holdout_indices]
            train_iter, train_meta, valid_iter, valid_meta = [], [], [], []  # meta data is (pattern_type, nprops) pairs
            formula2count = defaultdict(int)
            for pattern_type, props_str, utt, ltl in dataset:
                props = deserialize_props_str(props_str)
                formula = (pattern_type, len(props))
                if formula in holdout_formulas:
                    valid_iter.append((utt, ltl))
                    valid_meta.append((pattern_type, props))
                elif formula in all_formulas:
                    if firstn:
                        if formula2count[formula] < firstn:
                            train_iter.append((utt, ltl))
                            train_meta.append((pattern_type, props))
                            formula2count[formula] += 1
                    else:
                        train_iter.append((utt, ltl))
                        train_meta.append((pattern_type, props))
            if firstn:
                split_fpath = f"{split_dpath}/{dataset_name}_{holdout_type}_{size}_{seed}_fold{fold_idx}_first{firstn}.pkl"
            else:
                split_fpath = f"{split_dpath}/{dataset_name}_{holdout_type}_{size}_{seed}_fold{fold_idx}.pkl"
            save_split_dataset(split_fpath, train_iter, train_meta, valid_iter, valid_meta,
                               size, seed, holdout_type, holdout_formulas)
    elif holdout_type == "utt":  # hold out a specified ratio of utts for every (pattern type, nprops) pair
        train_iter, train_meta, valid_iter, valid_meta = [], [], [], []  # meta data is (pattern_type, nprops) pairs
        meta2data = defaultdict(list)
        for pattern_type, props_str, utt, ltl in dataset:
            props = deserialize_props_str(props_str)
            if pattern_type not in filter_types:
                meta2data[(pattern_type, len(props))].append((utt, ltl))
        for (pattern_type, nprops), data in meta2data.items():
            train_dataset, valid_dataset = train_test_split(data, test_size=size, random_state=seed)
            if firstn:
                train_dataset = train_dataset[:firstn]
            for utt, ltl in train_dataset:
                if perm_props:
                    permute(pattern_type, nprops, utt, train_iter, train_meta)
                else:
                    train_iter.append((utt, ltl))
                    train_meta.append((pattern_type, PROPS[:nprops]))
            for utt, ltl in valid_dataset:
                if perm_props:
                    permute(pattern_type, nprops, utt, valid_iter, valid_meta)
                else:
                    valid_iter.append((utt, ltl))
                    valid_meta.append((pattern_type, PROPS[:nprops]))
        if firstn:
            split_fpath = f"{split_dpath}/{dataset_name}_{holdout_type}_{size}_{seed}_first{firstn}.pkl"
        else:
            split_fpath = f"{split_dpath}/{dataset_name}_{holdout_type}_{size}_{seed}.pkl"
        save_split_dataset(split_fpath, train_iter, train_meta, valid_iter, valid_meta,
                           size, seed, holdout_type, list(meta2data.keys()))
    else:
        raise ValueError(f"ERROR unrecognized holdout type: {holdout_type}.")


def permute(pattern_type, nprops, utt, data, meta):
    """
    Add utterances and LTL formulas with all possible permutations to data and meta data.
    """
    ltls_perm, props_perm = sample_formulas(pattern_type, nprops, False)  # sample ltls w/ all possible perms
    for ltl_perm, prop_perm in zip(ltls_perm, props_perm):
        sub_map = {prop_old: prop_new for prop_old, prop_new in zip(PROPS[:nprops], prop_perm)}
        utt_perm = substitute_single_letter(utt, sub_map)  # sub props in utt w/ permutation corres to ltl
        data.append((utt_perm, ltl_perm))
        meta.append((pattern_type, prop_perm))


def save_split_dataset(split_fpath, train_iter, train_meta, valid_iter, valid_meta, size, seed, holdout_type=None, holdout_meta=None):
    split_dataset = {
        "train_iter": train_iter, "train_meta": train_meta, "valid_iter": valid_iter, "valid_meta": valid_meta,
        "holdout_type": holdout_type,
        "holdout_meta": holdout_meta,
        "size": size,
        "seed": seed
    }
    save_to_file(split_dataset, split_fpath)


def load_split_dataset(split_fpath):
    dataset = load_from_file(split_fpath)
    return dataset["train_iter"], dataset["train_meta"], dataset["valid_iter"], dataset["valid_meta"]


def generate_prompts_from_split_dataset(split_fpath, prompt_dpath, nexamples, seed):
    """
    :param split_fpath: path to pickle file containing train, test split for a holdout type
    :param nexamples: number of examples for 1 formula
    :return:
    """
    train_iter, train_meta, _, _ = load_split_dataset(split_fpath)

    meta2data = defaultdict(list)
    for idx, ((utt, ltl), (pattern_type, props)) in enumerate(zip(train_iter, train_meta)):
        meta2data[(pattern_type, len(props))].append((utt, ltl))
    sorted(meta2data.items(), key=lambda kv: kv[0])

    prompt = "Your task is to translate English utterances into linear temporal logic (LTL) formulas.\n\n"
    for (pattern_type, nprop), data in meta2data.items():
        random.seed(seed)
        examples = random.sample(data, nexamples)
        for utt, ltl in examples:
            prompt += f"Utterance: {utt}\nLTL: {ltl}\n\n"
            # print(f"{pattern_type} | {nprop}\n{utt}\n{ltl}\n")
    prompt += "Utterance:"

    split_dataset_name = Path(split_fpath).stem
    prompt_fpath = f"{prompt_dpath}/prompt_nexamples{nexamples}_{split_dataset_name}.txt"
    save_to_file(prompt, prompt_fpath)


def generate_lc_splits(split_fpath, portions=[0.1, 0.3, 0.5, 0.7], seed=42):
    """
    Split one seed/fold for plotting learning curve.
    """
    train_iter, train_meta, valid_iter, valid_meta = load_split_dataset(split_fpath)

    meta2data = defaultdict(list)
    for idx, ((utt, ltl), (pattern_type, props)) in enumerate(zip(train_iter, train_meta)):
        meta2data[(pattern_type, len(props))].append((utt, ltl))

    for portion in portions:
        train_iter_new, train_meta_new = [], []
        for (pattern_type, nprop), data in meta2data.items():
            random.seed(seed)
            data = sorted(data)
            random.shuffle(data)
            examples = data[:int(len(data)*portion)]
            print(f"Num of {pattern_type}, {nprop}: {len(examples)}")
            for pair in examples:
                train_iter_new.append(pair)
                train_meta_new.append((pattern_type, nprop))
        split_dataset_name = Path(split_fpath).stem
        lc_split_fname = f"lc_{portion}_{split_dataset_name}.pkl"
        split_pkl_new = {"train_iter": train_iter_new, "train_meta": train_meta_new, "valid_iter": valid_iter, "valid_meta": valid_meta}
        save_to_file(split_pkl_new, os.path.join(os.path.dirname(split_fpath), lc_split_fname))


if __name__ == "__main__":
    # python dataset_symbolic.py --perm --update --merge --nexamples=1
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", action="store", type=str, nargs="+", default=["data/aggregated_responses_batch1.csv", "data/aggregated_responses_batch2.csv"], help="fpath to aggregated Google form responses.")
    parser.add_argument("--perm", action="store_true", help="True if permute props after train, test split.")
    parser.add_argument("--update", action="store_true", help="True if update existing symbolic dataset w/ new responses.")
    parser.add_argument("--merge", action="store_true", help="True if merge Google form responses from batches.")
    parser.add_argument("--remove_pun", action="store_true", help="True if remove punctuations.")
    parser.add_argument("--seeds_split", action="store", type=int, nargs="+", default=[0, 1, 2, 42, 111], help="1 or more random seeds for train, test split.")
    parser.add_argument("--firstn", type=int, default=None, help="only use first n training samples per formula.")
    parser.add_argument("--nexamples", action="store", type=int, nargs="+", default=[1, 2, 3], help="number of examples per formula in prompt.")
    parser.add_argument("--seed_prompt", type=int, default=42, help="random seed for choosing prompt examples.")
    args = parser.parse_args()

    # Merge Google form responses from batches and construct non-permuted symbolic dataset for each batch
    data_fpaths = args.data_fpath if isinstance(args.data_fpath, list) else [args.data_fpath]
    postfix = "".join([f"{i}" for i in range(1, len(data_fpaths) + 1)])
    if args.merge:
        for data_fpath in data_fpaths:
            batch_id = Path(data_fpath).stem.split('_')[-1]
            symbolic_fpath = append_ids_to_path(f"data/symbolic_{batch_id}_noperm.csv", args.remove_pun, "nopun", "pun")
            create_symbolic_dataset(data_fpath, symbolic_fpath, FILTER_TYPES, args.update, args.remove_pun)
            analyze_symbolic_dataset(remove_id_from_path(symbolic_fpath, "pun"))
        data_fpath = merge_batches(data_fpaths)
        symbolic_fpath = append_ids_to_path(f"data/symbolic_batch{postfix}_noperm.csv", args.remove_pun, "nopun", "pun")
        split_dpath = append_ids_to_path(f"data/holdout_split_batch{postfix}", [args.perm, args.remove_pun], ["perm", "nopun"], ["noperm", "pun"])  # dpath to save train, test split
        prompt_dpath = append_ids_to_path(f"data/prompt_symbolic_batch{postfix}", [args.perm, args.remove_pun], ["perm", "nopun"], ["noperm", "pun"])  # dpath to save prompts
    else:
        data_fpath = data_fpaths[0]
        batch_id = Path(data_fpath).stem.split('_')[-1]
        symbolic_fpath = f"data/symbolic_{batch_id}_noperm.csv"
        split_dpath = append_ids_to_path(f"data/holdout_split_{batch_id}", args.perm, "perm", "noperm")  # dpath to save train, test split
        prompt_dpath = append_ids_to_path(f"data/prompt_symbolic_{batch_id}", args.perm, "perm", "noperm")  # dpath to save prompts
    os.makedirs(split_dpath, exist_ok=True)
    os.makedirs(prompt_dpath, exist_ok=True)

    # Construct non-permuted symbolic dataset from Google form responses
    create_symbolic_dataset(data_fpath, symbolic_fpath, FILTER_TYPES, args.update, args.remove_pun)
    analyze_symbolic_dataset(remove_id_from_path(symbolic_fpath, "pun"))

    # Construct train, test split for utt holdout; permute if asked
    seeds = args.seeds_split if isinstance(args.seeds_split, list) else [args.seeds_split]
    for seed in seeds:
        construct_split_dataset(symbolic_fpath, split_dpath, "utt", FEASIBLE_TYPES, FILTER_TYPES, args.perm, size=0.2, seed=seed, firstn=args.firstn)

    # Construct train, test split for formula, type holdout; permute if asked
    if args.perm:
        symbolic_fpath = f"data/symbolic_batch{postfix}_perm.csv" if args.merge else f"data/symbolic_{batch_id}_perm.csv"
        symbolic_fpath = append_ids_to_path(symbolic_fpath, args.remove_pun, "nopun", "pun")
        create_symbolic_dataset(data_fpath, symbolic_fpath, FILTER_TYPES, args.update, args.remove_pun, True)
        analyze_symbolic_dataset(remove_id_from_path(symbolic_fpath, "pun"))
    construct_split_dataset(symbolic_fpath, split_dpath, "ltl_type", FEASIBLE_TYPES, FILTER_TYPES, args.perm, size=3, seed=42, firstn=args.firstn)
    construct_split_dataset(symbolic_fpath, split_dpath, "ltl_formula", FEASIBLE_TYPES, FILTER_TYPES, args.perm, size=9, seed=42, firstn=args.firstn)

    # Generate prompts for off-the-shelf GPT-3
    split_fpaths = [os.path.join(split_dpath, fname) for fname in os.listdir(split_dpath) if "pkl" in fname]
    nexamples = args.nexamples if isinstance(args.nexamples, list) else [args.nexamples]
    for split_fpath in split_fpaths:
        for n in nexamples:
            generate_prompts_from_split_dataset(split_fpath, prompt_dpath, n, args.seed_prompt)
