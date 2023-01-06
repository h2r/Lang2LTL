from pathlib import Path
import random
import string
from collections import defaultdict
from sklearn.model_selection import train_test_split
import spot

from utils import load_from_file, save_to_file, substitute, substitute_single_fix
from formula_sampler import PROPS, ALL_TYPES, sample_formulas


def generate_tar_file():
    """
    Convert letters to words to represent propositions in ground truth LTLs for spot.formula to work.
    e.g. F & B F C -> F & blue_room F green_room
    For [Gopalan et al. 18 dataset] https://github.com/h2r/language_datasets/tree/master/RSS_2018_Gopalan_et_al
    """
    sub_map = {
        "B": "blue_room",
        "C": "green_room",
        "D": "red_room",
        "Y": "yellow_room",
        "E": "chair_in_green_room",
        "Z": "chair_in_blue_room"
    }
    raw_pairs = load_from_file("data/cleanup_cleaned.csv")
    raw_utts, raw_true_ltls = [], []
    for utt, ltl in raw_pairs:
        raw_utts.append(utt)
        raw_true_ltls.append(ltl)

    true_ltls = substitute(raw_true_ltls, [sub_map]*len(raw_true_ltls))

    pairs = [["Language Command", "LTL Formula"]]
    for utt, ltl in zip(raw_utts, true_ltls):
        pairs.append([utt.strip(), ltl.strip()])
    save_to_file(pairs, "data/cleanup_corlw.csv")


def create_osm_dataset():
    """
    To test generalization capability of LLMs, create an OSM dataset.
    """
    data = load_from_file('data/providence_500.csv')
    data_rand = random.sample(data[:361], 50)

    csv_data = [["Language Command", "LTL Formula"]]
    for symbolic_ltl, utt, ltl in data_rand:
        csv_data.append([utt.lower().strip(), ltl.strip()])
    save_to_file(csv_data, 'data/osm_corlw.csv')

    # manually change all names in LTLs to lower case, connected with underscores

    # check LTL formulas compatible with Spot
    pairs = load_from_file('data/osm_corlw.csv')
    for utt, ltl in pairs:
        spot.formula(ltl)


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
                utt_perm, _ = substitute_single_fix(utt, sub_map)  # sub propositions in utt w/ perm corres to ltl
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


def construct_split_dataset(data_fpath, holdout_type, filter_types, test_size, seed):
    """
    :param data_fpath: path to symbolic dataset
    :param holdout_type: type of holdout test. choices are ltl_type, ltl_instance, utt.
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
        print(f"Train LTL types: {[instance for instance in ALL_TYPES if instance not in holdout_types]}")
        print(f"Holdout LTL types: {holdout_types}")
        for pattern_type, props, utt, ltl in dataset:
            props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
            if pattern_type in holdout_types:
                valid_iter.append((utt, ltl))
                valid_meta.append((pattern_type, len(props)))
            else:
                train_iter.append((utt, ltl))
                train_meta.append((pattern_type, len(props)))
    elif holdout_type == "ltl_instance":  # hold out specified (pattern type, nprops) pairs
        all_instances = []
        for pattern_type, props, _, _ in dataset:
            props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
            instance = (pattern_type, len(props))
            if pattern_type not in filter_types and instance not in all_instances:
                all_instances.append(instance)
        random.seed(seed)
        holdout_instances = random.sample(all_instances, int(len(all_instances)*test_size))
        print(f"Train LTL instances: {[instance for instance in all_instances if instance not in holdout_instances]}")
        print(f"Holdout LTL instances: {holdout_instances}")
        for pattern_type, props, utt, ltl in dataset:
            props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
            if (pattern_type, len(props)) in holdout_instances:
                valid_iter.append((utt, ltl))
                valid_meta.append((pattern_type, len(props)))
            else:
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
    split_fpath = f"data/split_{dataset_name}_{holdout_type}_{test_size}_{seed}.pkl"
    save_to_file(split_dataset, split_fpath)

    # Testing by loading the saved split dataset back
    # train_iter, train_meta, valid_iter, valid_meta = load_split_datast(split_fpath)
    # for idx, ((utt, ltl), (pattern_type, nprop)) in enumerate(zip(train_iter, train_meta)):
    #     print(f"{idx}: {pattern_type} | {nprop} | {ltl} | {utt}")
    # for idx, ((utt, ltl), (pattern_type, nprop)) in enumerate(zip(valid_iter, valid_meta)):
    #     print(f"{idx}: {pattern_type} | {nprop} | {ltl} | {utt}")
    print(f"Number of training samples: {len(train_iter)}")
    print(f"Number of validation samples: {len(valid_iter)}\n")


def load_split_dataset(split_fpath):
    dataset = load_from_file(split_fpath)
    return dataset["train_iter"], dataset["train_meta"], dataset["valid_iter"], dataset["valid_meta"]


if __name__ == '__main__':
    # generate_tar_file()
    # create_osm_dataset()

    # Construct train, test split for 3 types of holdout
    create_symbolic_dataset('data/aggregated_responses_batch1.csv', False)
    create_symbolic_dataset('data/aggregated_responses_batch1.csv', True)

    data_fpath = "data/symbolic_no_perm_batch1.csv"
    filter_types = ["fair_visit"]
    seed = 42
    construct_split_dataset(data_fpath, holdout_type="ltl_type", filter_types=filter_types, test_size=2, seed=seed)
    construct_split_dataset(data_fpath, holdout_type="ltl_instance", filter_types=filter_types, test_size=0.2, seed=seed)
    construct_split_dataset(data_fpath, holdout_type="utt", filter_types=filter_types, test_size=0.2, seed=seed)
