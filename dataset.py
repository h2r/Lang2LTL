import random
import string
from collections import defaultdict
from sklearn.model_selection import train_test_split
import spot

from utils import load_from_file, save_to_file, substitute, substitute_single_fix
from formula_sampler import PROPS, sample_formulas


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
    for pattern_type, props, utt, ltl in pairs:
        print(f"{pattern_type}, {props}, {utt}")
        print(spot.formula(ltl))
    print(f"Total number of utt, ltl pairs: {len(pairs)}")


def construct_dataset(data_fpath, holdout_type, **kwargs):
    dataset = load_from_file(data_fpath)
    train_iter, train_meta, valid_iter, valid_meta = [], [], [], []  # meta data is (pattern_type, nprops) pairs

    if holdout_type == "ltl_template":  # hold out specified pattern types
        for pattern_type, props, utt, ltl in dataset:
            props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
            if pattern_type in kwargs["holdout_templates"]:
                valid_iter.append((utt, ltl))
                valid_meta.append((pattern_type, len(props)))
            else:
                train_iter.append((utt, ltl))
                train_meta.append((pattern_type, len(props)))
    elif holdout_type == "ltl_instance":  # hold out specified (pattern type, nprops) pairs
        for pattern_type, props, utt, ltl in dataset:
            props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
            if (pattern_type, len(props)) in kwargs["holdout_instances"]:
                valid_iter.append((utt, ltl))
                valid_meta.append((pattern_type, len(props)))
            else:
                train_iter.append((utt, ltl))
                train_meta.append((pattern_type, len(props)))
    elif holdout_type == "utt":  # hold out a specified ratio of utts for every (pattern type, nprops) pair
        meta2data = defaultdict(list)
        for pattern_type, props, utt, ltl in dataset:
            props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
            meta2data[(pattern_type, len(props))].append((utt, ltl))
        for meta, data in meta2data.items():
            if len(data) == 1:
                train_iter.append(data[0])
                train_meta.append(meta)
            else:
                train_dataset, valid_dataset = train_test_split(data, test_size=kwargs["test_size"], random_state=kwargs["seed"])
                for utt, ltl in train_dataset:
                    train_iter.append((utt, ltl))
                    train_meta.append(meta)
                for utt, ltl in valid_dataset:
                    valid_iter.append((utt, ltl))
                    valid_meta.append(meta)
    else:
        raise ValueError(f"ERROR unrecognized holdout type: {holdout_type}.")

    return train_iter, train_meta, valid_iter, valid_meta


if __name__ == '__main__':
    # generate_tar_file()
    # create_osm_dataset()
    # create_symbolic_dataset('data/aggregated_responses_batch1.csv', True)

    # For testing 3 types of holdout test split
    holdout_type = "utt"
    data_fpath = 'data/symbolic_no_perm_batch1.csv'
    if holdout_type == "ltl_template":
        kwargs = {"holdout_templates": ["strictly_ordered_visit"]}
    elif holdout_type == "ltl_instance":
        kwargs = {"holdout_instances": [("sequenced_visit", 3)]}
    elif holdout_type == "utt":
        kwargs = {"test_size": 0.2, "seed": 42}
    else:
        raise ValueError(f"ERROR unrecognized holdout type: {holdout_type}.")
    train_iter, train_meta, valid_iter, valid_meta = construct_dataset(data_fpath, holdout_type, **kwargs)
    # for idx, ((utt, ltl), (pattern_type, nprop)) in enumerate(zip(train_iter, train_meta)):
    #     print(f"{idx}: {pattern_type} | {nprop} | {ltl} | {utt}")
    for idx, ((utt, ltl), (pattern_type, nprop)) in enumerate(zip(valid_iter, valid_meta)):
        print(f"{idx}: {pattern_type} | {nprop} | {ltl} | {utt}")
