import random
import string
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
    :param load_fpath: file path to load Google form responses
    :param perm_props: True if permute propositions in utterances and LTL formulas
    """
    data = load_from_file(load_fpath)

    csv_symbolic = [["utterance", "ltl_formula"]]
    for _, _, pattern_type, nprops, utt in data:
        utt = utt.translate(str.maketrans('', '', string.punctuation))  # remove punctuations for substitution
        pattern_type = "_".join(pattern_type.lower().split())
        ltls, props_perm = sample_formulas(pattern_type, int(nprops), False)  # sample ltls w/ all possible permutations
        if perm_props:  # all possible permutations of propositions
            for ltl, prop_perm in zip(ltls, props_perm):
                sub_map = {prop_old: prop_new for prop_old, prop_new in zip(PROPS[:int(nprops)], prop_perm)}
                utt_perm, _ = substitute_single_fix(utt, sub_map)  # sub propositions in utt w/ perm corres to ltl
                csv_symbolic.append([utt_perm.lower().strip(), ltl.strip().replace('\r', '')])
        else:  # propositions only appear in ascending order
            csv_symbolic.append([utt.lower().strip(), ltls[0].strip().replace('\r', '')])

    save_fpath = 'data/symbolic_pairs_perm.csv' if perm_props else 'data/symbolic_pairs_no_perm.csv'
    save_to_file(csv_symbolic, save_fpath)

    pairs = load_from_file(save_fpath)
    for utt, ltl in pairs:
        print(utt)
        print(spot.formula(ltl))


if __name__ == '__main__':
    # generate_tar_file()
    # create_osm_dataset()
    create_symbolic_dataset('data/aggregated_responses_batch_1.csv', True)
