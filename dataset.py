import random
import spot

from utils import load_from_file, save_to_file, substitute


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
    raw_true_ltls = load_from_file("data/test_tar_cleaned.txt")
    true_ltls = substitute(raw_true_ltls, [sub_map])[0]
    data = '\n'.join(true_ltls) + '\n'
    save_to_file(data, "data/test_tar_corlw.txt")


def create_osm_dataset():
    """
    To test generalization capability of LLMs, create an OSM dataset.
    """
    data = load_from_file('data/providence_500.csv')
    data_rand = random.sample(data[:361], 50)

    utts, ltls = [], []
    for entry in data_rand:
        utts.append(entry[1].lower().strip())
        ltls.append(entry[2].strip())

    save_to_file('\n'.join(utts)+'\n', 'data/osm_src_corlw.txt')
    save_to_file('\n'.join(ltls)+'\n', 'data/osm_tar_corlw.txt')

    # manually change all names in LTLs to lower case, connected with underscores

    # check LTL formulas compatible with Spot
    ltls = load_from_file('data/osm_tar_corlw.txt')
    for ltl in ltls:
        spot.formula(ltl)


if __name__ == '__main__':
    generate_tar_file()
