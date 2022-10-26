from utils import load_from_file, save_to_file, substitute


def generate_tar_file():
    """
    Convert letters to words to represent propositions in ground truth LTLs.
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
    save_to_file('\n'.join(true_ltls), "data/test_tar_corlw.txt")


if __name__ == '__main__':
    generate_tar_file()
