import os
import re
from pprint import pprint

from utils import load_from_file, save_to_file, name_to_prop


def filter_cleanup(env):
    cleanup_dpath = os.path.join("data", env)
    cleanup_fpath = os.path.join(cleanup_dpath, "cleanup_corlw.csv")
    dataset = load_from_file(cleanup_fpath)

    data, meta = [], []
    filter_dataset = [["Pattern Type", "Number of Propositions", "Utterance", "True LTL"]]
    for utt, ltl in dataset:
        if "|" not in ltl and "G" not in ltl:
            pattern_type = "visit"
            nprops = ltl.count("F")
            data.append((utt, ltl))
            meta.append((pattern_type, nprops, "cleanup"))
            filter_dataset.append((pattern_type, nprops, utt, ltl))

    cleanup_filtered_fpath = os.path.join(cleanup_dpath, "cleanup_filtered.pkl")
    save_to_file({"valid_iter": data, "valid_meta": meta}, cleanup_filtered_fpath)

    cleanup_filtered_fpath = os.path.join(cleanup_dpath, "cleanup_filtered.csv")
    save_to_file(filter_dataset, cleanup_filtered_fpath)

    print(f"Filter CleanUp dataset from Gopalan et al.: {len(dataset)} -> {len(filter_dataset)}")


def filter_osm(env):
    # def convert_prop(ltl_list):
    #     ltl_list_new = []
    #     for op in ltl_list:  # try to match prop in true LTL to prop converted from complete lmk name
    #         op = op.replace("tomo_sushiramen", "tomo_sushi_and_ramen")
    #         op = op.replace("beijing", "beijing_restaurant")
    #         op = op.replace("nikki_fine_jewelers", "nikki_cofine_jewelers")
    #         op = op.replace("buon_appetito", "buon_appetito_restaurant")
    #         op = op.replace("dahlia_nightclub", "dahlia_night_club")
    #         op = op.replace("el_arado", "el_arado_mexican_grill")
    #         ltl_list_new.append(op)
    #     return ltl_list_new

    osm_dpath = os.path.join("data", env)
    osm_fpath = os.path.join(osm_dpath, "osm_berg.csv")
    dataset = load_from_file(osm_fpath)

    data, meta = [], []
    filter_dataset = [["Pattern Type", "Number of Propositions", "Utterance", "True LTL", "City"]]
    for utt, ltl, city in dataset:
        if "~" not in ltl:
            pattern_type = "visit"
            nprops = ltl.count("F")

            # Convert infix to prefix only for sequenced visit 1, 2
            if nprops == 1:
                ltl = ltl.translate(str.maketrans('', '', "()&'")).strip()
                ltl = re.sub(' +', ' ', ltl)
                ltl_list = ltl.split(" ")
                # ltl_list = convert_prop(ltl_list)
                ltl = " ".join(ltl_list)
            elif nprops == 2:
                ltl = ltl.translate(str.maketrans('', '', "()&'")).strip()
                ltl = re.sub(' +', ' ', ltl)
                ltl_list = ltl.split(" ")
                # ltl_list = convert_prop(ltl_list)
                ltl_list.insert(1, "&")
                ltl = " ".join(ltl_list)
            else:
                raise ValueError(f"ERROR: invalid nprops {nprops}:\n{utt}\n{ltl}")

            data.append((utt, ltl))
            meta.append((pattern_type, nprops, city))
            filter_dataset.append((pattern_type, nprops, utt, ltl, city))

    osm_filtered_fpath = os.path.join(osm_dpath, "osm_filtered.pkl")
    save_to_file({"valid_iter": data, "valid_meta": meta}, osm_filtered_fpath)

    osm_filtered_fpath = os.path.join(osm_dpath, "osm_filtered.csv")
    save_to_file(filter_dataset, osm_filtered_fpath)

    print(f"Filter OSM dataset from Berg et al.: {len(dataset)} -> {len(filter_dataset)}")


if __name__ == "__main__":
    filter_cleanup("cleanup_gopalan")
    filter_osm("osm_berg")
