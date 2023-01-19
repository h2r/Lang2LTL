"""
Generate utterance-language grounded dataset, where propositions are from OSM landmarks.

OSM landmarks from https://github.com/jasonxyliu/magic-skydio/tree/master/language/data/map_info_dicts
Map ID to city name mapping from https://github.com/jasonxyliu/magic-skydio/blob/5b16d3f5151b250576ec1cdd513283f377368170/language/copynet/end2end_eval.py#L31
"""
import argparse
import os
from pathlib import Path
import string
from pprint import pprint
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

from utils import load_from_file, save_to_file, substitute_single_letter


def rename_map_files(osm_lmks_dpath):
    """
    Change file names for better interpretation.
    e.g. Map00.json -> new_york_1.json
    One time change after pulling dataset from CopyNet repo.
    """
    mapid2cityname = {
        "Map00": "New York #1", "Map01": "New York #2", "Map02": "Los Angeles #1",
        "Map03": "Los Angeles #2", "Map04": "Chicago #1", "Map05": "Chicago #2",
        "Map07": "Houston", "Map10": "Philadelphia #1", "Map11": "Philadelphia #2",
        "Map12": "San Diego #1", "Map13": "San Diego #2", "Map14": "Jacksonville #1",
        "Map15": "Jacksonville #2", "Map16": "Columbus #1", "Map17": "Columbus #2",
        "Map18": "Charlotte #1", "Map19": "Charlotte #2", "Map20": "Indianapolis",
        "Map22": "Seattle", "Map24": "Denver #1", "Map25": "Denver #2", "Map29": "Boston"
    }
    map_fnames = [fname for fname in os.listdir(osm_lmks_dpath) if os.path.splitext(fname)[1] == ".json"]

    for map_fname in map_fnames:
        if "Map" in map_fname:
            map_fpath = os.path.join(osm_lmks_dpath, map_fname)
            map_id = Path(map_fpath).stem
            if map_id in mapid2cityname:
                city_name = mapid2cityname[map_id]
                if "#" in city_name:
                    city_name = city_name.translate(str.maketrans('', '', string.punctuation))  # remove #
                new_map_fname = "_".join(city_name.lower().split(" ")) + ".json"
                new_map_fpath = os.path.join(osm_lmks_dpath, new_map_fname)
                os.rename(map_fpath, new_map_fpath)


def construct_lmk2prop(osm_lmks_dpath):
    """
    For testing conversion of every landmark name to proposition and inspecting OSM landmark names.
    Construct landmark names to propositions mapping.
    """
    lmk_fnames = [fname for fname in os.listdir(osm_lmks_dpath) if os.path.splitext(fname)[1] == ".json"]
    for lmk_fname in lmk_fnames:
        city_name = os.path.splitext(lmk_fname)[0]
        print(city_name)
        lmk2prop = {}
        lmk_fpath = os.path.join(osm_lmks_dpath, lmk_fname)
        lmk_json = load_from_file(lmk_fpath)
        lmk_names = lmk_json.keys()
        for lmk_name in lmk_names:
            lmk2prop[lmk_name] = lmk_to_prop(lmk_name)
        pprint(lmk2prop)
        print("\n")


def lmk_to_prop(lmk_name):
    """
    :param lmk_name: landmark name, e.g. Canal Street, TD Bank.
    :return: proposition that corresponds to input landmark name and is compatible with Spot.
    """
    return "_".join(lmk_name.translate(str.maketrans('/()-–', '     ', "'’,.!?")).lower().split())


def lmk_to_prop_copynet(lmk_name):
    """
    :param lmk_name: landmark name, e.g. Canal Street, TD Bank.
    :return: proposition in specific form required by CopyNet.
    """
    return f"lm( {lmk_name} )lm"


def construct_osm_dataset(data_fpath, city, filter_types, size, seed, firstn, model):
    dataset_symbolic = load_from_file(data_fpath)
    meta2data = defaultdict(list)
    for pattern_type, props, utt, ltl in dataset_symbolic:
        if pattern_type not in filter_types:
            meta2data[(pattern_type, props)].append((utt, ltl))

    train_iter, train_meta, valid_iter, valid_meta = [], [], [], []  # meta data (type, props, lmk_names, seed_lmk)
    osm_lmks = list(load_from_file(os.path.join(osm_lmks_dpath, f"{city}.json")).keys())
    for (pattern_type, props), data in meta2data.items():
        props = [prop.replace("'", "") for prop in list(props.strip("][").split(", "))]  # "['a', 'b']" -> ['a', 'b']
        _, valid_dataset = train_test_split(data, test_size=size, random_state=seed)
        train_dataset = data[:firstn] if firstn else data
        for idx, (utt, ltl) in enumerate(train_dataset):
            seed_lmk = idx
            utt_grounded, ltl_grounded, lmk_names = substitute_lmk(utt, ltl, osm_lmks, props, seed_lmk, model)
            train_iter.append((utt_grounded, ltl_grounded))
            train_meta.append((pattern_type, props, lmk_names, seed_lmk))
        for idx, (utt, ltl) in enumerate(valid_dataset):
            seed_lmk = idx + 100  # +100 sample diff lmks from train
            utt_grounded, ltl_grounded, lmk_names = substitute_lmk(utt, ltl, osm_lmks, props, seed_lmk, model)
            valid_iter.append((utt_grounded, ltl_grounded))
            valid_meta.append((pattern_type, props, lmk_names, seed_lmk))

    osm_dataset = {
        "train_iter": train_iter, "train_meta": train_meta, "valid_iter": valid_iter, "valid_meta": valid_meta,
        "city": city, "size": size, "seed": seed,
    }
    dataset_name = Path(data_fpath).stem
    osm_dataset_fpath = f"data/osm/{model}_{dataset_name}_{city}_size{size}_seed{seed}"
    if firstn:
        osm_dataset_fpath = f"{osm_dataset_fpath}_first{firstn}.pkl"
    else:
        osm_dataset_fpath = f"{osm_dataset_fpath}.pkl"
    save_to_file(osm_dataset, osm_dataset_fpath)


def substitute_lmk(utt, ltl, osm_lmks, props, seed, model):
    random.seed(seed)
    lmk_names = random.sample(osm_lmks, len(props))

    sub_map = {prop: name for prop, name in zip(props, lmk_names)}
    utt_grounded = substitute_single_letter(utt, sub_map)

    lmk_to_prop_fn = lmk_to_prop if model == "lang2ltl" else lmk_to_prop_copynet
    sub_map = {prop: lmk_to_prop_fn(name) for prop, name in zip(props, lmk_names)}
    ltl_grounded = substitute_single_letter(ltl, sub_map)

    return utt_grounded, ltl_grounded, lmk_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", type=str, default="data/symbolic_no_perm_batch1_0.csv", help="fpath to symbolic dataset.")
    parser.add_argument("--city", type=str, default="all", help="city landmarks from. fname in data/osm/osm_lmks.")
    parser.add_argument("--size", type=float, default=0.2, help="train, test split ratio.")
    parser.add_argument("--seed", action="store", type=int, nargs="+", default=42, help="random seed for train, test split.")
    parser.add_argument("--firstn", type=int, default=None, help="first n training samples.")
    parser.add_argument("--model", type=str, default="copynet", choices=["lang2ltl", "copynet"], help="fpath to symbolic dataset.")
    args = parser.parse_args()

    osm_dpath = os.path.join("data", "osm")
    osm_lmks_dpath = os.path.join(osm_dpath, "osm_lmks")
    # construct_lmk2prop(osm_lmks_dpath)  # for testing

    filter_types = ["fair_visit"]
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    for seed in seeds:
        if args.city == "all":
            for fname in os.listdir(osm_lmks_dpath):
                if ".json" in fname:
                    city = os.path.splitext(fname)[0]
                    construct_osm_dataset(args.data_fpath, city, filter_types, args.size, seed, args.firstn, args.model)
        else:
            construct_osm_dataset(args.data_fpath, args.city, filter_types, args.size, seed, args.firstn, args.model)
