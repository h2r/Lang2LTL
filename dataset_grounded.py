"""
Generate utterance-language grounded dataset with props substituted by lmks from a specific domain, e.g. OSM, CleanUp.
"""
import argparse
import os
from pathlib import Path
import string
from itertools import permutations
import random
from collections import defaultdict
from pprint import pprint
import time

from utils import load_from_file, save_to_file, substitute_single_letter, name_to_prop
from formula_sampler import PROPS


def rename_map_files(osm_lmks_dpath):
    """
    Change OSM map file names for better interpretation.
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


def construct_lmk2prop(osm_lmks_dpath, model):
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
            lmk2prop[lmk_name] = name_to_prop(lmk_name, model)
        pprint(lmk2prop)
        print("\n")


def construct_grounded_dataset(split_dpath, lmks, city, remove_perm, seed, nsamples, add_comma, model, save_dpath):
    print(f"Constructing grounded dataset for city: {city}")
    split_fnames = sorted([fname for fname in os.listdir(split_dpath) if "pkl" in fname])
    save_dpath = os.path.join(save_dpath, "grounded", city)
    os.makedirs(save_dpath, exist_ok=True)

    nprops2lmkperms = {} if nsamples else None

    for split_fname in split_fnames:
        print(split_fname)
        dataset = load_from_file(os.path.join(split_dpath, split_fname))
        train_iter, train_meta, valid_iter, valid_meta = dataset["train_iter"], dataset["train_meta"], dataset["valid_iter"], dataset["valid_meta"]

        print(f"old train, test size: {len(train_iter)} {len(train_meta)} {len(valid_iter)} {len(valid_meta)}")
        print(f"test size before remove_perm: {len(valid_iter)} {len(valid_meta)}")
        if remove_perm:  # remove prop perms from valid_iter
            valid_iter_noperm, valid_meta_noperm = [], []
            formula2data = defaultdict(list)
            for (utt, ltl), (pattern_type, props) in zip(valid_iter, valid_meta):
                props = list(props)
                sub_map = {old_prop: new_prop for old_prop, new_prop in zip(props, PROPS[:len(props)])}  # remove perm
                utt_noperm = substitute_single_letter(utt, sub_map)
                ltl_noperm = substitute_single_letter(ltl, sub_map)
                formula2data[(pattern_type, len(props))].append((utt_noperm, ltl_noperm))
            for (pattern_type, nprops), data in formula2data.items():
                data = list(dict.fromkeys(data))  # unique utt structures per formula in data. same order across runs
                for utt, ltl in data:
                    valid_iter_noperm.append((utt, ltl))
                    valid_meta_noperm.append((pattern_type, PROPS[:nprops]))
            valid_iter, valid_meta = valid_iter_noperm, valid_meta_noperm
        unique_formulas = set([(pattern_type, len(props)) for pattern_type, props in valid_meta])
        print(f"num of unique formulas: {len(unique_formulas)}")
        print(f"unique formulas:\n{unique_formulas}")
        print(f"test size after remove_perm: {len(valid_iter)} {len(valid_meta)}")
        print(f"num of lmks: {len(lmks)}")

        print("generating grounded train")
        dataset["train_iter"], dataset["train_meta"] = substitute_lmks(train_iter, train_meta, lmks, seed, add_comma, model)
        print("generating grounded valid")
        start_time = time.time()
        dataset["valid_iter"], dataset["valid_meta"] = substitute_lmks(valid_iter, valid_meta, lmks, seed+10000000, add_comma, model, nprops2lmkperms, nsamples)  # +10000000 avoid sampele lmks w/ same seeds as train set
        print(f"generate valid took: {(time.time()-start_time) / 60}")
        start_time = time.time()
        dataset["city"], dataset["seed_lmk"], dataset["remove_perm"], dataset["model"] = city, seed, remove_perm, model
        save_to_file(dataset, os.path.join(save_dpath, split_fname))
        print(f"saving took: {(time.time() - start_time) / 60}")

        print(f"new train, test size: {len(dataset['train_iter'])} {len(dataset['train_meta'])} {len(dataset['valid_iter'])} {len(dataset['valid_meta'])}")


def substitute_lmks(data, meta_data, lmks, seed, add_comma, model, nprops2lmkperms=None, nsamples=None):
    data_grounded, meta_data_grounded = [], []

    for idx, ((utt, ltl), (pattern_type, props)) in enumerate(zip(data, meta_data)):
        props = [prop.translate(str.maketrans('', '', string.punctuation)).strip() for prop in props]  # ['a,'] -> ['a']
        seed += idx  # diff seed to sample diff lmks for each utt-ltl pair
        if nsamples:  # permute lmks for valid_iter
            if len(props) not in nprops2lmkperms:
                nprops2lmkperms[len(props)] = list(permutations(lmks, len(props)))  # each perm contain len(props) lmks
            lmk_perms = nprops2lmkperms[len(props)]
            random.seed(seed)
            lmk_subs = random.sample(lmk_perms, nsamples)  # sample nsamples lmk perms
            # print(f"{idx}: {pattern_type} {props}\n{utt}\n{ltl}")
            # breakpoint()
        else:
            lmk_subs = [lmks]
        for lmk_sub in lmk_subs:
            utt_grounded, ltl_grounded, lmk_names = substitute_lmk(utt, ltl, lmk_sub, props, seed, add_comma, model)
            if utt == utt_grounded or ltl == ltl_grounded:
                raise ValueError(f"ERROR\n{utt}=={utt_grounded}\n{ltl}=={ltl_grounded}")
            data_grounded.append((utt_grounded, ltl_grounded))
            meta_data_grounded.append((utt, ltl, pattern_type, props, lmk_names, seed))
    return data_grounded, meta_data_grounded


def substitute_lmk(utt, ltl, lmks, props, seed, add_comma, model):
    if len(lmks) == len(props):  # already sample lmks from all perms for valid_iter
        lmk_names = lmks
    else:  # randomly sample lmks from complete lmk list for train_iter
        random.seed(seed)
        lmk_names = random.sample(lmks, len(props))

    if add_comma:
        sub_map = {prop: f"{name}," for prop, name in zip(props, lmk_names)}  # add comma after name for RER by GPT-3
    else:
        sub_map = {prop: name for prop, name in zip(props, lmk_names)}
    utt_grounded = substitute_single_letter(utt, sub_map).strip(",")  # if name at end of utt, remove extra comma if add

    sub_map = {prop: name_to_prop(name, model) for prop, name in zip(props, lmk_names)}
    ltl_grounded = substitute_single_letter(ltl, sub_map)

    return utt_grounded, ltl_grounded, lmk_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dpath", type=str, default="data/holdout_split_batch12_perm", help="dpath to all split datasets.")
    parser.add_argument("--env", type=str, default="osm", choices=["osm", "cleanup"], help="environment name.")
    parser.add_argument("--city", type=str, default="boston", help="city landmarks from 1 or all json files in data/osm/osm_lmks.")
    parser.add_argument("--seed", type=int, default=42, help="random seed to sample lmks.")
    parser.add_argument("--nsamples", type=int, default=15, help="number of samples per utt strucutre when perm lmks for valid set.")
    parser.add_argument("--remove_perm", action="store_true", help="True to remove permutations in validation set.")
    parser.add_argument("--add_comma", action="store_true", help="True to add comma after lmk name.")
    parser.add_argument("--model", type=str, default="copynet", choices=["lang2ltl", "copynet"], help="model name.")
    args = parser.parse_args()

    env_dpath = os.path.join("data", args.env)
    env_lmks_dpath = os.path.join(env_dpath, "lmks")
    # construct_lmk2prop(osm_lmks_dpath, args.model)  # for testing
    if args.env == "osm":
        if args.city == "all":
            cities = [os.path.splitext(fname)[0] for fname in os.listdir(env_lmks_dpath) if "json" in fname]
        else:
            cities = [args.city]
        for city in cities:
            city_lmks = list(load_from_file(os.path.join(env_lmks_dpath, f"{city}.json")).keys())
            construct_grounded_dataset(args.split_dpath, city_lmks, city, args.remove_perm, args.seed, args.nsamples, args.add_comma, args.model, env_dpath)
