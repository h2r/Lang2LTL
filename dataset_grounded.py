"""
Generate utterance-language grounded dataset with props substituted by lmks from a specific domain, e.g. OSM, CleanUp.
"""
import argparse
import os
import logging
from pathlib import Path
import time
import string
import re
import random
from pprint import pprint
from itertools import permutations
from collections import defaultdict

from utils import load_from_file, save_to_file, substitute_single_letter, remove_prop_perms, name_to_prop
from formula_sampler import PROPS
from dataset_symbolic import load_split_dataset


def rename_map_files(osm_lmks_dpath):
    """
    Change OSM map file names for better interpretation.
    e.g. Map00.json -> new_york_1.json
    One time change after pulling dataset from CopyNet repo.
    """
    mapid2envname = {
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
            if map_id in mapid2envname:
                env_name = mapid2envname[map_id]
                if "#" in env_name:
                    env_name = env_name.translate(str.maketrans('', '', string.punctuation))  # remove #
                new_map_fname = "_".join(env_name.lower().split(" ")) + ".json"
                new_map_fpath = os.path.join(osm_lmks_dpath, new_map_fname)
                os.rename(map_fpath, new_map_fpath)


def construct_lmk2prop(osm_lmks_dpath, model):
    """
    For testing conversion of every landmark name to proposition and inspecting OSM landmark names.
    Construct landmark names to propositions mapping.
    """
    lmk_fnames = [fname for fname in os.listdir(osm_lmks_dpath) if os.path.splitext(fname)[1] == ".json"]
    for lmk_fname in lmk_fnames:
        env_name = os.path.splitext(lmk_fname)[0]
        print(env_name)
        lmk2prop = {}
        lmk_fpath = os.path.join(osm_lmks_dpath, lmk_fname)
        lmk_json = load_from_file(lmk_fpath)
        lmk_names = lmk_json.keys()
        for lmk_name in lmk_names:
            lmk2prop[lmk_name] = name_to_prop(lmk_name, model)
        pprint(lmk2prop)
        print("\n")


def construct_grounded_dataset(split_dpath, lmks, env, remove_perm, seed, nsamples, add_comma, model, save_dpath):
    # TODO: update to work with multiple diverse REs instead of 1 lmk name for each OSM lmk, i.e. lmk2res.
    logging.info(f"Constructing grounded dataset for env: {env}")
    split_fnames = sorted([fname for fname in os.listdir(split_dpath) if "pkl" in fname])
    save_dpath = os.path.join(save_dpath, model, env)
    os.makedirs(save_dpath, exist_ok=True)

    nprops2lmkperms = {} if nsamples else None

    if len(lmks) > 40:
        random.seed(seed)
        lmks = random.sample(lmks, 40)  # down sample lmks to save memory during perms

    for split_fname in split_fnames:
        logging.info(f"{split_fname}")
        dataset = load_from_file(os.path.join(split_dpath, split_fname))
        train_iter, train_meta, valid_iter, valid_meta = dataset["train_iter"], dataset["train_meta"], dataset["valid_iter"], dataset["valid_meta"]

        logging.info(f"old train, test size: {len(train_iter)} {len(train_meta)} {len(valid_iter)} {len(valid_meta)}")
        logging.info(f"test size before remove prop perm: {len(valid_iter)} {len(valid_meta)}")
        if remove_perm:  # remove prop perms from valid_iter
            valid_iter, valid_meta = remove_prop_perms(valid_iter, valid_meta, PROPS)
        unique_formulas = set([(pattern_type, len(props)) for pattern_type, props in valid_meta])
        logging.info(f"num of unique formulas: {len(unique_formulas)}")
        logging.info(f"unique formulas:\n{unique_formulas}")
        logging.info(f"num of lmks: {len(lmks)}")
        logging.info(f"test size after remove prop perm: {len(valid_iter)} {len(valid_meta)}\n")

        logging.info("generating grounded train")
        dataset["train_iter"], dataset["train_meta"] = substitute_lmks(train_iter, train_meta, lmks, seed, add_comma, model)
        logging.info("generating grounded valid")
        start_time = time.time()
        dataset["valid_iter"], dataset["valid_meta"] = substitute_lmks(valid_iter, valid_meta, lmks, seed+10000000, add_comma, model, nprops2lmkperms, nsamples)  # +10000000 avoid sampele lmks w/ same seeds as train set
        logging.info(f"generate valid took: {(time.time()-start_time) / 60}")
        start_time = time.time()
        dataset["env"], dataset["seed_lmk"], dataset["remove_perm"], dataset["model"] = env, seed, remove_perm, model
        save_to_file(dataset, os.path.join(save_dpath, split_fname))
        logging.info(f"saving took: {(time.time() - start_time) / 60}")
        logging.info(f"{os.path.join(save_dpath, split_fname)}")

        logging.info(f"new train, test size: {len(dataset['train_iter'])} {len(dataset['train_meta'])} {len(dataset['valid_iter'])} {len(dataset['valid_meta'])}\n\n")


def substitute_lmks(data, meta_data, lmks, seed, add_comma, model, nprops2lmkperms=None, nsamples=None):
    data_grounded, meta_data_grounded = [], []

    for idx, ((utt, ltl), (pattern_type, props)) in enumerate(zip(data, meta_data)):
        props = [prop.translate(str.maketrans('', '', string.punctuation)).strip() for prop in props]  # ['a,'] -> ['a']
        props_norepeat = list(set(props))  # repeat props in restricted avoidance, e.g., [a, a, a]
        seed += idx  # diff seed to sample diff lmks for each utt-ltl pair
        if nsamples:  # permute lmks for valid_iter
            nprops_norepeat = len(props_norepeat)  # restricted avoidance repeat props, e.g., [a, a, a]
            if nprops_norepeat not in nprops2lmkperms:
                nprops2lmkperms[nprops_norepeat] = list(permutations(lmks, nprops_norepeat))  # each perm contain nprops_norepeat lmks
            lmk_perms = nprops2lmkperms[nprops_norepeat]
            random.seed(seed)
            lmk_subs = random.sample(lmk_perms, nsamples)  # sample nsamples lmk perms
            # print(f"{idx}: {pattern_type} {props}\n{utt}\n{ltl}")
            # breakpoint()
        else:
            lmk_subs = [lmks]
        for lmk_sub in lmk_subs:  # substitute multiple lmks in each (utt, ltl)
            utt_grounded, ltl_grounded, lmk_names = substitute_lmk(utt, ltl, lmk_sub, props_norepeat, seed, add_comma, model)
            if utt == utt_grounded:
                raise ValueError(f"ERROR\n{utt}\n==\n{utt_grounded}")
            if ltl == ltl_grounded:
                raise ValueError(f"ERROR\n{ltl}\n==\n{ltl_grounded}")
            utt_grounded.replace(", , ", ", ")  # remove duplicated commas. artifact of adding commas in compose and lmk substitution
            data_grounded.append((utt_grounded, ltl_grounded))
            meta_data_grounded.append((utt, ltl, pattern_type, props_norepeat, lmk_names, seed))
    return data_grounded, meta_data_grounded


def substitute_lmk(utt, ltl, lmks, props_norepeat, seed, add_comma, model):
    nprops_norepeat = len(props_norepeat)
    if len(lmks) == nprops_norepeat:  # already sample lmks from all perms for valid_iter
        lmk_names = lmks
    else:  # randomly sample lmks from complete lmk list for train_iter
        random.seed(seed)
        lmk_names = random.sample(lmks, nprops_norepeat)

    if add_comma:
        sub_map = {prop: f"{name}," for prop, name in zip(props_norepeat, lmk_names)}  # add comma after name for RER by GPT-3
    else:
        sub_map = {prop: name for prop, name in zip(props_norepeat, lmk_names)}
    utt_grounded = substitute_single_letter(utt, sub_map).strip(",")  # if name at end of utt, remove extra comma if add_comma=True

    sub_map = {prop: name_to_prop(name, model) for prop, name in zip(props_norepeat, lmk_names)}
    ltl_grounded = substitute_single_letter(ltl, sub_map)

    return utt_grounded, ltl_grounded, lmk_names


def construct_cleanup_res(env_res_dpath):
    sizes = ["", "big", "small"]
    colors = ["green", "blue", "red", "yellow"]
    shapes = [""]
    amenities = ["room", "region"]

    sem_lmks = {re.sub(' +', ' ', f"{size} {color} {shape} {amenity}".strip()): {} for color in colors for size in sizes for shape in shapes for amenity in amenities}

    os.makedirs(env_res_dpath, exist_ok=True)
    env_fpath = os.path.join(env_res_dpath, "cleanup.json")
    save_to_file(sem_lmks, env_fpath)
    logging.info(f"generated {len(sem_lmks)} lmks for CleanUp\nfrom\nsizes: {sizes}\ncolors: {colors}\nshapes: {shapes}\namenities: {amenities}")


def generate_prompts_from_grounded_split_dataset(split_fpath, prompt_dpath, nexamples, seed):
    """
    :param split_fpath: pickle file containing train, test split for a holdout type of a grounded dataset.
    :param prompt_dpath: directory to save constructed prompts.
    :param nexamples: number of examples for 1 formula.
    :param seed: seed to sample landmarks for a formula.
    """
    train_iter, train_meta, _, _ = load_split_dataset(split_fpath)

    meta2data = defaultdict(list)
    for idx, ((utt, ltl), (_, _, pattern_type, props, _, _, _)) in enumerate(zip(train_iter, train_meta)):
        meta2data[(pattern_type, len(props))].append((utt, ltl))
    sorted(meta2data.items(), key=lambda kv: kv[0])

    prompt = "Your task is to first find referred landmarks from a given list then use them as propositions to translate English utterances to linear temporal logic (LTL) formulas.\n\n"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dpath", type=str, default="data/holdout_split_batch12_perm", help="dpath to all split datasets.")
    parser.add_argument("--domain", type=str, default="osm", choices=["osm", "cleanup"], help="domain name.")
    parser.add_argument("--env", type=str, default="boston", choices=["boston", "all", "cleanup"], help="env lmks from 1 or all json files in data/osm/lmks or data/clenaup/lmks.")
    parser.add_argument("--model", type=str, default="lang2ltl", choices=["lang2ltl", "copynet"], help="model name.")
    parser.add_argument("--seed", type=int, default=42, help="random seed to sample lmks.")
    parser.add_argument("--nsamples", type=int, default=2, help="number of samples per utt-ltl pair when perm lmks for valid set.")
    parser.add_argument("--remove_perm", action="store_false", help="True to keep prop perms in valid set. Default True.")
    parser.add_argument("--add_comma", action="store_true", help="True to add comma after lmk name.")
    parser.add_argument("--env_prompt", type=str, default="boston", help="env dataset to generate full translation prompt.")
    parser.add_argument("--nexamples", action="store", type=int, nargs="+", default=[1], help="number of examples per formula in prompt.")
    parser.add_argument("--seed_prompt", type=int, default=42, help="random seed to sample prompt examples.")
    parser.add_argument("--cleanup_re", action="store_true", help="True if construct referring expression json for CleanUp lmks.")
    args = parser.parse_args()

    domain_dpath = os.path.join("data", args.domain)
    domain_res_dpath = os.path.join(domain_dpath, "ref_exps", "diverse_res")
    model_dpath = os.path.join(domain_dpath, f"{args.model}_diverse-re_downsampled")
    os.makedirs(model_dpath, exist_ok=True)
    # construct_lmk2prop(osm_lmks_dpath, args.model)  # for testing

    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(model_dpath, f'log_gen_grounded_{args.env}.log'),
                                                mode='w'),
                            logging.StreamHandler()
                        ]
    )

    if args.cleanup_re:  # construct RE json file for CleanUp World
        construct_cleanup_res(domain_res_dpath)

    if args.env == "all":
        envs = [os.path.splitext(fname)[0][40:] for fname in os.listdir(domain_res_dpath) if fname.endswith("csv")]  # csv for diverse RE
        # envs = [os.path.splitext(fname)[0] for fname in os.listdir(domain_res_dpath) if fname.endswith("json")]  # json for simple RE (lmk name)
    else:
        envs = [args.env]

    for env in envs:
        lmk2res = {entry[0]: entry[2:] for entry in load_from_file(os.path.join(domain_res_dpath, f"paraphrased-res-gpt4_filtered-attributes_{env}.csv"))}
        # env_res = [name.replace(u'\xa0', u' ') for name in list(load_from_file(os.path.join(domain_res_dpath, f"{env}.json")).keys())]  # remove unicode space \xa0 or NBSP

        # construct_grounded_dataset(args.split_dpath, env_res, env, args.remove_perm, args.seed, args.nsamples, args.add_comma, args.model, domain_dpath)

        # Construct full translation prompts from grounded dataset for a given env
        if env == args.env_prompt:
            prompt_dpath = os.path.join(domain_dpath, "full_translation_prompt_diverse-re", args.env_prompt)
            os.makedirs(prompt_dpath, exist_ok=True)
            grounded_split_dpath = os.path.join(model_dpath, args.env_prompt)
            grounded_split_fpaths = [os.path.join(grounded_split_dpath, grounded_split_fname) for grounded_split_fname in os.listdir(grounded_split_dpath) if grounded_split_fname.endswith("pkl")]
            for grounded_split_fpath in grounded_split_fpaths:
                for nexamples in args.nexamples:
                    generate_prompts_from_grounded_split_dataset(grounded_split_fpath, prompt_dpath, nexamples, args.seed_prompt)
