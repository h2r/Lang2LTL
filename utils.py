import os
from pathlib import Path
import json
import dill
import csv
import string
from collections import defaultdict
import numpy as np
import random
import nltk
import tiktoken

from gpt import GPT3


def build_placeholder_map(name_entities, convert_rule, props):
    # breakpoint()

    placeholder_map, placeholder_map_inv = {}, {}
    for name, letter in zip(name_entities, props[:len(name_entities)]):
        placeholder_map[name] = letter
        placeholder_map_inv[letter] = name_to_prop(name, convert_rule)
    return placeholder_map, placeholder_map_inv


def substitute(input_strs, substitute_maps, is_utt):
    """
    Substitute every occurrence of key in the input string by its corresponding value in substitute_maps.
    :param input_strs: input strings
    :param substitute_maps: map substring to substitutions
    :param is_utt: True if input_strs are utts; False if input_strs are LTL formulas
    :return: substituted strings and their corresponding substitutions
    """
    # breakpoint()

    output_strs, subs_per_str = [], []
    for input_str, sub_map in zip(input_strs, substitute_maps):
        if is_utt:
            out_str, subs_done = substitute_single_word(input_str, sub_map)
        else:
            out_str = substitute_single_letter(input_str, sub_map)
            subs_done = set()
        # out_str = out_str.translate(str.maketrans('', '', ',.'))  # remove comma, period since sym translation module finetuned on utts w/o puns
        output_strs.append(out_str)
        subs_per_str.append(subs_done)
    # breakpoint()

    return output_strs, subs_per_str


def substitute_single_word(in_str, sub_map):
    """
    Substitute words and phrases to words or phrases in a single utterance.
    Assume numbers are not keys of sub_map.
    """
    # breakpoint()

    sub_map = sorted(sub_map.items(), key=lambda kv: len(kv[0]), reverse=True)  # start substitution with long strings
    subs_done = set()

    # swap every k with a unique number
    for n, (k, v) in enumerate(sub_map):
        in_str = in_str.replace(k, f"[{n}]")  # escape number

    # swap every number with corresponding v
    for n, (k, v) in enumerate(sub_map):
        in_str = in_str.replace(f"[{n}]", v)  # escape number
        subs_done.add(v)

    # breakpoint()

    return in_str.strip(), subs_done


def substitute_single_letter(in_str, sub_map):
    """
    :param in_str: input string can utterance or LTL formula.
    :param sub_map: dict maps letters to noun phrases for lmk names (for utterance) or letters (for LTL formula).

    Substitute English letters to letters, words or phrases in a single utterance.
    e.g. input_str="go to a then go to b", sub_map={'a': 'b', 'b': 'a'} -> return="go to a then go to a"

    Require `input_str` to be normalized, i.e. all punctuations removed. If not, "go to a. after that go to b." not tokenized correctly
    Only work with letters, e.g. a, b, c, etc, not phrases, e.g. green one -> green room.
    """
    in_str_list = nltk.word_tokenize(in_str)
    # in_str_list = in_str.split(" ")
    sub_map = sorted(sub_map.items(), key=lambda kv: len(kv[0]), reverse=True)  # start substitution with long strings

    # Record indices of all keys in sub_map in *original* input_str.
    key2indices = defaultdict(list)
    for k, _ in sub_map:
        key2indices[k] = [idx for idx, word in enumerate(in_str_list) if word == k]

    # Loop through indices and keys to replace each key with value.
    for k, v in sub_map:
        indices = key2indices[k]
        for idx in indices:
            in_str_list[idx] = v

    return ' '.join(in_str_list).strip()


def remove_prop_perms(data, meta, all_props):
    formula2data = defaultdict(list)  # expensive: entire symbolic iter, meta in memory
    for (utt, ltl), (pattern_type, props) in zip(data, meta):
        props = list(props)
        if props.count(props[0]) == len(props):  # restricted avoidance formulas all props are the same
            props_noperm = [all_props[0]]*len(props)  # b, b -> a, a
        else:
            props_noperm = all_props[:len(props)]
        sub_map = {old_prop: new_prop for old_prop, new_prop in zip(props, props_noperm)}
        utt_noperm = substitute_single_letter(utt, sub_map)
        ltl_noperm = substitute_single_letter(ltl, sub_map)
        formula2data[(pattern_type, tuple(props_noperm))].append((utt_noperm, ltl_noperm))

    data_noperm, meta_noperm = [], []
    for (pattern_type, props_noperm), data in formula2data.items():
        data = list(dict.fromkeys(data))  # remove duplicate utt structures per formula. same order across runs
        for utt, ltl in data:
            data_noperm.append((utt, ltl))
            meta_noperm.append((pattern_type, list(props_noperm)))
    return data_noperm, meta_noperm


def sample_small_dataset(data_fpath):
    """
    Sample smaller dataset for testing, mostly full pipeline with GPT-3.
    1 utt per unique formula (type, nprops). Permuted props replaced by non-permute props starting at a.
    """
    dataset = load_from_file(data_fpath)
    data, meta = dataset["valid_iter"], dataset["valid_meta"]

    formula2data = defaultdict(list)  # expensive: entire grounded iter, meta in memory
    for (utt_grounded, ltl_grounded), (utt, ltl, pattern_type, props, lmk_names, seed) in zip(data, meta):
        props = list(props)
        formula2data[(pattern_type, len(props))].append((utt_grounded, ltl_grounded, utt, ltl, pattern_type, tuple(props), lmk_names, seed))

    data_small, meta_small = [], []
    for _, formula_data in formula2data.items():
        random.seed(42)
        formula_data_random = random.sample(formula_data, 1)[0]
        utt_grounded, ltl_grounded, utt, ltl, pattern_type, props, lmk_names, seed = formula_data_random
        data_small.append((utt_grounded, ltl_grounded))
        meta_small.append((utt, ltl, pattern_type, props, lmk_names, seed))
    dataset["valid_iter"], dataset["valid_meta"] = data_small, meta_small
    save_fpath = os.path.join(os.path.dirname(data_fpath), f"small_{os.path.basename(data_fpath)}")
    save_to_file(dataset, save_fpath)


def name_to_prop(name, convert_rule):
    """
    :param name: name, e.g. Canal Street, TD Bank.
    :param convert_rule: identifier for conversion rule.
    :return: proposition that corresponds to input landmark name and is compatible with Spot.
    """
    if convert_rule == "lang2ltl":
        return "_".join(name.translate(str.maketrans('/()-–', '     ', "'’,.!?")).lower().split())
    elif convert_rule == "copynet":
        return f"lm( {name} )lm"
    elif convert_rule == "cleanup":
        return "_".join(name.split()).strip()
    else:
        raise ValueError(f"ERROR: unrecognized conversion rule: {convert_rule}")


def shorten_prop(prop):
    return '_'.join([word[:2] for word in prop.split('_')])


def deserialize_props_str(props_str):
    """
    Deserialize json string of propositions.
    :param props_str: "('a',)", "('a', 'b')", "['a',]", "['a', 'b']",
    :return: ['a'], ['a', 'b'], ['a'], ['a', 'b']
    """
    props = [prop.translate(str.maketrans('', '', string.punctuation)).strip() for prop in list(props_str.strip("()[]").split(", "))]
    return props


def props_in_formula(formula, feasible_props):
    return [prop for prop in feasible_props if prop in formula]


def props_in_utt(utt, feasible_props):
    return [word.strip() for word in utt.split() if word.strip() in feasible_props]


def save_to_file(data, fpth, mode=None):
    ftype = os.path.splitext(fpth)[-1][1:]
    if ftype == 'pkl':
        with open(fpth, mode if mode else 'wb') as wfile:
            dill.dump(data, wfile)
    elif ftype == 'txt':
        with open(fpth, mode if mode else 'w') as wfile:
            wfile.write(data)
    elif ftype == 'json':
        with open(fpth, mode if mode else 'w') as wfile:
            json.dump(data, wfile)
    elif ftype == 'csv':
        with open(fpth, mode if mode else 'w', newline='') as wfile:
            writer = csv.writer(wfile)
            writer.writerows(data)
    else:
        raise ValueError(f"ERROR: file type {ftype} not recognized")


def load_from_file(fpath, noheader=True):
    ftype = os.path.splitext(fpath)[-1][1:]
    if ftype == 'pkl':
        with open(fpath, 'rb') as rfile:
            out = dill.load(rfile)
    elif ftype == 'txt':
        with open(fpath, 'r') as rfile:
            if 'prompt' in fpath:
                out = "".join(rfile.readlines())
            else:
                out = [line[:-1] for line in rfile.readlines()]
    elif ftype == 'json':
        with open(fpath, 'r') as rfile:
            out = json.load(rfile)
    elif ftype == 'csv':
        with open(fpath, 'r') as rfile:
            csvreader = csv.reader(rfile)
            if noheader:
                fileds = next(csvreader)
            out = [row for row in csvreader]
    else:
        raise ValueError(f"ERROR: file type {ftype} not recognized")
    return out


def append_ids_to_path(pth, appends, id_trues, id_falses):
    """
    Append identifier at the end of file or directory path based on `append`.
    """
    if not isinstance(appends, list):
        appends, id_trues, id_falses = [appends], [id_trues], [id_falses]
    for append, id_true, id_false in zip(appends, id_trues, id_falses):
        if os.path.isfile(pth):
            base_pth = os.path.join(os.path.dirname(pth), f"{Path(pth).stem}")
            pth = f"{base_pth}_{id_true}{os.path.splitext(pth)[1]}" if append else f"{base_pth}_{id_false}{os.path.splitext(pth)[1]}"
        else:
            pth = f"{pth}_{id_true}" if append else f"{pth}_{id_false}"
    return pth


def remove_id_from_path(pth, identifier):
    fname = [fname_sub for fname_sub in Path(pth).stem.split("_") if fname_sub != identifier]
    return os.path.join(os.path.dirname(pth), f"{'_'.join(fname)}{os.path.splitext(pth)[1]}")


def prefix_to_infix(formula):
    """
    :param formula: LTL formula string in prefix order
    :return: LTL formula string in infix order
    Spot's prefix parser uses i for implies and e for equivalent. https://spot.lre.epita.fr/ioltl.html#prefix
    """
    BINARY_OPERATORS = {"&", "|", "U", "W", "R", "->", "i", "<->", "e"}
    UNARY_OPERATORS = {"!", "X", "F", "G"}
    formula_in = formula.split()
    stack = []  # stack

    while formula_in:
        op = formula_in.pop(-1)
        if op == ">":
            op += formula_in.pop(-1)  # implication operator has 2 chars, ->
        if formula_in and formula_in[-1] == "<":
            op += formula_in.pop(-1)  # equivalent operator has 3 chars, <->

        if op in BINARY_OPERATORS:
            formula_out = "(" + stack.pop(0) + " " + op + " " + stack.pop(0) + ")"
            stack.insert(0, formula_out)
        elif op in UNARY_OPERATORS:
            formula_out = op + "(" + stack.pop(0) + ")"
            stack.insert(0, formula_out)
        else:
            stack.insert(0, op)

    return stack[0]


def count_params(model):
    """
    :param model: a PyTorch module
    :return: the number of trainable paramters in input PyTorch module
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_ntokens(text, model):
    """
    https://github.com/openai/tiktoken
    :param model: GPT model name, e.g. gpt-4, text-davinci-003
    :param text: English text to count tokens
    :return: number of tokens by OpenAI tokenizer
    """
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)


def count_prompt_ntokens(prompt_fpath, model):
    prompt_text = load_from_file(prompt_fpath)
    print(f"ntokens for prompt: {prompt_fpath}\n{count_ntokens(prompt_text, model)}")


def count_lmk_ntokens(model):
    """
    Count total number of tokens in all lmks in an OSM city
    """
    env_lmks_dpath = os.path.join("data", "osm", "lmks")
    cities = [os.path.splitext(fname)[0] for fname in os.listdir(env_lmks_dpath) if "json" in fname]
    for city in cities:
        lmks = list(load_from_file(os.path.join(env_lmks_dpath, f"{city}.json")).keys())
        lmk_list = ', '.join(lmks)
        print(f"{city}:\t {lmk_list}\n{count_ntokens(lmk_list, model)}")  # https://platform.openai.com/tokenizer
