import os
import json
import dill
import csv
from collections import defaultdict
import logging
import numpy as np


def build_placeholder_map(name_entities):
    placeholder_map, placeholder_map_inv = {}, {}
    letter = "A"
    for ne in name_entities:
        placeholder_map[ne] = letter
        placeholder_map_inv[letter] = "_".join(ne.split()).strip()
        letter = chr(ord(letter) + 1)  # increment placeholder using its ascii value
    return placeholder_map, placeholder_map_inv


def substitute(input_strs, substitute_maps):
    """
    Substitute every occurrence of key in the input string by its corresponding value in substitute_maps.
    :param input_strs: input strings
    :param substitute_maps: map substring to substitutions
    :return: substituted strings and their corresponding substitutions
    """
    output_strs, subs_per_str = [], []
    for input_str, sub_map in zip(input_strs, substitute_maps):
        out_str, subs_done = substitute_single(input_str, sub_map)
        output_strs.append(out_str)
        subs_per_str.append(subs_done)
    return output_strs, subs_per_str


def substitute_single(input_str, sub_map):
    """
    Substitute words and phrases from a single utterance.

    Assume no swaps in `sub_map`, e.g. {key: val, val: key}
    TODO: handle substitution map: {green one: green room, green place: green room, green: green room}
    TODO: handle substitution map: {a: b, b: a}
    """
    sub_map = sorted(sub_map.items(), key=lambda kv: len(kv[0]), reverse=True)  # start substitution with long strings
    subs_done = set()  # sub key once when diff keys map to same val, e.g. {green one: green room, green: green room}

    for k, v in sub_map:
        if k not in input_str:
            logging.info(f"Name entity {k} not found in input string: {input_str}")
        else:
            # k_v_overlap = False  # to handle substitution map {green one: green room, green: green room}
            # for sub_done in subs_done:
            #     if k in sub_done:
            #         k_v_overlap = True
            # if not k_v_overlap:
            if v not in subs_done:  # TODO: 'the name' sub first then 'name' sub again. redundant if same grounding, wrong otherwise
                subs_done.add(v)
                input_str = input_str.replace(k, v)
    return input_str.strip(), subs_done


def substitute_single_fix(input_str, sub_map):
    """
    Correcting substitute_single with
    e.g. input_str='go to a then go to b', sub_map={'a': 'b', 'b': 'a'}
    return='go to a then go to a'

    Require `input_str` to be normalized, i.e. all punctuations removed.

    TODO: only work with substitute single words, e.g. a, b, c, etc, not phrases, e.g. green one -> green room.
    """
    input_str_list = input_str.split(" ")

    sub_map = sorted(sub_map.items(), key=lambda kv: len(kv[0]), reverse=True)  # start substitution with long strings
    subs_done = set()  # sub key once when diff keys map to same val, e.g. {green one: green room, green: green room}

    # Record indices of all keys in sub_map in *original* input_str.
    key2indices = defaultdict(list)
    for k, _ in sub_map:
        key2indices[k] = [idx for idx, word in enumerate(input_str_list) if word == k]

    # Loop through indices and keys to replace each key with value.
    for k, v in sub_map:
        indices = key2indices[k]
        for idx in indices:
            input_str_list[idx] = v
            subs_done.add(v)

    return ' '.join(input_str_list).strip(), subs_done


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


def save_to_file(data, fpth):
    ftype = os.path.splitext(fpth)[-1][1:]
    if ftype == 'pkl':
        with open(fpth, 'wb') as wfile:
            dill.dump(data, wfile)
    elif ftype == 'txt':
        with open(fpth, 'w') as wfile:
            wfile.write(data)
    elif ftype == 'json':
        with open(fpth, 'w') as wfile:
            json.dump(data, wfile)
    elif ftype == 'csv':
        with open(fpth, 'w', newline='') as wfile:
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


def equal(item1, item2):
    """
    For testing.
    """
    if isinstance(item1, list) and isinstance(item2, list):
        for elem1, elem2 in zip(item1, item2):
            assert elem1 == elem2, f"{elem1} != {elem2}"
    elif isinstance(item1, dict) and isinstance(item1, dict):
        assert len(item1) == len(item2)
        for k in item1:
            assert np.all(item1[k] == item2[k]), f"{item1[k]} != {item2[k]}"


if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)

    input_utterances = [
        "Go to Heng Thai, after that visit Providence Palace, but before Providence Palace, go to Chinatown"
    ]
    save_to_file(input_utterances, os.path.join("data", "input.pkl"))

    input_utterances_loaded = load_from_file(os.path.join("data", "input.pkl"))
    equal(input_utterances, input_utterances_loaded)

    true_ltls = [
        "F ( Heng Thai & F ( Chinatown & F ( Providence Palace ) )"
    ]
    save_to_file(true_ltls, os.path.join("data", "true_ltls.pkl"))

    true_ltls_loaded = load_from_file(os.path.join("data", "true_ltls.pkl"))
    equal(true_ltls, true_ltls_loaded)

    e2e_prompt = \
        "English: Go to Bookstore then to Science Library\n" \
        "Landmarks: Bookstore | Science Library\n" \
        "LTL: F ( Bookstore & F ( Science Library ) )\n\n" \
        "English: Go to Bookstore then reach Science Library\n" \
        "Landmarks: Bookstore | Science Library\n" \
        "LTL: F ( Bookstore & F ( Science Library ) )\n\n" \
        "English: Find Bookstore then go to Science Library\n" \
        "Landmarks: Bookstore | Science Library\n" \
        "LTL: F ( Bookstore & F ( Science Library ) )\n\n" \
        "English: Go to Burger Queen then to black stone park, but after KFC\n" \
        "Landmarks: Burger Queen | black stone park | KFC\n" \
        "LTL: F ( Burger Queen & F ( KFC & F ( black stone park ) )\n\n" \
        "English: Go to Burger Queen then to black stone park; go to KFC before black stone park and after Burger Queen\n"\
        "Landmarks: Burger Queen | black stone park | KFC\n" \
        "LTL: F ( Burger Queen & F ( KFC & F ( black stone park ) )\n\n" \
        "English: "
    save_to_file(e2e_prompt, os.path.join("data", "e2e_prompt.txt"))

    e2e_prompt_loaded = load_from_file(os.path.join("data", "e2e_prompt.txt"))
    equal(e2e_prompt, e2e_prompt_loaded)

    ner_prompt = \
        "English: Go to Bookstore then to Science Library\n" \
        "Landmarks: Bookstore | Science Library\n\n" \
        "English: Go to Bookstore then reach Science Library\n" \
        "Landmarks: Bookstore | Science Library\n\n" \
        "English: Find Bookstore then go to Science Library\n" \
        "Landmarks: Bookstore | Science Library\n\n" \
        "English: Go to Burger Queen then to black stone park, but after KFC\n" \
        "Landmarks: Burger Queen | black stone park | KFC\n\n" \
        "English: Go to Burger Queen then to black stone park; go to KFC before black stone park and after Burger Queen\n" \
        "Landmarks: Burger Queen | black stone park | KFC\n\n" \
        "English: "
    save_to_file(ner_prompt, os.path.join("data", "ner_prompt.txt"))

    ner_prompt_loaded = load_from_file(os.path.join("data", "ner_prompt.txt"))
    equal(ner_prompt, ner_prompt_loaded)

    trans_prompt = \
        "English: Go to A then to B\nLTL: F ( A & F ( B ) )\n\n" \
        "English: Go to A then reach B\nLTL: F ( A & F ( B ) )\n\n" \
        "English: Find A then go to B\nLTL: F ( A & F ( B ) )\n\n" \
        "English: Go to A then to B, but after C\nLTL: F ( A & F ( C & F ( B ) )\n\n" \
        "English: Go to A then to B; go to C before B and after A\nLTL: F ( A & F ( C & F ( B ) )\n\n" \
        "English: "
    save_to_file(trans_prompt, os.path.join("data", "trans_prompt.txt"))

    trans_prompt_loaded = load_from_file(os.path.join("data", "trans_prompt.txt"))
    equal(trans_prompt, trans_prompt_loaded)

    name2embed = {
        "restaurant": np.random.rand(1, 2048),
        "mall": np.random.rand(1, 2048),
    }
    save_to_file(name2embed, os.path.join("data", "name_embed.pkl"))

    name2embed_loaded = load_from_file(os.path.join("data", "name_embed.pkl"))
    equal(name2embed, name2embed_loaded)
