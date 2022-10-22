import os
import json
import dill
import numpy as np


def build_placeholder_map(name_entities):
    placeholder_map, placeholder_map_inv = {}, {}
    letter = chr(ord("A") - 1)
    for ne in name_entities:
        letter = chr(ord(letter) + 1)  # increment placeholder using its ascii value
        placeholder_map[ne] = letter
        placeholder_map_inv[letter] = ne
    return placeholder_map, placeholder_map_inv


def substitute(input_strs, substitute_maps):
    """
    Substitute every occurrence of key in the input string by its corresponding value in substitute_maps.
    :param input_strs: input strings
    :param substitute_maps: map substring to substitutions
    :return: substituted string and corresponding substitutions
    """
    output_strs, str2subs = [], {}
    if len(substitute_maps) == 1:  # same substitute map for all input strings
        for input_str in input_strs:
            out_str, subs_done = substitute_single(input_str, substitute_maps[0])
            output_strs.append(out_str)
            str2subs[out_str] = subs_done
    else:
        for input_str, sub_map in zip(input_strs, substitute_maps):
            out_str, subs_done = substitute_single(input_str, sub_map)
            output_strs.append(out_str)
            str2subs[out_str] = subs_done
    return output_strs, str2subs


def substitute_single(input_str, sub_map):
    sub_map = sorted(sub_map.items(), key=lambda kv: len(kv[0]), reverse=True)  # start substitution with long strings
    subs_done = set()  # sub key once when diff keys map to same val, e.g. {green one: green room, green: green room}
    for k, v in sub_map:
        if k not in input_str:
            print(f"Name entity {k} not found in input string: {input_str}")
        else:
            if v not in subs_done:
                subs_done.add(v)
                input_str = input_str.replace(k, v)
    return input_str.strip(), subs_done


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
    else:
        raise ValueError(f"ERROR: file type {ftype} not recognized")


def load_from_file(fpath):
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
