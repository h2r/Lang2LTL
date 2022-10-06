import os
import dill
import numpy as np
import json

def build_placeholder_map(name_entities):
    placeholder_map = {}
    letter = "A"
    for idx, ne in enumerate(name_entities):
        placeholder_map[ne] = chr(ord(letter) + idx)  # increment placeholder using its ascii value
    return placeholder_map


def substitute(input_strs, placeholder_maps):
    output_strs = []
    for input_str, placeholder_map in zip(input_strs, placeholder_maps):
        for k, v in placeholder_map.items():
            input_str_sub = input_str.replace(k, v)
            if input_str_sub == input_str:  # name entity not found in utterance
                raise ValueError(f"Name entity {k} not found in input utterance {input_str}")
            else:
                input_str = input_str_sub
        output_strs.append(input_str)
    return output_strs


def save_to_file(data, fpth):
    ftype = os.path.splitext(fpth)[-1][1:]
    if ftype == 'pkl':
        with open(fpth, 'wb') as wfile:
            dill.dump(data, wfile)
    elif ftype == 'txt':
        with open(fpth, 'w') as wfile:
            wfile.write(data)
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
                out = [ line[:-1] for line in rfile.readlines()]
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

def clean_str(string):
    def clean(string):
        string = string.strip()
        if string[:2] == '\n':
            string = string[2:]
        elif string[-2:] == '\n':
            string = string[:-2]
        return string
    buffer = string
    while True:
        string = clean(string)
        if buffer == string:
            break
        buffer = string
    return string

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
