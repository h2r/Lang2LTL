import os
import dill
import csv
import spacy


def ner_spacy():
    nlp = spacy.load("en_core_web_sm")
    # doc = nlp("Go to Burger Queen then to cafe on Thayer")
    doc = nlp("Go to Starbucks then to war memorial")

    for token in doc:
        print(token, token.tag_, token.pos_, spacy.explain(token.tag_))

    spacy.displacy.serve(doc, style="dep")


def ner_llm():
    with open("data/providence_500.csv", mode='r') as rf:
        csvfile = csv.reader(rf)
        for lines in csvfile:
            out_str = f"English: {lines[1]}; LTL: {lines[2]}"
            print(out_str)


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
            out = "".join(rfile.readlines())

    else:
        raise ValueError(f"ERROR: file type {ftype} not recognized")
    return out


if __name__ == '__main__':
    input_utterances = [
        "Go to Heng Thai, after that visit Providence Palace, but before Providence Palace, go to Chinatown"
    ]
    save_to_file(input_utterances, os.path.join("data", "input.pkl"))

    # input_utterances = load_from_file(os.path.join("data", "input.pkl"))
    # print(f"input_utterances:\n{input_utterances}")


    true_ltls = [
        "F ( Heng Thai & F ( Chinatown & F ( Providence Palace ) )"
    ]
    save_to_file(true_ltls, os.path.join("data", "true_ltls.pkl"))

    # true_ltls = load_from_file(os.path.join("data", "true_ltls.pkl"))
    # print(f"true_ltls:\n{true_ltls}")


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

    # e2e_prompt = load_from_file(os.path.join("data", "e2e_prompt.txt"))
    # print(f"e2e_prompt:\n{e2e_prompt}")


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

    # ner_prompt = load_from_file(os.path.join("data", "ner_prompt.txt"))
    # print(f"ner_prompt:\n{ner_prompt}")


    trans_prompt = \
        "English: Go to a then to b\nLTL: F ( a & F ( b ) )\n\n" \
        "English: Go to a then reach b\nLTL: F ( a & F ( b ) )\n\n" \
        "English: Find a then go to b\nLTL: F ( a & F ( b ) )\n\n" \
        "English: Go to a then to b, but after c\nLTL: F ( a & F ( c & F ( b ) )\n\n" \
        "English: Go to a then to b; go to c before b and after a\nLTL: F ( a & F ( c & F ( b ) )\n\n" \
        "English: "
    save_to_file(trans_prompt, os.path.join("data", "trans_prompt.txt"))

    # trans_prompt = load_from_file(os.path.join("data", "trans_prompt.txt"))
    # print(f"trans_prompt:\n{trans_prompt}")
