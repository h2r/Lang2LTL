import argparse

from gpt3 import GPT3
from utils import load_from_file


def run_exp_e2e():
    input_utterances = load_from_file(args.input)
    true_ltls = load_from_file(args.true_output)
    e2e_prompt = load_from_file(args.e2e_prompt)

    model = GPT3()

    queries = [e2e_prompt+utt+"\nLTL:" for utt in input_utterances]
    out_ltls = [model.translate(query) for query in queries]

    acc = evalulate(out_ltls, true_ltls)
    print(acc)


def run_exp_modular():
    # Read data
    input_utterances = load_from_file(args.input)
    ner_prompt = load_from_file(args.ner_prompt)
    trans_prompt = load_from_file(args.trans_prompt)
    true_ltls = load_from_file(args.true_output)

    # Load modules and construct queries
    if args.ground == 'gpt3':
        ground_module = GPT3()
    # elif args.ground == 'bert':
    #     ground_module = BERT()
    else:
        raise ValueError("ERROR: grounding module not recognized")

    if args.ner == 'gpt3':
        ner_module = GPT3()
        ner_queries = [ner_prompt+utt+"\nLandmarks:" for utt in input_utterances]
    # elif args.ner == 'bert':
    #     ner_module = BERT()
    else:
        raise ValueError("ERROR: NER module not recognized")

    if args.trans == 'gpt3':
        trans_module = GPT3()
        trans_queries = [trans_prompt+utt+"\nLTL:" for utt in input_utterances]
    # elif args.trans == 's2s_sup':
    #     trans_module = Seq2Seq()
    else:
        raise ValueError("ERROR: translation module not recognized")

    # Query module
    if args.trans == 's2s_sup':
        placeholder_maps = ner_task(ner_module, ner_queries)
        trans_queries = substitue(input_utterances, placeholder_maps)

    output_ltls = trans_task(trans_module, trans_queries)

    # Evaluate system output
    acc = evalulate(output_ltls, true_ltls)
    print(acc)


def ner_task(ner_module, ner_queries):
    return [ner_module.extract(query) for query in ner_queries]


def trans_task(trans_module, trans_queries):
    return [trans_module.translate(query) for query in trans_queries]


def substitue(input_utterances, placeholder_maps):
    input_symbolics = []
    for utt, placeholder_map in zip(input_utterances, placeholder_maps):
        for name, symbol in placeholder_map.items():
            utt_sub = utt.replace(name, symbol)
            if utt_sub == utt:  # name entity not found in utterance
                raise ValueError(f"Name entity {name} not found in input utterance {utt}")
            else:
                utt = utt_sub
        input_symbolics.append(utt)
    return input_symbolics


def evalulate(output_ltls, true_ltls):
    accs = []
    for out_ltl in output_ltls:
        accs.append(out_ltl in true_ltls)
    return sum(accs) / len(accs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--e2e_gpt3', type=bool, default=False, help="Solve translation task end-to-end using GPT-3")
    parser.add_argument('--ground', type=str, default='gpt3', help='grounding module: gpt3, bert')
    parser.add_argument('--ner', type=str, default='gpt3', help='NER module: gpt3, bert')
    parser.add_argument('--trans', type=str, default='gpt3', help='translation module: gpt3, s2s_sup, s2s_weaksup')
    parser.add_argument('--input', type=str, default='data/input.pkl', help='file path to input utterances')
    parser.add_argument('--true_output', type=str, default='data/true_output.pkl', help='file path to true LTLs')
    parser.add_argument('--e2e_prompt', type=str, default='data/e2e_prompt.txt', help='file path to end-to-end prompt')
    parser.add_argument('--ner_prompt', type=str, default='data/ner_prompt.txt', help='file path to NER prompt')
    parser.add_argument('--trans_prompt', type=str, default='data/trans_prompt.txt', help='file path to trans prompt')
    args = parser.parse_args()

    if args.e2e_gpt3:
        run_exp_e2e()
    else:
        run_exp_modular()
