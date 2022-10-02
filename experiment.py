import argparse

from gpt3 import GPT3
from utils import load_from_file


def run_exp():
    if args.e2e_gpt3:
        output_ltls = run_exp_e2e()
    else:
        output_ltls = run_exp_modular()

    # ground_prompt = load_from_file(args.ground_prompt)
    # known_names = load_from_file(args.known_names)

    if args.ground == 'gpt3':
        ground_module = GPT3()
        # ground_queries = [ground_prompt+utt for utt in input_utterances]
    # elif args.ground == 'bert':
    #     ground_module = BERT()
    else:
        raise ValueError("ERROR: grounding module not recognized")

    # ground_maps = ground_task(ground_module, known_names, ground_queries)

    # true_trajs = load_from_file(args.true_trajs)
    # acc_plan = plan(output_ltls, true_trajs, ground_maps)
    # print(acc_plan)


def run_exp_e2e():
    input_utterances = load_from_file(args.input)
    true_ltls = load_from_file(args.true_ltls)
    e2e_prompt = load_from_file(args.e2e_prompt)

    model = GPT3()

    queries = [e2e_prompt+utt+"\nLTL:" for utt in input_utterances]
    output_ltls = [model.translate(query) for query in queries]

    acc_lang = evalulate_lang(output_ltls, true_ltls)
    print(acc_lang)

    return output_ltls


def run_exp_modular():
    # Load data
    input_utterances = load_from_file(args.input)
    true_ltls = load_from_file(args.true_ltls)
    ner_prompt = load_from_file(args.ner_prompt)
    trans_prompt = load_from_file(args.trans_prompt)

    # Load modules
    if args.ner == 'gpt3':
        ner_module = GPT3()
        ner_queries = [ner_prompt+utt+"\nLandmarks:" for utt in input_utterances]
    # elif args.ner == 'bert':
    #     ner_module = BERT()
    else:
        raise ValueError("ERROR: NER module not recognized")

    if args.trans == 'gpt3':
        trans_module = GPT3()
    # elif args.trans == 's2s_sup':
    #     trans_module = Seq2Seq()
    else:
        raise ValueError("ERROR: translation module not recognized")

    # Query modules
    placeholder_maps = ner_task(ner_module, ner_queries)

    trans_queries = substitue(input_utterances, placeholder_maps)  # replace names by symbols
    if args.trans == 'gpt3':  # add prompt if translation module uses GPT-3
        trans_queries = [trans_prompt + query + "\nLTL:" for query in trans_queries]

    output_ltls = trans_task(trans_module, trans_queries)

    placeholder_maps_inv = [
        {letter: name for name, letter in placeholder_map.items()}
        for placeholder_map in placeholder_maps
    ]
    output_ltls = substitue(output_ltls, placeholder_maps_inv)  # replace symbols by names

    # Evaluate
    acc_lang = evalulate_lang(output_ltls, true_ltls)
    print(acc_lang)

    return output_ltls


def ner_task(ner_module, ner_queries):
    return [ner_module.extract(query) for query in ner_queries]


def trans_task(trans_module, trans_queries):
    return [trans_module.translate(query) for query in trans_queries]


def substitue(input_strs, placeholder_maps):
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


def evalulate_lang(output_ltls, true_ltls):
    accs = []
    for out_ltl in output_ltls:
        accs.append(out_ltl in true_ltls)
    return sum(accs) / len(accs)


def ground_task(ground_module, known_names, queries):
    ground_maps = []
    return ground_maps


def plan(output_ltls, true_trajs):
    accs = []
    planner = None
    for out_ltl, true_traj in zip(output_ltls, true_trajs):
        out_traj = planner.plan(out_ltl)
        accs.append(evaluate_plan(out_traj, true_traj))
    return sum(accs) / len(accs)


def evaluate_plan(out_traj, true_traj):
    return out_traj == true_traj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--e2e_gpt3', action='store_true', help="solve translation task end-to-end using GPT-3")
    parser.add_argument('--e2e_prompt', type=str, default='data/e2e_prompt.txt', help='path to end-to-end prompt')
    parser.add_argument('--ner', type=str, default='gpt3', help='NER module: gpt3, bert')
    parser.add_argument('--trans', type=str, default='gpt3', help='translation module: gpt3, s2s_sup, s2s_weaksup')
    parser.add_argument('--input', type=str, default='data/input.pkl', help='file path to input utterances')
    parser.add_argument('--true_ltls', type=str, default='data/true_ltls.pkl', help='path to true LTLs')
    parser.add_argument('--ner_prompt', type=str, default='data/ner_prompt.txt', help='path to NER prompt')
    parser.add_argument('--trans_prompt', type=str, default='data/trans_prompt.txt', help='path to trans prompt')
    parser.add_argument('--ground', type=str, default='gpt3', help='grounding module: gpt3, bert')
    parser.add_argument('--known_names', type=str, default='data/known_names.pkl', help='path to known_names')
    parser.add_argument('--ground_prompt', type=str, default='data/ground_prompt.txt', help='path to ground prompt')
    parser.add_argument('--true_trajs', type=str, default='data/true_trajs.pkl', help='path to true trajectories')
    args = parser.parse_args()

    run_exp()
