import argparse
from openai.embeddings_utils import cosine_similarity

from gpt3 import GPT3
from utils import load_from_file, build_placeholder_map, substitute


def run_exp():
    # NER: extract name entities from input utterance
    ner_prompt = load_from_file(args.ner_prompt)

    if args.ner == 'gpt3':
        ner_module = GPT3()
    # elif args.ner == 'bert':
    #     ner_module = BERT()
    else:
        raise ValueError("ERROR: NER module not recognized")

    utt2names = {utt: ner_module.extract_ne(utt, prompt=ner_prompt) for utt in input_utterances}

    # Translation: language to LTl
    if args.e2e_gpt3:
        output_ltls = translate_e2e()
    else:
        output_ltls = translate_modular(utt2names)

    # Evaluate translation
    evaluate_lang(output_ltls, true_ltls)

    # # Grounding: ground name entities to objects in the environment
    # name2embed = load_from_file(args.name_embed)  # Pandas Dataframe
    #
    # if args.ground == 'gpt3':
    #     ground_module = GPT3()
    # # elif args.ground == 'bert':
    # #     ground_module = BERT()
    # else:
    #     raise ValueError("ERROR: grounding module not recognized")
    #
    # ground_maps = ground_task(ground_module, name2embed, utt2names)

    # true_trajs = load_from_file(args.true_trajs)
    # acc_plan = plan(output_ltls, true_trajs, ground_maps)
    # print(acc_plan)


def translate_e2e():
    e2e_prompt = load_from_file(args.e2e_prompt)
    model = GPT3()
    output_ltls = [model.translate(utt, prompt=e2e_prompt) for utt in input_utterances]
    return output_ltls


def translate_modular(utt2names):
    trans_prompt = load_from_file(args.trans_prompt)

    if args.trans == 'gpt3':
        trans_module = GPT3()
    # elif args.trans == 's2s_sup':
    #     trans_module = Seq2Seq()
    else:
        raise ValueError("ERROR: translation module not recognized")

    placeholder_maps = [build_placeholder_map(names) for names in utt2names.values()]
    trans_queries = substitute(input_utterances, placeholder_maps)  # replace names by symbols
    output_ltls = [trans_module.translate(query, prompt=trans_prompt) for query in trans_queries]

    placeholder_maps_inv = [
        {letter: name for name, letter in placeholder_map.items()}
        for placeholder_map in placeholder_maps
    ]
    output_ltls = substitute(output_ltls, placeholder_maps_inv)  # replace symbols by names

    return output_ltls


def evaluate_lang(output_ltls, true_ltls):
    accs = []
    for out_ltl in output_ltls:
        accs.append(out_ltl in true_ltls)
    acc = sum(accs) / len(accs)
    print(f"Lang2LTL translation accuracy: {acc}")
    return acc


def ground_task(ground_module, name2embed_df, utt2names):
    embeddings = []
    for utt, names in utt2names.items():
        embeddings.append(ground_module.get_embedding(query))

    ground_maps = []

    for embed in embeddings:
        name2embed_df.apply(lambda x: cosine_similarity(x, embed))

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
    parser.add_argument('--name_embed', type=str, default='data/name_embed.csv', help='path to name to embedding')
    parser.add_argument('--true_trajs', type=str, default='data/true_trajs.pkl', help='path to true trajectories')
    args = parser.parse_args()

    input_utterances = load_from_file(args.input)
    true_ltls = load_from_file(args.true_ltls)

    run_exp()
