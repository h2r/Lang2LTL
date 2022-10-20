import argparse
import itertools
from openai.embeddings_utils import cosine_similarity

from gpt3 import GPT3
from utils import load_from_file, save_to_file, build_placeholder_map, substitute


def run_exp():
    # Language tasks: grounding, NER, translation
    if args.overall_e2e:  # End-to-End
        e2e_module = GPT3()
        e2e_prompt = load_from_file(args.overall_e2e_prompt)
        output_ltls = [e2e_module.translate(query, prompt=e2e_prompt) for query in input_utterances]
    else:  # Modular
        utt2names = ner()

        names = set(list(itertools.chain.from_iterable(utt2names.values())))  # flatten list of lists; remove duplicates
        name2grounds = grounding(names)

        # TODO: ground language then translate grounded languaage to LTL instead of translate then ground output LTL?

        if args.translate_e2e:
            output_ltls = translate_e2e()
        else:
            output_ltls, symbolic_ltls, placeholder_maps = translate_modular(utt2names)

        output_ltls = ground_ltls(output_ltls, name2grounds)  # replace landmarks by their groundings
        output_ltls = [output_ltl.strip() for output_ltl in output_ltls]

    for input_utt, output_ltl, true_ltl in zip(input_utterances, output_ltls, true_ltls):
        print(f'Input utterance: {input_utt}\nOutput LTLs: {output_ltl}\nTrue LTLs: {true_ltl}\n')
    acc = evaluate_lang(output_ltls, true_ltls)

    # Planning task: LTL + MDP -> policy
    # true_trajs = load_from_file(args.true_trajs)
    # plan(output_ltls, true_trajs, name2grounds)

    if args.save_result_path:
        final_results = {
            'NER': utt2names if not args.overall_e2e else None,
            'Grounding': name2grounds if not args.overall_e2e else None,
            'Placeholder maps': placeholder_maps if not (args.translate_e2e or args.overall_e2e) else None,
            'Symbolic LTLs': symbolic_ltls if not (args.translate_e2e or args.overall_e2e) else None,
            'Output LTLs': output_ltls,
            'Input utterances': input_utterances,
            'Ground truth': true_ltls,
            'Accuracy': acc
            }
        save_to_file(final_results, args.save_result_path)


def ner():
    """
    Name Entity Recognition: extract name entities from input utterance
    """
    ner_prompt = load_from_file(args.ner_prompt)

    if args.ner == 'gpt3':
        ner_module = GPT3()
    # elif args.ner == 'bert':
    #     ner_module = BERT()
    else:
        raise ValueError("ERROR: NER module not recognized")

    utt2names = {utt: [name.strip() for name in ner_module.extract_ne(utt, prompt=ner_prompt)]
                 for utt in input_utterances}
    return utt2names


def translate_e2e():
    """
    Translation language to LTL using a single GPT-3.
    """
    trans_e2e_prompt = load_from_file(args.trans_e2e_prompt)
    model = GPT3()
    output_ltls = [model.translate(utt, prompt=trans_e2e_prompt) for utt in input_utterances]
    return output_ltls


def translate_modular(utt2names):
    """
    Translation language to LTL modular approach.
    """
    trans_prompt = load_from_file(args.trans_prompt)

    if args.trans == 'gpt3':
        trans_module = GPT3()
    # elif args.trans == 's2s_sup':
    #     trans_module = Seq2Seq()
    else:
        raise ValueError("ERROR: translation module not recognized")

    placeholder_maps = [build_placeholder_map(names) for names in utt2names.values()]  # TODO: also build inv here
    trans_queries = substitute(input_utterances, placeholder_maps)  # replace name entities by symbols
    symbolic_ltls = [trans_module.translate(query, prompt=trans_prompt).strip() for query in trans_queries]  # TODO: some symbolic ltls contain newline

    placeholder_maps_inv = [
        {letter: name for name, letter in placeholder_map.items()}
        for placeholder_map in placeholder_maps
    ]

    output_ltls = substitute(symbolic_ltls, placeholder_maps_inv)  # replace symbols by name entities

    return output_ltls, symbolic_ltls, placeholder_maps


def ground_ltls(output_ltls, name2grounds):
    """
    Replace landmarks in output LTLs with objects in the environment and output final LTLs for planning.
    """
    name2ground = {name: grounds[0] for name, grounds in name2grounds.items()}  # TODO: redundant item when k == v
    output_ltls = substitute(output_ltls, [name2ground])
    return output_ltls


def evaluate_lang(output_ltls, true_ltls):
    accs = []
    for out_ltl, true_ltl in zip(output_ltls, true_ltls):
        accs.append(out_ltl == true_ltl)
    acc = sum(accs) / len(accs)
    print(f"Lang2LTL translation accuracy: {acc}")
    return acc


def grounding(names):
    """
    Ground name entities in LTL formulas to objects in the environment.
    """
    name2embed = load_from_file(args.name_embed)

    if args.ground == 'gpt3':
        ground_module = GPT3()
    # elif args.ground == 'bert':
    #     ground_module = BERT()
    else:
        raise ValueError("ERROR: grounding module not recognized")

    name2grounds = {}
    for name in names:
        embed = ground_module.get_embedding(name, args.engine)
        sims = {n: cosine_similarity(e, embed) for n, e in name2embed.items()}
        sims_sorted = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        name2grounds[name] = list(dict(sims_sorted[:args.topk]).keys())

    return name2grounds


def plan(output_ltls, true_trajs, name2grounds):
    """
    Planning with translated LTL as task specification
    """
    accs = []
    planner = None
    for out_ltl, true_traj in zip(output_ltls, true_trajs):
        out_traj = planner.plan(out_ltl, name2grounds)
        accs.append(evaluate_plan(out_traj, true_traj))
    acc = sum(accs) / len(accs)
    print(f"{acc}")
    return acc


def evaluate_plan(out_traj, true_traj):
    return out_traj == true_traj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overall_e2e', action='store_true', help="solve translation and ground end-to-end using GPT-3")
    parser.add_argument('--overall_e2e_prompt', type=str, default='data/overall_e2e_prompt.txt', help='path to overal end-to-end prompt')
    parser.add_argument('--translate_e2e', action='store_true', help="solve translation task end-to-end using GPT-3")
    parser.add_argument('--trans_e2e_prompt', type=str, default='data/trans_e2e_prompt.txt', help='path to translation end-to-end prompt')
    parser.add_argument('--ner', type=str, default='gpt3', choices=['gpt3', 'bert'], help='NER module')
    parser.add_argument('--trans', type=str, default='gpt3', choices=['gpt3', 's2s_sup', 's2s_weaksup'], help='translation module')
    parser.add_argument('--input', type=str, default='data/test_src.txt', help='file path to input utterances')
    parser.add_argument('--true_ltls', type=str, default='data/test_tar.txt', help='path to true LTLs')
    parser.add_argument('--ner_prompt', type=str, default='data/ner_prompt.txt', help='path to NER prompt')
    parser.add_argument('--trans_prompt', type=str, default='data/trans_prompt.txt', help='path to trans prompt')
    parser.add_argument('--ground', type=str, default='gpt3', help='grounding module: gpt3, bert')
    parser.add_argument('--name_embed', type=str, default='data/name2embed_davinci.json', help='path to known name embedding')
    parser.add_argument('--topk', type=int, default=2, help='top k similar known names to name entity')
    parser.add_argument('--true_trajs', type=str, default='data/true_trajs.pkl', help='path to true trajectories')
    parser.add_argument('--engine', type=str, default='davinci', choices=['ada', 'babbage', 'curie', 'davinci'], help='gpt-3 engine')
    parser.add_argument('--save_result_path', type=str, default='data/test_result.json', help='file path to save outputs of each model in a json file')
    args = parser.parse_args()

    input_utterances = load_from_file(args.input)
    true_ltls = load_from_file(args.true_ltls)

    run_exp()
