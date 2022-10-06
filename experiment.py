import argparse
import itertools
from openai.embeddings_utils import cosine_similarity

from gpt3 import GPT3
from utils import load_from_file, build_placeholder_map, substitute, clean_str


def run_exp():
    utt2names = ner()
    
    names = set(list(itertools.chain.from_iterable(utt2names.values())))  # flatten list of lists
    name2grounds = grounding(names)
    
    if args.e2e_gpt3:
        output_ltls = translate_e2e()
    else:
        output_ltls = translate_modular(utt2names, name2grounds)
    output_ltls = [clean_str(output_ltl) for output_ltl in output_ltls]
    print('Generated LTLs:\n', output_ltls,
          '\n\nGround Truth LTLs:\n', true_ltls)
    evaluate_lang(output_ltls, true_ltls)

    # true_trajs = load_from_file(args.true_trajs)
    # plan(output_ltls, true_trajs, name2grounds)


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

    utt2names = {utt: [clean_str(name) for name in ner_module.extract_ne(utt, prompt=ner_prompt)] for utt in input_utterances}
    return utt2names


def translate_e2e():
    """
    Translation language to LTL using a single GPT-3.
    """
    e2e_prompt = load_from_file(args.e2e_prompt)
    model = GPT3()
    output_ltls = [model.translate(utt, prompt=e2e_prompt) for utt in input_utterances]
    return output_ltls


def translate_modular(utt2names, name2grounds):
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

    placeholder_maps = [build_placeholder_map(names) for names in utt2names.values()]
    trans_queries = substitute(input_utterances, placeholder_maps)  # replace names by symbols
    output_ltls = [trans_module.translate(query, prompt=trans_prompt) for query in trans_queries]

    placeholder_maps_inv = [
        {letter: name2grounds[name][0] for name, letter in placeholder_map.items()}
        for placeholder_map in placeholder_maps
    ]
    
    output_ltls = substitute(output_ltls, placeholder_maps_inv)  # replace symbols by names

    return output_ltls


def evaluate_lang(output_ltls, true_ltls):
    accs = []
    for out_ltl, true_ltl in zip(output_ltls,true_ltls):
        accs.append(out_ltl == true_ltl)
    acc = sum(accs) / len(accs)
    print(f"Lang2LTL translation accuracy: {acc}")
    return acc


def grounding(names):
    """
    Ground name entities to objects in the environment.
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
        
        embed = ground_module.get_embedding(name)
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
    parser.add_argument('--e2e_gpt3', action='store_true', help="solve translation task end-to-end using GPT-3")
    parser.add_argument('--e2e_prompt', type=str, default='data/e2e_prompt.txt', help='path to end-to-end prompt')
    parser.add_argument('--ner', type=str, default='gpt3', help='NER module: gpt3, bert')
    parser.add_argument('--trans', type=str, default='gpt3', help='translation module: gpt3, s2s_sup, s2s_weaksup')
    parser.add_argument('--input', type=str, default='data/test_src.txt', help='file path to input utterances')
    parser.add_argument('--true_ltls', type=str, default='data/test_tar.txt', help='path to true LTLs')
    parser.add_argument('--ner_prompt', type=str, default='data/ner_prompt.txt', help='path to NER prompt')
    parser.add_argument('--trans_prompt', type=str, default='data/trans_prompt.txt', help='path to trans prompt')
    parser.add_argument('--ground', type=str, default='gpt3', help='grounding module: gpt3, bert')
    parser.add_argument('--name_embed', type=str, default='data/name2embed_davinci.json', help='path to name to embedding')
    parser.add_argument('--topk', type=int, default=2, help='top k similar known names to name entity')
    parser.add_argument('--true_trajs', type=str, default='data/true_trajs.pkl', help='path to true trajectories')
    args = parser.parse_args()

    input_utterances = load_from_file(args.input)
    true_ltls = load_from_file(args.true_ltls)

    run_exp()
