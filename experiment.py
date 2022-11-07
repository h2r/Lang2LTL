import os
import argparse
import logging
import random
import numpy as np
import spot
from openai.embeddings_utils import cosine_similarity

from gpt3 import GPT3
from s2s_sup import Seq2Seq
from utils import load_from_file, save_to_file, build_placeholder_map, substitute


def run_exp():
    # Language tasks: grounding, NER, translation
    if args.full_e2e:  # Full end-to-end from language to LTL
        full_e2e_module = GPT3()
        full_e2e_prompt = load_from_file(args.full_e2e_prompt)
        output_ltls = [full_e2e_module.translate(query, prompt=full_e2e_prompt) for query in input_utts]
    else:  # Modular
        names, utt2names = ner()
        name2grounds = grounding(names)
        grounded_utts, objs_per_utt = ground_utterances(input_utts, utt2names, name2grounds)  # ground names to objects in env
        if args.trans_e2e:
            output_ltls = translate_e2e(grounded_utts)
        else:
            output_ltls, symbolic_ltls, placeholder_maps = translate_modular(grounded_utts, objs_per_utt)

    if len(input_utts) != len(output_ltls):
        logging.info(f'ERROR: # input utterances {len(input_utts)} != # output LTLs {len(output_ltls)}')
    accs_lang, acc_lang = evaluate_lang(output_ltls, true_ltls)
    for idx, (input_utt, output_ltl, true_ltl, acc) in enumerate(zip(input_utts, output_ltls, true_ltls, accs_lang)):
        logging.info(f'{idx}\nInput utterance: {input_utt}\nTrue LTL: {true_ltl}\nOutput LTL: {output_ltl}\n{acc}\n')
    logging.info(f'Language to LTL translation accuracy: {acc_lang}')

    final_results = {
        'NER': utt2names if not args.full_e2e else None,
        'Grounding': name2grounds if not args.full_e2e else None,
        'Placeholder maps': placeholder_maps if not (args.trans_e2e or args.full_e2e) else None,
        'Input utterances': input_utts,
        'Symbolic LTLs': symbolic_ltls if not (args.trans_e2e or args.full_e2e) else None,
        'Output LTLs': output_ltls,
        'Ground truth': true_ltls
    }
    save_to_file(final_results, args.save_result_path)

    # Planning task: LTL + MDP -> policy
    # true_trajs = load_from_file(args.true_trajs)
    # acc_plan = plan(output_ltls, name2grounds)
    # logging.info(f'Planning accuracy: {acc_plan}')


def ner():
    """
    Name Entity Recognition: extract name entities from input utterances.
    """
    ner_prompt = load_from_file(args.ner_prompt)

    if args.ner == 'gpt3':
        ner_module = GPT3()
    # elif args.ner == 'bert':
    #     ner_module = BERT()
    else:
        raise ValueError(f'ERROR: NER module not recognized: {args.ner}')

    names, utt2names = set(), []  # name entity list names should not have duplicates
    for idx_utt, utt in enumerate(input_utts):
        logging.info(f'Extracting name entities from utterance: {idx_utt}')
        names_per_utt = [name.lower().strip() for name in ner_module.extract_ne(utt, prompt=ner_prompt)]

        extra_names = []  # make sure both 'name' and 'the name' are in names_per_utt to mitigate NER error
        for name in names_per_utt:
            name_words = name.split()
            if name_words[0] == 'the':
                extra_name = ' '.join(name_words[1:])
            else:
                name_words.insert(0, 'the')
                extra_name = ' '.join(name_words)
            if extra_name not in names_per_utt:
                extra_names.append(extra_name)
        names_per_utt += extra_names

        names.update(names_per_utt)
        utt2names.append((utt, names_per_utt))
    return names, utt2names


def grounding(names):
    """
    Find groundings (objects in given environment) of name entities in input utterances.
    """
    obj2embed = load_from_file(args.obj_embed)  # load embeddings of known objects in given environment
    if os.path.exists(args.name_embed):  # load cached embeddings of name entities
        name2embed = load_from_file(args.name_embed)
    else:
        name2embed = {}

    if args.ground == 'gpt3':
        ground_module = GPT3()
    # elif args.ground == 'bert':
    #     ground_module = BERT()
    else:
        raise ValueError(f'ERROR: grounding module not recognized: {args.ground}')

    name2grounds, is_embed_added = {}, False
    for name in names:
        if name in name2embed:  # use cached embedding if exists
            embed = name2embed[name]
        else:
            embed = ground_module.get_embedding(name, args.engine)
            name2embed[name] = embed
            is_embed_added = True

        sims = {n: cosine_similarity(e, embed) for n, e in obj2embed.items()}
        sims_sorted = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        name2grounds[name] = list(dict(sims_sorted[:args.topk]).keys())

        if is_embed_added:
            save_to_file(name2embed, args.name_embed)

    return name2grounds


def ground_utterances(input_strs, utt2names, name2grounds):
    """
    Replace name entities in input strings (e.g. utterances, LTL formulas) with objects in given environment.
    """
    grounding_maps = []  # name to grounding map per utterance
    for _, names in utt2names:
        grounding_maps.append({name: name2grounds[name][0] for name in names})
    return substitute(input_strs, grounding_maps)


def translate_e2e(grounded_utts):
    """
    Translation language to LTL using a single GPT-3.
    """
    trans_e2e_prompt = load_from_file(args.trans_e2e_prompt)
    model = GPT3()
    output_ltls = [model.translate(utt, prompt=trans_e2e_prompt) for utt in grounded_utts]
    return output_ltls


def translate_modular(grounded_utts, objs_per_utt):
    """
    Translation language to LTL modular approach.
    :param grounded_utts: Input utterances with name entities grounded to objects in given environment.
    :param objs_per_utt: grounding objects for each input utterance
    :return: output grounded LTL formulas, corresponding intermediate symbolic LTL formulas, placeholder maps
    """
    trans_modular_prompt = load_from_file(args.trans_modular_prompt)

    if args.trans == 'gpt3':
        trans_module = GPT3()
    elif args.trans == 's2s_sup':
        trans_module = Seq2Seq()
    else:
        raise ValueError("ERROR: translation module not recognized")

    placeholder_maps, placeholder_maps_inv = [], []
    for objs in objs_per_utt:
        placeholder_map, placeholder_map_inv = build_placeholder_map(objs)
        placeholder_maps.append(placeholder_map)
        placeholder_maps_inv.append(placeholder_map_inv)

    trans_queries, _ = substitute(grounded_utts, placeholder_maps)  # replace name entities by symbols
    symbolic_ltls = [trans_module.translate(query, prompt=trans_modular_prompt) for query in trans_queries]
    output_ltls, _ = substitute(symbolic_ltls, placeholder_maps_inv)  # replace symbols by name entities
    return output_ltls, symbolic_ltls, placeholder_maps


def evaluate_lang(output_ltls, true_ltls):
    """
    Parse LTL formulas in infix or prefix (spot.formula) then check semantic equivalence (spot.are_equivalent).
    """
    accs = []
    for out_ltl, true_ltl in zip(output_ltls, true_ltls):
        try:  # output LTL formula may have syntax error
            accs.append(spot.are_equivalent(spot.formula(out_ltl), spot.formula(true_ltl)))
        except SyntaxError:
            logging.info(f'Syntax error in output LTL: {out_ltl}')
            accs.append(False)
    acc = np.mean(accs)
    return accs, acc


def plan(output_ltls, true_trajs, name2grounds):
    """
    Planning with translated LTL as task specification
    """
    accs = []
    planner = None
    for out_ltl, true_traj in zip(output_ltls, true_trajs):
        out_traj = planner.plan(out_ltl, name2grounds)
        accs.append(evaluate_plan(out_traj, true_traj))
    acc = np.mean(accs)
    return acc


def evaluate_plan(out_traj, true_traj):
    return out_traj == true_traj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/cleanup_src_corlw.txt', help='file path to input utterances')
    parser.add_argument('--true_ltls', type=str, default='data/cleanup_tar_corlw.txt', help='path to true grounded LTL formulas')
    parser.add_argument('--nsamples', type=int, default=None, help='randomly sample nsamples pairs or None to use all')
    parser.add_argument('--true_trajs', type=str, default='data/true_trajs.pkl', help='path to true trajectories')
    parser.add_argument('--full_e2e', action='store_true', help="solve translation and ground end-to-end using GPT-3")
    parser.add_argument('--full_e2e_prompt', type=str, default='data/cleanup_full_e2e_prompt_15.txt', help='path to full end-to-end prompt')
    parser.add_argument('--trans_e2e', action='store_true', help="solve translation task end-to-end using GPT-3")
    parser.add_argument('--trans_e2e_prompt', type=str, default='data/cleanup_trans_e2e_prompt_15.txt', help='path to translation end-to-end prompt')
    parser.add_argument('--ner', type=str, default='gpt3', choices=['gpt3', 'bert'], help='NER module')
    parser.add_argument('--ner_prompt', type=str, default='data/cleanup_ner_prompt_15.txt', help='path to NER prompt')
    parser.add_argument('--trans', type=str, default='gpt3', choices=['gpt3', 's2s_sup', 's2s_weaksup'], help='translation module')
    parser.add_argument('--trans_modular_prompt', type=str, default='data/cleanup_trans_modular_prompt_15.txt', help='path to trans prompt')
    parser.add_argument('--ground', type=str, default='gpt3', choices=['gpt3', 'bert'], help='grounding module')
    parser.add_argument('--obj_embed', type=str, default='data/cleanup_obj2embed_gpt3_davinci.pkl', help='path to embedding of objects in env')
    parser.add_argument('--name_embed', type=str, default='data/cleanup_name2embed_gpt3_davinci.pkl', help='path to embedding of names in language')
    parser.add_argument('--topk', type=int, default=2, help='top k similar known names to name entity')
    parser.add_argument('--engine', type=str, default='davinci', choices=['ada', 'babbage', 'curie', 'davinci'], help='gpt-3 engine')
    parser.add_argument('--save_result_path', type=str, default='results/modular_prompt15_cleanup_corlw.json', help='file path to save outputs of each model in a json file')
    args = parser.parse_args()

    input_utts, true_ltls = load_from_file(args.input), load_from_file(args.true_ltls)
    if args.nsamples:  # for testing, randomly sample 50 pairs to cover diversity of dataset
        input_utts, true_ltls = zip(*random.sample(list(zip(input_utts, true_ltls)), args.nsamples))

    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(f'{os.path.splitext(args.save_result_path)[0]}.log', mode='w'),
                            logging.StreamHandler()
                        ]
    )

    run_exp()
