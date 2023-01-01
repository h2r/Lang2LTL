import os
import argparse
import logging
import random
import numpy as np
import spot
from openai.embeddings_utils import cosine_similarity

from gpt3 import GPT3
from s2s_sup import Seq2Seq, T5_MODELS
from s2s_pt_transformer import construct_dataset
from utils import load_from_file, save_to_file, build_placeholder_map, substitute
from evaluation import evaluate_lang, evaluate_plan


def run_exp(save_result_path):
    # Language tasks: grounding, NER, translation
    if args.full_e2e:  # Full end-to-end from language to LTL
        full_e2e_module = GPT3()
        full_e2e_prompt = load_from_file(args.full_e2e_prompt)
        output_ltls = [full_e2e_module.translate(query, prompt=full_e2e_prompt, n=1) for query in input_utts]
    else:  # Modular
        names, utt2names = ner()
        name2grounds = grounding(names)
        grounded_utts, objs_per_utt = ground_utterances(input_utts, utt2names, name2grounds)  # ground names to objects in env
        if args.trans_e2e:
            output_ltls = translate_e2e(grounded_utts)
        else:
            output_ltls, symbolic_ltls, placeholder_maps = translate_modular(grounded_utts, objs_per_utt)

    if len(input_utts) != len(output_ltls):
        logging.info(f"ERROR: # input utterances {len(input_utts)} != # output LTLs {len(output_ltls)}")
    accs_lang, acc_lang = evaluate_lang(output_ltls, true_ltls)
    for idx, (input_utt, output_ltl, true_ltl, acc) in enumerate(zip(input_utts, output_ltls, true_ltls, accs_lang)):
        logging.info(f"{idx}\nInput utterance: {input_utt}\nTrue LTL: {true_ltl}\nOutput LTL: {output_ltl}\n{acc}\n")
    logging.info(f"Language to LTL translation accuracy: {acc_lang}")

    final_results = {
        'NER': utt2names if not args.full_e2e else None,
        'Grounding': name2grounds if not args.full_e2e else None,
        'Placeholder maps': placeholder_maps if not (args.trans_e2e or args.full_e2e) else None,
        'Input utterances': input_utts,
        'Symbolic LTLs': symbolic_ltls if not (args.trans_e2e or args.full_e2e) else None,
        'Output LTLs': output_ltls,
        'Ground truth': true_ltls,
        'Accuracy': acc_lang
    }
    save_to_file(final_results, save_result_path)

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
        raise ValueError(f"ERROR: NER module not recognized: {args.ner}")

    names, utt2names = set(), []  # name entity list names should not have duplicates
    for idx_utt, utt in enumerate(input_utts):
        logging.info(f"Extracting name entities from utterance: {idx_utt}")
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
        raise ValueError(f"ERROR: grounding module not recognized: {args.ground}")

    name2grounds, is_embed_added = {}, False
    for name in names:
        if name in name2embed:  # use cached embedding if exists
            embed = name2embed[name]
        else:
            embed = ground_module.get_embedding(name, args.embed_engine)
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
    output_ltls = [model.translate(utt, prompt=trans_e2e_prompt, n=1) for utt in grounded_utts]
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
    elif args.trans in T5_MODELS:
        trans_module = Seq2Seq(args.trans)
    elif args.trans == 'pt_transformer':
        _, _, vocab_transform, text_transform, src_vocab_size, tar_vocab_size = construct_dataset(args.s2s_sup_data)
        model_params = f"model/s2s_{args.trans}.pth"
        trans_module = Seq2Seq(args.trans,
                               vocab_transform=vocab_transform, text_transform=text_transform,
                               src_vocab_sz=src_vocab_size, tar_vocab_sz=tar_vocab_size, fpath_load=model_params)
    else:
        raise ValueError(f"ERROR: translation module not recognized: {args.trans}")

    placeholder_maps, placeholder_maps_inv = [], []
    for objs in objs_per_utt:
        placeholder_map, placeholder_map_inv = build_placeholder_map(objs)
        placeholder_maps.append(placeholder_map)
        placeholder_maps_inv.append(placeholder_map_inv)
    trans_queries, _ = substitute(grounded_utts, placeholder_maps)  # replace name entities by symbols

    symbolic_ltls = []
    for query in trans_queries:
        ltl = trans_module.translate(query, prompt=trans_modular_prompt, n=1)
        try:
            spot.formula(ltl)
        except SyntaxError:
            ltl = feedback_module(trans_module, query, trans_modular_prompt, ltl)
        symbolic_ltls.append(ltl)

    output_ltls, _ = substitute(symbolic_ltls, placeholder_maps_inv)  # replace symbols by name entities
    return output_ltls, symbolic_ltls, placeholder_maps


def feedback_module(trans_module, query, trans_modular_prompt, ltl_incorrect, n=20):
    """
    :param trans_module: model for the translation module.
    :param query: input utterance.
    :param ltl_incorrect:  LTL formula that has syntax error.
    :param trans_modular_prompt: prompt for GPT-3 translation module
    :param n: number of outupt to sample from the translation module.
    :return: LTL formula of correct syntax and most likely to be correct translation of utterance `query'.
    """
    breakpoint()
    logging.info(f"Syntax error in {query} | {ltl_incorrect}")
    if isinstance(trans_module, GPT3):
        ltls_fix = trans_module.translate(query, prompt=trans_modular_prompt, n=n)
    else:
        ltls_fix = trans_module.translate(query, n=n)
    logging.info(f"{n} candidate LTL formulas: {ltls_fix}")
    ltl_fix = ""
    for ltl in ltls_fix:
        try:
            spot.formula(ltl)
            ltl_fix = ltl
            break
        except SyntaxError:
            continue
    logging.info(f"Fixed LTL: {ltl_fix}")
    return ltl_fix


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=str, default='data/cleanup_corlw.csv', help='file path to utterance, ltl pairs')
    parser.add_argument('--nsamples', type=int, default=None, help='randomly sample nsamples pairs or None to use all')
    parser.add_argument('--nruns', type=int, default=1, help='number of runs to test each model')
    parser.add_argument('--true_trajs', type=str, default='data/true_trajs.pkl', help='path to true trajectories')
    parser.add_argument('--full_e2e', action='store_true', help="solve translation and ground end-to-end using GPT-3")
    parser.add_argument('--full_e2e_prompt', type=str, default='data/cleanup_full_e2e_prompt_15.txt', help='path to full end-to-end prompt')
    parser.add_argument('--trans_e2e', action='store_true', help="solve translation task end-to-end using GPT-3")
    parser.add_argument('--trans_e2e_prompt', type=str, default='data/cleanup_trans_e2e_prompt_15.txt', help='path to translation end-to-end prompt')
    parser.add_argument('--ner', type=str, default='gpt3', choices=['gpt3', 'bert'], help='NER module')
    parser.add_argument('--ner_prompt', type=str, default='data/cleanup_ner_prompt_15.txt', help='path to NER prompt')
    parser.add_argument('--trans', type=str, default='gpt3', choices=['gpt3', 't5-base', 't5-small', 'pt_transformer'], help='translation module')
    parser.add_argument('--trans_modular_prompt', type=str, default='data/cleanup_trans_modular_prompt_15.txt', help='path to trans prompt')
    parser.add_argument('--ground', type=str, default='gpt3', choices=['gpt3', 'bert'], help='grounding module')
    parser.add_argument('--obj_embed', type=str, default='data/cleanup_obj2embed_gpt3_ada-002.pkl', help='path to embedding of objects in env')
    parser.add_argument('--name_embed', type=str, default='data/cleanup_name2embed_gpt3_ada-002.pkl', help='path to embedding of names in language')
    parser.add_argument('--topk', type=int, default=2, help='top k similar known names to name entity')
    parser.add_argument('--embed_engine', type=str, default='text-embedding-ada-002', help='gpt-3 embedding engine')
    parser.add_argument('--s2s_sup_data', type=str, default='data/symbolic_pairs.csv', help='file path to train and test data for supervised seq2seq')
    parser.add_argument('--save_result_path', type=str, default='results/modular_prompt15_cleanup_test.json', help='file path to save outputs of each model in a json file')
    args = parser.parse_args()

    pairs = load_from_file(args.pairs)
    input_utts, true_ltls = [], []
    for utt, ltl in pairs:
        input_utts.append(utt)
        true_ltls.append(ltl)
    assert len(input_utts) == len(true_ltls), f"ERROR: # input utterances {len(input_utts)} != # output LTLs {len(true_ltls)}"
    if args.nsamples:  # for testing, randomly sample `nsamples` pairs to cover diversity of dataset
        random.seed(42)
        input_utts, true_ltls = zip(*random.sample(list(zip(input_utts, true_ltls)), args.nsamples))

    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(f'{os.path.splitext(args.save_result_path)[0]}.log', mode='w'),
                            logging.StreamHandler()
                        ]
    )

    for run in range(args.nruns):
        fpath_tup = os.path.splitext(args.save_result_path)
        save_result_path = f"{fpath_tup[0]}_run{run}" + fpath_tup[1]
        logging.info(f"\n\n\nRUN: {run}")
        run_exp(save_result_path)
