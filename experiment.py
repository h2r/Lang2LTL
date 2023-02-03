import os
import argparse
import logging
from pathlib import Path
import random
import numpy as np
import spot
from openai.embeddings_utils import cosine_similarity

from gpt3 import GPT3
from s2s_sup import Seq2Seq, T5_MODELS
from s2s_pt_transformer import construct_dataset_meta
from dataset_symbolic import load_split_dataset
from utils import load_from_file, save_to_file, build_placeholder_map, substitute, substitute_single_letter
from evaluation import evaluate_lang_0, evaluate_lang, evaluate_plan
from formula_sampler import TYPE2NPROPS
from analyze_results import find_all_formulas

PROPS = ["a", "b", "c", "d", "h", "j", "k", "l", "n", "o", "p", "q", "r", "s", "y", "z"]


def run_exp():
    # Language tasks: RER, grounding translation
    if args.full_e2e:  # Full end-to-end from language to LTL
        full_e2e_module = GPT3(translation_engine)
        full_e2e_prompt = load_from_file(args.full_e2e_prompt)
        out_ltls = [full_e2e_module.translate(query, full_e2e_prompt) for query in input_utts]

        accs, accumulated_acc = evaluate_lang_0(true_ltls, out_ltls)
        for idx, (input_utt, output_ltl, true_ltl, acc) in enumerate(zip(input_utts, true_ltls, out_ltls, accs)):
            logging.info(f"{idx}\nInput utterance: {input_utt}\nTrue LTL: {true_ltl}\nOutput LTL: {output_ltl}\n{acc}\n")
        logging.info(f"Language to LTL translation accuracy: {accumulated_acc}")
    else:  # Modular
        names, utt2names = rer()
        out_names = [utt_names[1] for utt_names in utt2names]  # referring expressions
        name2grounds = ground_names(names)
        grounded_utts, objs_per_utt = ground_utterances(input_utts, utt2names, name2grounds)  # ground names to objects in env
        if args.trans_e2e:
            out_ltls = translate_e2e(grounded_utts)
        else:
            out_ltls, out_sym_ltls, placeholder_maps = translate_modular(grounded_utts, objs_per_utt)

            # out_sym_ltls_sub = []
            # for props, out_sym_ltl, placeholder_map in zip(propositions, out_sym_ltls, placeholder_maps.items()):
            #     out_sym_ltls_sub.append(substitute_single_letter(out_sym_ltl, {letter: prop for (_, letter), prop in zip(placeholder_map.items(), props)}))
            # out_sym_ltls = out_sym_ltls_sub

            accs, accumulated_acc = evaluate_lang(true_ltls, out_ltls, true_names, out_names, objs_per_utt, args.convert_rule, PROPS)
            # accs, accumulated_acc = evaluate_lang_new(true_ltls, out_ltls, true_sym_ltls, out_sym_ltls, true_names, out_names, objs_per_utt)

            pair_results = [["Pattern Type", "Propositions", "Utterance", "True LTL", "Out LTL", "True Symbolic LTL", "Out Symbolic LTL", "True Lmks", "Out Lmks", "Out Lmk Ground", "Placeholder Map", "Accuracy"]]
            for idx, (pattern_type, props, input_utt, true_ltl, out_ltl, true_sym_ltl, out_sym_ltl, true_name, out_name, out_grnd, placeholder_maps, acc) in enumerate(zip(pattern_types, propositions, input_utts, true_ltls, out_ltls, true_sym_ltls, out_sym_ltls, true_names, out_names, objs_per_utt, placeholder_maps, accs)):
                logging.info(f"{idx}\n{pattern_type} {props} Input utterance: {input_utt}\n"
                             f"True Ground LTL: {true_ltl}\nOut Ground LTL: {out_ltl}\n"
                             f"True Symbolic LTL: {true_sym_ltl}\nOut Symbolic LTL: {out_sym_ltl}\n"
                             f"True Lmks: {true_name}\nOut Lmks:{out_name}\nOut Grounds: {out_grnd}\nPlaceholder Map: {placeholder_maps}\n"
                             f"{acc}\n")
                pair_results.append((pattern_type, props, input_utt, true_ltl, out_ltl, true_sym_ltl, out_sym_ltl, true_name, out_name, out_grnd, placeholder_maps, acc))
            logging.info(f"Language to LTL translation accuracy: {accumulated_acc}\n\n")
            save_to_file(pair_results, pair_result_fpath)

    if len(input_utts) != len(out_ltls):
        logging.info(f"ERROR: # input utterances {len(input_utts)} != # output LTLs {len(out_ltls)}")

    all_results = {
        "RER": utt2names if not args.full_e2e else None,
        "Grounding": name2grounds if not args.full_e2e else None,
        "Placeholder maps": placeholder_maps if not (args.trans_e2e or args.full_e2e) else None,
        "Input utterances": input_utts,
        "Output Symbolic LTLs": out_sym_ltls if not (args.trans_e2e or args.full_e2e) else None,
        "Output Grounded LTLs": out_ltls,
        "True Grounded LTLs": true_ltls,
        "Meta": meta_iter,
        "Accuracies": accs,
        "Accumulated Accuracy": accumulated_acc
    }
    save_to_file(all_results, all_result_fpath)
    save_to_file(all_results, os.path.join(os.path.dirname(all_result_fpath), f"{Path(all_result_fpath).stem}.pkl"))  # also save to pkl to preserve data type

    # Planning task: LTL + MDP -> policy
    # true_trajs = load_from_file(args.true_trajs)
    # acc_plan = plan(output_ltls, name2grounds)
    # logging.info(f"Planning accuracy: {acc_plan}")


def rer():
    """
    Referring Expression Recognition: extract name entities from input utterances.
    """
    rer_prompt = load_from_file(args.rer_prompt)

    if args.rer == "gpt3":
        rer_module = GPT3(args.rer_engine)
    # elif args.rer == "bert":
    #     rer_module = BERT()
    else:
        raise ValueError(f"ERROR: RER module not recognized: {args.rer}")

    names, utt2names = set(), []  # name entity list names should not have duplicates
    for idx_utt, utt in enumerate(input_utts):

        # print(f"{rer_prompt.strip()} {utt}\nPropositions:\n{idx_utt}")
        # breakpoint()

        logging.info(f"Extracting referring expressions from utterance: {idx_utt}/{len(input_utts)}")
        names_per_utt = [name.strip() for name in rer_module.extract_ne(query=f"{rer_prompt.strip()} {utt}\nPropositions:")]
        names_per_utt = list(set(names_per_utt))  # remove duplicated RE

        # print(names_per_utt)
        # breakpoint()

        # extra_names = []  # make sure both 'name' and 'the name' are in names_per_utt to mitigate RER error
        # for name in names_per_utt:
        #     name_words = name.split()
        #     if name_words[0] == "the":
        #         extra_name = " ".join(name_words[1:])
        #     else:
        #         name_words.insert(0, "the")
        #         extra_name = " ".join(name_words)
        #     if extra_name not in names_per_utt:
        #         extra_names.append(extra_name)
        # names_per_utt += extra_names

        names.update(names_per_utt)
        utt2names.append((utt, names_per_utt))

    # breakpoint()

    return names, utt2names


def ground_names(names):
    """
    Find groundings (objects in given environment) of name entities in input utterances.
    """
    obj2embed = load_from_file(obj_embed)  # load embeddings of known objects in given environment
    if os.path.exists(name_embed):  # load cached embeddings of name entities
        name2embed = load_from_file(name_embed)
    else:
        name2embed = {}

    if args.ground == "gpt3":
        ground_module = GPT3(args.embed_engine)
    # elif args.ground == "bert":
    #     ground_module = BERT()
    else:
        raise ValueError(f"ERROR: grounding module not recognized: {args.ground}")

    name2grounds = {}
    is_embed_added = False
    for name in names:
        logging.info(f"grounding landmark: {name}")
        if name in name2embed:  # use cached embedding if exists
            logging.info(f"use cached embedding: {name}")
            embed = name2embed[name]
        else:
            embed = ground_module.get_embedding(name)
            name2embed[name] = embed
            is_embed_added = True

        sims = {n: cosine_similarity(e, embed) for n, e in obj2embed.items()}
        sims_sorted = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        name2grounds[name] = list(dict(sims_sorted[:args.topk]).keys())

        if is_embed_added:
            save_to_file(name2embed, name_embed)

    return name2grounds


def ground_utterances(input_strs, utt2names, name2grounds):
    """
    Replace name entities in input strings (e.g. utterances, LTL formulas) with objects in given environment.
    """
    grounding_maps = []  # name to grounding map per utterance
    for _, names in utt2names:
        grounding_maps.append({name: name2grounds[name][0] for name in names})

    output_strs, subs_per_str = substitute(input_strs, grounding_maps, is_utt=True)

    # breakpoint()

    return output_strs, subs_per_str


def translate_modular(grounded_utts, objs_per_utt):
    """
    Translation language to LTL modular approach.
    :param grounded_utts: Input utterances with name entities grounded to objects in given environment.
    :param objs_per_utt: grounding objects for each input utterance
    :return: output grounded LTL formulas, corresponding intermediate symbolic LTL formulas, placeholder maps
    """
    if "ft" in translation_engine:
        trans_modular_prompt = ""
    elif "text-davinci" in translation_engine:
        trans_modular_prompt = load_from_file(args.trans_modular_prompt)
    else:
        raise ValueError(f"ERROR: Unrecognized translation engine: {translation_engine}")

    if "gpt3" in args.sym_trans:
        trans_module = GPT3(translation_engine)
    elif args.sym_trans in T5_MODELS:
        trans_module = Seq2Seq(args.sym_trans)
    elif args.sym_trans == "pt_transformer":
        train_iter, _, _, _ = load_split_dataset(args.s2s_sup_data)
        vocab_transform, text_transform, src_vocab_size, tar_vocab_size = construct_dataset_meta(train_iter)
        model_params = f"model/s2s_{args.sym_trans}.pth"
        trans_module = Seq2Seq(args.sym_trans,
                               vocab_transform=vocab_transform, text_transform=text_transform,
                               src_vocab_sz=src_vocab_size, tar_vocab_sz=tar_vocab_size, fpath_load=model_params)
    else:
        raise ValueError(f"ERROR: translation module not recognized: {args.sym_trans}")

    # breakpoint()

    placeholder_maps, placeholder_maps_inv = [], []
    for objs in objs_per_utt:
        placeholder_map, placeholder_map_inv = build_placeholder_map(objs, args.convert_rule, PROPS)
        placeholder_maps.append(placeholder_map)
        placeholder_maps_inv.append(placeholder_map_inv)
    trans_queries, _ = substitute(grounded_utts, placeholder_maps, is_utt=True)  # replace names by symbols

    # breakpoint()

    symbolic_ltls = []
    for idx, query in enumerate(trans_queries):
        logging.info(f"Symbolic Translation: {idx}/{len(trans_queries)}")
        query = query.translate(str.maketrans('', '', ',.'))
        query = f"Utterance: {query}\nLTL:"  # query format for finetuned GPT-3
        ltl = trans_module.translate(query, trans_modular_prompt)[0]
        # try:
        #     spot.formula(ltl)
        # except SyntaxError:
        #     ltl = feedback_module(trans_module, query, trans_modular_prompt, ltl)
        symbolic_ltls.append(ltl)

    # breakpoint()

    output_ltls, _ = substitute(symbolic_ltls, placeholder_maps_inv, is_utt=False)  # replace symbols by props

    # breakpoint()

    return output_ltls, symbolic_ltls, placeholder_maps


def translate_e2e(grounded_utts):
    """
    Translation language to LTL using a single GPT-3.
    """
    trans_e2e_prompt = load_from_file(args.trans_e2e_prompt)
    model = GPT3(translation_engine)
    output_ltls = [model.translate(utt, trans_e2e_prompt) for utt in grounded_utts]
    return output_ltls


def feedback_module(trans_module, query, trans_modular_prompt, ltl_incorrect, n=100):
    """
    :param trans_module: model for the translation module.
    :param query: input utterance.
    :param ltl_incorrect:  LTL formula that has syntax error.
    :param trans_modular_prompt: prompt for GPT-3 translation module
    :param n: number of outupt to sample from the translation module.
    :return: LTL formula of correct syntax and most likely to be correct translation of utterance `query'.
    """
    breakpoint()
    logging.info(f"Syntax error: {query} | {ltl_incorrect}")
    if isinstance(trans_module, GPT3):
        trans_module.n = n
        ltls_fix = trans_module.translate(query, trans_modular_prompt)
    else:
        ltls_fix = trans_module.translate(query)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="osm", choices=["osm", "cleanup"], help="environment name.")
    parser.add_argument("--cities", action="store", type=str, nargs="+", default=["philadelphia_2", "new_york_1"], help="list of cities.")
    parser.add_argument("--holdout", type=str, default="utt", choices=["utt", "formula", "type"], help="type of holdout testing.")
    parser.add_argument("--rer", type=str, default="gpt3", choices=["gpt3", "bert"], help="Referring Expressoin Recognition module")
    parser.add_argument("--rer_engine", type=str, default="text-davinci-003", help="pretrained GPT-3 for RER.")
    parser.add_argument("--rer_prompt", type=str, default="data/osm/rer_prompt_16.txt", help="path to RER prompt")
    parser.add_argument("--ground", type=str, default="gpt3", choices=["gpt3", "bert"], help="grounding module")
    parser.add_argument("--embed_engine", type=str, default="text-embedding-ada-002", help="gpt-3 embedding engine")
    parser.add_argument("--topk", type=int, default=2, help="top k similar known names to re")
    parser.add_argument("--sym_trans", type=str, default="gpt3_finetuned", choices=["gpt3_finetuned", "gpt3_pretrained", "t5-base", "t5-small", "pt_transformer"], help="symbolic translation module")
    parser.add_argument("--convert_rule", type=str, default="lang2ltl", choices=["lang2ltl", "cleanup"], help="name to prop conversion rule.")
    parser.add_argument("--full_e2e", action="store_true", help="solve translation and ground end-to-end using GPT-3")
    parser.add_argument("--full_e2e_prompt", type=str, default="data/cleanup_full_e2e_prompt_15.txt", help="path to full end-to-end prompt")
    parser.add_argument("--nsamples", type=int, default=None, help="randomly sample nsamples pairs or None to use all")
    # parser.add_argument("--obj_embed", type=str, default="data/osm/lmk_sem_embeds/obj2embed_boston_text-embedding-ada-002.pkl", help="embedding of known obj in env")
    # parser.add_argument("--name_embed", type=str, default="data/osm/lmk_name_embeds/name2embed_boston_text-embedding-ada-002.pkl", help="embedding of re in language")
    # parser.add_argument("--translation_engine", type=str, default="gpt3_finetuned_symbolic_batch12_perm_utt_0.2_42", help="finetuned or pretrained gpt-3 for symbolic translation.")
    # parser.add_argument("--data_fpath", type=str, default="data/osm/lang2ltl/boston/small_symbolic_batch12_perm_utt_0.2_42.pkl", help="test dataset.")
    # parser.add_argument("--result_dpath", type=str, default="results/lang2ltl/osm/boston", help="dpath to save outputs of each model in a json file")
    # parser.add_argument("--trans_modular_prompt", type=str, default="data/cleanup/cleanup_trans_modular_prompt_15.txt", help="symbolic translation prompt")
    # parser.add_argument("--result_fpath", type=str, default="results/corlw/modular_prompt15_cleanup_test.json", help="file path to save outputs of each model in a json file")
    parser.add_argument("--nruns", type=int, default=1, help="number of runs to test each model")
    parser.add_argument("--trans_e2e", action="store_true", help="solve translation task end-to-end using GPT-3")
    parser.add_argument("--trans_e2e_prompt", type=str, default="data/cleanup_trans_e2e_prompt_15.txt", help="path to translation end-to-end prompt")
    parser.add_argument("--s2s_sup_data", type=str, default="data/symbolic_pairs.csv", help="file path to train and test data for supervised seq2seq")
    parser.add_argument("--true_trajs", type=str, default="data/true_trajs.pkl", help="path to true trajectories")
    parser.add_argument("--debug", action="store_true", help="True to print debug trace.")
    args = parser.parse_args()

    env_dpath = os.path.join("data", args.env)
    env_lmks_dpath = os.path.join(env_dpath, "lmks")

    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join("results", "lang2ltl", f'log_raw_results_{"_".join(args.cities)}_new.log'), mode='w'),
                            logging.StreamHandler()
                        ]
    )

    if args.env == "osm" or args.env == "cleanup":
        # if args.city == "all":
        #     cities = [os.path.splitext(fname)[0] for fname in os.listdir(env_lmks_dpath) if "json" in fname and fname != "boston"]  # Boston dataset for finetune prompt and train baseline
        # else:
        #     cities = [args.city]
        for city in args.cities:
            city_dpath = os.path.join(env_dpath, "lang2ltl", city)
            data_fpaths = [os.path.join(city_dpath, fname) for fname in os.listdir(city_dpath) if fname.startswith("symbolic")]
            data_fpaths = sorted(data_fpaths, reverse=True)

            obj_embed = os.path.join(env_dpath, "lmk_sem_embeds", f"obj2embed_{city}_{args.embed_engine}.pkl")
            name_embed = os.path.join(env_dpath, "lmk_name_embeds", f"name2embed_{city}_{args.embed_engine}.pkl")

            for data_fpath in data_fpaths:
                if "utt" in data_fpath:
                    result_subd = "utt_holdout_batch12"
                elif "formula" in data_fpath:
                    result_subd = "formula_holdout_batch12"
                elif "type" in data_fpath:
                    result_subd = "type_holdout_batch12"
                else:
                    raise ValueError(f"ERROR: unrecognized data fpath\n{data_fpath}")

                result_dpath = os.path.join("results", "lang2ltl", args.env, city, result_subd)
                os.makedirs(result_dpath, exist_ok=True)
                all_result_fpath = os.path.join(result_dpath, f"acc_{Path(data_fpath).stem}.json".replace("symbolic", "grounded"))
                pair_result_fpath = os.path.join(result_dpath, f"acc_{Path(data_fpath).stem}.csv".replace("symbolic", "grounded"))

                if os.path.basename(pair_result_fpath) not in os.listdir(result_dpath) and args.holdout in pair_result_fpath:  # only run unfinished exps of specified holdout type
                    dataset = load_from_file(data_fpath)
                    valid_iter = dataset["valid_iter"]
                    meta_iter = dataset["valid_meta"]

                    input_utts, true_ltls, true_sym_utts, true_sym_ltls, pattern_types, true_names, propositions = [], [], [], [], [], [], []
                    for (utt, ltl), (sym_utt, sym_ltl, pattern_type, props, lmk_names, seed) in zip(valid_iter, meta_iter):
                        input_utts.append(utt)
                        true_ltls.append(ltl)
                        true_sym_utts.append(sym_utt)
                        pattern_types.append(pattern_type)
                        if "restricted_avoidance" in pattern_type:
                            true_sym_ltls.append(substitute_single_letter(sym_ltl, {props[-1]: PROPS[0]}))
                            true_names.append(lmk_names[-1:])
                            propositions.append(props[-1:])
                        else:
                            true_sym_ltls.append(sym_ltl)
                            true_names.append(lmk_names)
                            propositions.append(props)

                    assert len(input_utts) == len(true_ltls) == len(true_sym_utts) == len(true_sym_ltls) == len(pattern_types) == len(true_names) == len(propositions), \
                        f"ERROR: input len != # out len: {len(input_utts)} {len(true_ltls)} {len(true_sym_utts)} {len(true_sym_ltls)} {len(pattern_types)} {len(true_names)} {len(propositions)}"
                    if args.nsamples:  # for testing, randomly sample `nsamples` pairs to cover diversity of dataset
                        random.seed(42)
                        input_utts, true_ltls, true_sym_ltls, pattern_types, true_names, propositions = zip(*random.sample(list(zip(input_utts, true_ltls, true_sym_ltls, pattern_types, true_names, propositions)), args.nsamples))

                    # logging.basicConfig(level=logging.DEBUG,
                    #                     format='%(message)s',
                    #                     handlers=[
                    #                         logging.FileHandler(f'{os.path.splitext(all_result_fpath)[0]}.log', mode='w'),
                    #                         logging.StreamHandler()
                    #                     ]
                    # )

                    logging.info(data_fpath)

                    logging.info(f"RER engine: {args.rer_engine}")
                    logging.info(f"Embedding engine: {args.embed_engine}")

                    if args.sym_trans == "gpt3_finetuned":
                        translation_engine = f"gpt3_finetuned_{Path(data_fpath).stem}"
                        translation_engine = load_from_file("model/gpt3_models.pkl")[translation_engine]
                    elif args.sym_trans == "gpt3_pretrained":
                        translation_engine = "text-davinci-003"
                    else:
                        raise ValueError(f"ERROR: unrecognized symbolic translation model: {args.sym_trans}")
                    logging.info(f"Symbolic translation engine: {translation_engine}")

                    logging.info(f"known lmk embed: {obj_embed}")
                    logging.info(f"cached lmk embed: {name_embed}")

                    # formula2type, formula2prop = find_all_formulas(TYPE2NPROPS, "noperm" in data_fpath)

                    for run in range(args.nruns):
                        logging.info(f"\n\n\nRUN: {run}")
                        run_exp()

    # # Test grounding
    # city_names = [os.path.splitext(fname)[0] for fname in os.listdir("data/osm/lmks") if "json" in fname]
    # filter_cities = ["boston", "chicago_2", "jacksonville_1", "san_diego_2"]
    # city_names = [city for city in city_names if city not in filter_cities]
    # for city in city_names:
    #     obj_embed = f"data/osm/lmk_sem_embeds/obj2embed_{city}_{embed_engine}.pkl"
    #     name_embed = f"data/osm/lmk_name_embeds/name2embed_{city}_{embed_engine}.pkl"
    #     print(obj_embed)
    #     print(name_embed)
    #     breakpoint()
    #     names = list(load_from_file(f"data/osm/lmks/{city}.json").keys())
    #     name2grounds = ground_names(names)
    #     for name, grounds in name2grounds.items():
    #         if name != grounds[0]:
    #             print(f"Landmark name does not match grounding\n{name}\n{grounds}\n\n")
