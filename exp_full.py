"""
Evaluate full translation system including referring expression recognition, proposition resolution and symbolic translation.
"""
import os
import argparse
import logging
from pathlib import Path
import random
import numpy as np
import spot

from lang2ltl import rer, ground_res, ground_utterances, translate_modular, PROPS
from gpt import GPT3, GPT4
from s2s_hf_transformers import HF_MODELS
from utils import load_from_file, save_to_file, substitute_single_letter
from eval import evaluate_lang_0, evaluate_lang, evaluate_plan
from formula_sampler import TYPE2NPROPS
from analyze_results import find_all_formulas


def run_exp():
    # Language tasks: RER, proposition resolution, symbolic translation
    if args.full_e2e:  # Full translation from language to LTL
        if args.full_e2e == "gpt3":
            translation_engine = "text-davinci-003"
            full_e2e_module = GPT3(translation_engine)
        elif args.full_e2e == "gpt4":
            translation_engine = "gpt-4"
            full_e2e_module = GPT4(translation_engine)
            breakpoint()
        else:
            raise ValueError(f"ERROR: unrecognized full translation model: {args.full_e2e}")
        logging.info(f"Full translation engine: {translation_engine}")

        full_e2e_prompt = load_from_file(full_e2e_prompt_fpath)
        out_ltls = []
        for idx, query in enumerate(input_utts):
            query = f"{full_e2e_prompt} {query}\nLTL:"
            out_ltl = full_e2e_module.translate(query)[0]
            out_ltls.append(out_ltl)
            logging.info(f"Full Translation{idx}:\n{query}\n{out_ltl}")

        accs, accumulated_acc = evaluate_lang_0(true_ltls, out_ltls, string_match=True)
        pair_results = [["Utterance", "True LTL", "Out LTL", "Accuracy"]]
        for idx, (input_utt, true_ltl, output_ltl, acc) in enumerate(zip(input_utts, true_ltls, out_ltls, accs)):
            logging.info(f"{idx}\nInput utterance: {input_utt}\nTrue LTL: {true_ltl}\nOutput LTL: {output_ltl}\n{acc}\n")
            pair_results.append((input_utt, true_ltl, output_ltl, acc))
        logging.info(f"Language to LTL translation accuracy: {accumulated_acc}")
        save_to_file(pair_results, pair_result_fpath)
    else:  # Modular
        logging.info(f"RER engine: {args.rer_engine}")
        logging.info(f"Embedding engine: {args.embed_engine}")
        logging.info(f"known obj embed: {obj_embed}")
        logging.info(f"cached RE embed: {re_embed}")

        res, utt2res = rer(args.rer, args.rer_engine, args.rer_prompt, input_utts)
        out_res = [utt_res[1] for utt_res in utt2res]  # referring expressions
        re2grounds = ground_res(res, re_embed, obj_embed, args.ground, args.embed_engine, args.topk)
        grounded_utts, objs_per_utt = ground_utterances(input_utts, utt2res, re2grounds)  # ground res to known objects in env

        if args.sym_trans in HF_MODELS:
            checkpoint = load_from_file(args.model2ckpt_fpath)[args.sym_trans]
            translation_engine = os.path.join(args.model_dpath, args.sym_trans, f"checkpoint-{checkpoint}")
        elif args.sym_trans == "gpt3_finetuned":
            translation_engine = f"gpt3_finetuned_{Path(data_fpath).stem}"
            translation_engine = load_from_file("model/gpt3_models.pkl")[translation_engine]
        elif args.sym_trans == "gpt3_pretrained":
            translation_engine = "text-davinci-003"
        else:
            raise ValueError(f"ERROR: unrecognized symbolic translation model: {args.sym_trans}")

        if args.trans_e2e:
            logging.info(f"End-to-end translation engine: {translation_engine}")
            out_ltls = translate_e2e(grounded_utts, translation_engine)
        else:
            logging.info(f"Symbolic translation engine: {translation_engine}")
            sym_utts, out_sym_ltls, out_ltls, placeholder_maps = translate_modular(grounded_utts, objs_per_utt, args.sym_trans, translation_engine, args.convert_rule, PROPS)

            # out_sym_ltls_sub = []
            # for props, out_sym_ltl, placeholder_map in zip(propositions, out_sym_ltls, placeholder_maps.items()):
            #     out_sym_ltls_sub.append(substitute_single_letter(out_sym_ltl, {letter: prop for (_, letter), prop in zip(placeholder_map.items(), props)}))
            # out_sym_ltls = out_sym_ltls_sub

            accs, accumulated_acc = evaluate_lang(true_ltls, out_ltls, true_res, out_res, objs_per_utt, args.convert_rule, PROPS)
            # accs, accumulated_acc = evaluate_lang_new(true_ltls, out_ltls, true_sym_ltls, out_sym_ltls, true_res, out_res, objs_per_utt)

            pair_results = [["Pattern Type", "Propositions", "Utterance", "Symolic Utterance", "True LTL", "Out LTL", "True Symbolic LTL", "Out Symbolic LTL", "True Lmks", "Out Lmks", "Out Lmk Ground", "Placeholder Map", "Accuracy"]]
            for idx, (pattern_type, props, in_utt, sym_utt, true_ltl, out_ltl, true_sym_ltl, out_sym_ltl, true_re, out_re, out_grnd, placeholder_maps, acc) in enumerate(zip(pattern_types, propositions, input_utts, sym_utts, true_ltls, out_ltls, true_sym_ltls, out_sym_ltls, true_res, out_res, objs_per_utt, placeholder_maps, accs)):
                logging.info(f"{idx}\n{pattern_type} {props}\nInput utterance: {in_utt}\nSymbolic utterance: {sym_utt}\n"
                             f"True Ground LTL: {true_ltl}\nOut Ground LTL: {out_ltl}\n"
                             f"True Symbolic LTL: {true_sym_ltl}\nOut Symbolic LTL: {out_sym_ltl}\n"
                             f"True REs: {true_re}\nOut REs:{out_re}\nOut Grounds: {out_grnd}\nPlaceholder Map: {placeholder_maps}\n"
                             f"{acc}\n")
                pair_results.append((pattern_type, props, in_utt, sym_utt, true_ltl, out_ltl, true_sym_ltl, out_sym_ltl, true_re, out_re, out_grnd, placeholder_maps, acc))
            logging.info(f"Language to LTL translation accuracy: {accumulated_acc}\n\n")
            save_to_file(pair_results, pair_result_fpath)

    if len(input_utts) != len(out_ltls):
        logging.info(f"ERROR: # input utterances {len(input_utts)} != # output LTLs {len(out_ltls)}")

    all_results = {
        "RER": utt2res if not args.full_e2e else None,
        "Grounding": re2grounds if not args.full_e2e else None,
        "Placeholder maps": placeholder_maps if not (args.trans_e2e or args.full_e2e) else None,
        "Input utterances": input_utts,
        "Symbolic utterances": sym_utts if not (args.trans_e2e or args.full_e2e) else None,
        "Output Symbolic LTLs": out_sym_ltls if not (args.trans_e2e or args.full_e2e) else None,
        "Output Grounded LTLs": out_ltls,
        "True Grounded LTLs": true_ltls,
        "Meta": valid_meta,
        "Accuracies": accs,
        "Accumulated Accuracy": accumulated_acc
    }
    save_to_file(all_results, all_result_fpath)
    save_to_file(all_results, os.path.join(os.path.dirname(all_result_fpath), f"{Path(all_result_fpath).stem}.pkl"))  # also save to pkl to preserve data type

    # Planning task: LTL + MDP -> policy
    # true_trajs = load_from_file(args.true_trajs)
    # acc_plan = plan(output_ltls, re2grounds)
    # logging.info(f"Planning accuracy: {acc_plan}")


def translate_e2e(grounded_utts, translation_engine):
    """
    Translation language to LTL using a single GPT-3.
    """
    trans_e2e_prompt = load_from_file(args.trans_e2e_prompt)
    model = GPT3(translation_engine)
    output_ltls = [model.translate(utt, trans_e2e_prompt)[0] for utt in grounded_utts]
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


def plan(output_ltls, true_trajs, re2grounds):
    """
    Planning with translated LTL as task specification
    """
    accs = []
    planner = None
    for out_ltl, true_traj in zip(output_ltls, true_trajs):
        out_traj = planner.plan(out_ltl, re2grounds)
        accs.append(evaluate_plan(out_traj, true_traj))
    acc = np.mean(accs)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="osm", choices=["osm", "cleanup"], help="domain name.")
    parser.add_argument("--envs", action="store", type=str, nargs="+", default=["boston"], help="list of envs.")
    parser.add_argument("--holdout", type=str, default="utt", choices=["utt", "formula", "type", None], help="type of holdout test or None for all types.")
    parser.add_argument("--rer", type=str, default="gpt3", choices=["gpt3", "gpt4", "llama-7B"], help="Referring Expressoin Recognition module.")
    parser.add_argument("--rer_engine", type=str, default="text-davinci-003", choices=["text-davinci-003", "gpt-4"], help="GPT engine for RER.")
    parser.add_argument("--rer_prompt", type=str, default="data/osm/rer_prompt_16.txt", help="path to RER prompt.")
    parser.add_argument("--ground", type=str, default="gpt3", choices=["gpt3"], help="grounding module.")
    parser.add_argument("--embed_engine", type=str, default="text-embedding-ada-002", help="gpt-3 embedding engine.")
    parser.add_argument("--topk", type=int, default=2, help="top k similar known obj names to re.")
    parser.add_argument("--sym_trans", type=str, default="t5-base", choices=["t5-base", "gpt3_finetuned", "gpt3_pretrained"], help="symbolic translation module.")
    parser.add_argument("--model_dpath", type=str, default=None, help="directory to model checkpoints.")
    parser.add_argument("--model2ckpt_fpath", type=str, default=None, help="best checkpoint for models.")
    parser.add_argument("--convert_rule", type=str, default="lang2ltl", choices=["lang2ltl", "cleanup"], help="re to prop conversion rule.")
    parser.add_argument("--full_e2e", type=str, default="gpt4", choices=["gpt3", "gpt4", "llama-7B", None], help="solve full translation using LLM.")
    parser.add_argument("--nexamples", type=int, default=1, help="number of examples per formula in prompt.")
    parser.add_argument("--nsamples", type=int, default=None, help="randomly sample nsamples pairs or None to use all.")
    # parser.add_argument("--trans_modular_prompt", type=str, default="data/cleanup/cleanup_trans_modular_prompt_15.txt", help="symbolic translation prompt.")
    parser.add_argument("--trans_e2e", action="store_true", help="solve translation task end-to-end using GPT-3.")
    parser.add_argument("--trans_e2e_prompt", type=str, default="data/cleanup_trans_e2e_prompt_15.txt", help="path to translation end-to-end prompt.")
    parser.add_argument("--s2s_sup_data", type=str, default="data/symbolic_pairs.csv", help="file path to train and test data for supervised seq2seq.")
    parser.add_argument("--true_trajs", type=str, default="data/true_trajs.pkl", help="path to true trajectories.")
    parser.add_argument("--nruns", type=int, default=1, help="number of runs to test each model.")
    parser.add_argument("--debug", action="store_true", help="True to print debug trace.")
    args = parser.parse_args()

    domain_dpath = os.path.join("data", args.domain)
    domain_lmks_dpath = os.path.join(domain_dpath, "lmks")

    e2e_id = "e2e" if args.full_e2e else ""
    log_dpath = os.path.join("results", "lang2ltl", args.domain, "log")
    os.makedirs(log_dpath, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(log_dpath, f'log_raw_results_{e2e_id}_{"_".join(args.envs)}_{args.holdout}_nexamples{args.nexamples}.log'), mode='w'),
                            logging.StreamHandler()
                        ]
    )

    if args.domain == "osm" or args.domain == "cleanup":  # TODO: separate exp_full_DOMIN.py for each DOMAIN
        # if args.envs == "all":
        #     envs = [os.path.splitext(fname)[0] for fname in os.listdir(domain_lmks_dpath) if "json" in fname and fname != "boston"]  # Boston dataset for finetune prompt and train baseline
        # else:
        #     envs = [args.envs]
        for env in args.envs:
            env_dpath = os.path.join(domain_dpath, "lang2ltl", env)
            data_fpaths = [os.path.join(env_dpath, fname) for fname in os.listdir(env_dpath) if fname.startswith("symbolic")]
            if args.holdout:
                data_fpaths = [data_fpath for data_fpath in data_fpaths if args.holdout in data_fpath]
            data_fpaths = sorted(data_fpaths, reverse=True)[2:]

            obj_embed = os.path.join(domain_dpath, "lmk_sem_embeds", f"obj2embed_{env}_{args.embed_engine}.pkl")
            re_embed = os.path.join(domain_dpath, "re_embeds", f"re2embed_{env}_{args.embed_engine}.pkl")

            for data_fpath in data_fpaths:
                if "utt" in data_fpath:
                    result_subd = "utt_holdout_batch12"
                elif "ltl_formula" in data_fpath:
                    result_subd = "formula_holdout_batch12"
                elif "ltl_type" in data_fpath:
                    result_subd = "type_holdout_batch12"
                else:
                    raise ValueError(f"ERROR: unrecognized data fpath\n{data_fpath}")
                if args.full_e2e:
                    result_dpath = os.path.join("results", "lang2ltl", args.domain, "e2e", args.full_e2e, env, result_subd)
                    full_e2e_prompt_fpath = os.path.join(domain_dpath, "full_translation_prompt", env, f"prompt_nexamples{args.nexamples}_{Path(data_fpath).stem}.txt")
                else:
                    result_dpath = os.path.join("results", "lang2ltl", args.domain, env, result_subd)
                os.makedirs(result_dpath, exist_ok=True)
                all_result_fpath = os.path.join(result_dpath, f"acc_{Path(data_fpath).stem}.json".replace("symbolic", "grounded"))
                pair_result_fpath = os.path.join(result_dpath, f"acc_{Path(data_fpath).stem}.csv".replace("symbolic", "grounded"))

                if os.path.basename(pair_result_fpath) not in os.listdir(result_dpath) and args.holdout in pair_result_fpath:  # only run unfinished exps of specified holdout type
                    dataset = load_from_file(data_fpath)
                    valid_iter, valid_meta = dataset["valid_iter"], dataset["valid_meta"]
                    if args.nsamples:  # for testing, randomly sample `nsamples` pairs to cover diversity of dataset
                        random.seed(42)
                        valid_iter, valid_meta = zip(*random.sample(list(zip(valid_iter, valid_meta)), args.nsamples))
                    logging.info(f"{data_fpath}\ntest set size: {len(valid_iter)}, {len(valid_meta)}")

                    breakpoint()

                    input_utts, true_ltls, true_sym_utts, true_sym_ltls, pattern_types, true_res, propositions = [], [], [], [], [], [], []
                    for (utt, ltl), (sym_utt, sym_ltl, pattern_type, props, res, seed) in zip(valid_iter, valid_meta):
                        input_utts.append(utt)
                        true_ltls.append(ltl)
                        true_sym_utts.append(sym_utt)
                        pattern_types.append(pattern_type)
                        if "restricted_avoidance" in pattern_type:  # X_restricted_avoidance formulas have only 1 prop
                            true_sym_ltls.append(substitute_single_letter(sym_ltl, {props[-1]: PROPS[0]}))
                            true_res.append(res[-1:])
                            propositions.append(props[-1:])
                        else:
                            true_sym_ltls.append(sym_ltl)
                            true_res.append(res)
                            propositions.append(props)

                    assert len(input_utts) == len(true_ltls) == len(true_sym_utts) == len(true_sym_ltls) == len(pattern_types) == len(true_res) == len(propositions), \
                        f"ERROR: input len != # out len: {len(input_utts)} {len(true_ltls)} {len(true_sym_utts)} {len(true_sym_ltls)} {len(pattern_types)} {len(true_res)} {len(propositions)}"

                    # logging.basicConfig(level=logging.DEBUG,
                    #                     format='%(message)s',
                    #                     handlers=[
                    #                         logging.FileHandler(f'{os.path.splitext(all_result_fpath)[0]}.log', mode='w'),
                    #                         logging.StreamHandler()
                    #                     ]
                    # )

                    # formula2type, formula2prop = find_all_formulas(TYPE2NPROPS, "noperm" in data_fpath)

                    for run in range(args.nruns):
                        logging.info(f"\n\n\nRUN: {run}")
                        run_exp()

    # # Test grounding
    # env_names = [os.path.splitext(fname)[0] for fname in os.listdir("data/osm/lmks") if "json" in fname]
    # filter_envs = ["boston", "chicago_2", "jacksonville_1", "san_diego_2"]
    # env_names = [env for env in env_names if env not in filter_envs]
    # for env in env_names:
    #     obj_embed = f"data/osm/lmk_sem_embeds/obj2embed_{env}_{embed_engine}.pkl"
    #     re_embed = f"data/osm/re_embeds/re2embed_{env}_{embed_engine}.pkl"
    #     print(obj_embed)
    #     print(re_embed)
    #     breakpoint()
    #     res = list(load_from_file(f"data/osm/lmks/{env}.json").keys())
    #     re2grounds = ground_res(res)
    #     for re, grounds in re2grounds.items():
    #         if re != grounds[0]:
    #             print(f"Landmark name does not match grounding\n{re}\n{grounds}\n\n")
