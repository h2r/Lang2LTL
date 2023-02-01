import os
from pathlib import Path
import argparse
import logging
from collections import defaultdict
import numpy as np
import spot
from pprint import pprint

from gpt3 import GPT3
from dataset_symbolic import load_split_dataset
from utils import load_from_file, save_to_file, name_to_prop, substitute_single_word


def evaluate_lang(true_ltls, out_ltls, true_names, out_names, out_grnds, convert_rule, all_props):
    accs = []
    for true_ltl, out_ltl, true_name, out_name, out_grnd in zip(true_ltls, out_ltls, true_names, out_names, out_grnds):
        if out_ltl == true_ltl:  # Spot cannot handle long but correct LTL formula, e.g. F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F 62_on_the_park
            is_correct = "True"
        else:
            try:  # output LTL formula may have syntax error
                spot_correct = spot.are_equivalent(spot.formula(true_ltl), spot.formula(out_ltl))
                is_correct = "True" if spot_correct else "False"
            except SyntaxError:
                logging.info(f"Syntax error OR formula too long:\n{true_ltl}\n{out_ltl}")
                # breakpoint()

                if set(true_name) == set(out_grnd):
                    true_props = [name_to_prop(name, convert_rule) for name in true_name]
                    true_sub_map = {prop: sym for prop, sym in zip(true_props, all_props[:len(true_props)])}
                    true_ltl_short = substitute_single_word(true_ltl, true_sub_map)[0]

                    out_props = [name_to_prop(name, convert_rule) for name in true_name]
                    out_sub_map = {prop: sym for prop, sym in zip(out_props, all_props[:len(out_props)])}
                    out_ltl_short = substitute_single_word(out_ltl, out_sub_map)[0]

                    logging.info(f"shorten LTLs:\n{true_ltl_short}\n{out_ltl_short}\n")
                    try:  # output LTL formula may have syntax error
                        spot_correct = spot.are_equivalent(spot.formula(true_ltl_short), spot.formula(out_ltl_short))
                        is_correct = "True" if spot_correct else "False"
                    except SyntaxError:
                        logging.info(f"Syntax error:\n{true_ltl_short}\n{out_ltl_short}\n")
                        # breakpoint()

                        is_correct = "Syntax Error"
                else:
                    is_correct = "RER or Grounding Error"
        accs.append(is_correct)
    acc = np.mean([True if acc == "True" else False for acc in accs])
    return accs, acc


def evaluate_lang_new(true_ltls, out_ltls, true_sym_ltls, out_sym_ltls, true_names, out_names, out_grnds):
    accs = []
    for true_ltl, out_ltl, true_sym_ltl, out_sym_ltl, true_name, out_name, out_grnd in zip(true_ltls, out_ltls, true_sym_ltls, out_sym_ltls, true_names, out_names, out_grnds):
        if true_ltl == out_ltl:
            is_correct = "True"
        else:
            try:  # output LTL formula may have syntax error
                spot_correct = spot.are_equivalent(spot.formula(true_sym_ltl), spot.formula(out_sym_ltl))
                if spot_correct:
                    if set(true_name) == set(out_name):  # TODO: check only work if RE == lmk_name when generate grounded dataset
                        if set(true_name) == set(out_grnd):
                            is_correct = "True"
                        else:
                            is_correct = "Grounding Error"
                    else:
                        is_correct = "RER Error"
                else:
                    is_correct = "Symbolic Translation Error"
                    if set(true_name) != set(out_name):
                        is_correct += " | RER Error"
                    if set(true_name) != set(out_grnd):
                        is_correct += " | Grounding Error"
            except SyntaxError:
                logging.info(f"Syntax error: {true_sym_ltl}\n{out_sym_ltl}\n")
                is_correct = "Syntax Error"
        accs.append(is_correct)
    acc = np.mean([True if acc == "True" else False for acc in accs])
    return accs, acc


# def evaluate_lang(output_ltls, true_ltls):
#     """
#     Parse LTL formulas in infix or prefix (spot.formula) then check semantic equivalence (spot.are_equivalent).
#     """
#     # TODO: catch errors
#
#     accs = []
#     for out_ltl, true_ltl in zip(output_ltls, true_ltls):
#         if out_ltl == true_ltl:  # Spot cannot handle long but correct LTL formula, e.g. F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F 62_on_the_park
#             accs.append("True")
#         else:
#             try:  # output LTL formula may have syntax error
#                 is_correct = spot.are_equivalent(spot.formula(out_ltl), spot.formula(true_ltl))
#                 is_correct = "True" if is_correct else "False"
#             except SyntaxError:
#                 logging.info(f"Syntax error: {true_ltl}\n{out_ltl}\n")
#                 is_correct = "Syntax Error"
#             accs.append(is_correct)
#     acc = np.mean([True if acc == "True" else False for acc in accs])
#     return accs, acc


def evaluate_lang_single(model, valid_iter, valid_meta, analysis_fpath, result_log_fpath, acc_fpath, valid_iter_len):
    """
    Evaluate translation accuracy per LTL pattern type.
    """
    result_log = [["train_or_valid", "pattern_type", "nprops", "prop_perm", "utterances", "true_ltl", "output_ltl", "is_correct"]]

    meta2accs = defaultdict(list)
    for idx, ((utt, true_ltl), (pattern_type, prop_perm)) in enumerate(zip(valid_iter, valid_meta)):
        nprops = len(prop_perm)
        train_or_valid = "valid" if idx < valid_iter_len else "train"  # TODO: remove after having enough data
        out_ltl = model.translate([utt])[0].strip()
        try:  # output LTL formula may have syntax error
            is_correct = spot.are_equivalent(spot.formula(out_ltl), spot.formula(true_ltl))
            is_correct = "True" if is_correct else "False"
        except SyntaxError:
            is_correct = "Syntax Error"
        logging.info(f"{idx}/{len(valid_iter)}\n{pattern_type} | {nprops} {prop_perm}\n{utt}\n{true_ltl}\n{out_ltl}\n{is_correct}\n")
        result_log.append([train_or_valid, pattern_type, nprops, prop_perm, utt, true_ltl, out_ltl, is_correct])
        if train_or_valid == "valid":
            meta2accs[(pattern_type, nprops)].append(is_correct)

    save_to_file(result_log, result_log_fpath)

    meta2acc = {meta: np.mean([True if acc == "True" else False for acc in accs]) for meta, accs in meta2accs.items()}
    logging.info(meta2acc)

    analysis = load_from_file(analysis_fpath)
    acc_anaysis = [["LTL Type", "Number of Propositions", "Number of Utterances", "Accuracy"]]
    for pattern_type, nprops, nutts in analysis:
        pattern_type = "_".join(pattern_type.lower().split())
        meta = (pattern_type, int(nprops))
        if meta in meta2acc:
            acc_anaysis.append([pattern_type, nprops, nutts, meta2acc[meta]])
        else:
            acc_anaysis.append([pattern_type, nprops, nutts, "no valid data"])
    save_to_file(acc_anaysis, acc_fpath)

    total_acc = np.mean([True if acc == "True" else False for accs in meta2accs.values() for acc in accs])
    logging.info(f"total validation accuracy: {total_acc}")

    return meta2acc, total_acc


def evaluate_lang_from_file(model, split_dataset_fpath, analysis_fpath, result_log_fpath, acc_fpath):
    _, _, valid_iter, valid_meta = load_split_dataset(split_dataset_fpath)
    return evaluate_lang_single(model, valid_iter, valid_meta,
                                analysis_fpath, result_log_fpath, acc_fpath, len(valid_iter))


def aggregate_results(result_fpaths, filter_types):
    """
    Aggregate accuracy-per-formula results from K-fold cross validation or multiple random seeds.
    Assume files have same columns (LTL Type, Number of Propositions, Number of Utterances, Accuracy)
    and same values for first 3 columns.
    :param result_fpaths: paths to results file to be aggregated
    """
    total_corrects, total_samples = 0, 0
    accs = []
    meta2stats = defaultdict(list)
    for n, result_fpath in enumerate(result_fpaths):
        result = load_from_file(result_fpath, noheader=True)
        print(result_fpath)
        corrects, samples = 0, 0
        for row_idx, row in enumerate(result):
            pattern_type, nprops, nutts, acc = row
            if pattern_type not in filter_types and acc != "no valid data":
                nprops, nutts, acc = int(nprops), int(nutts), float(acc)
                meta2stats[(pattern_type, nprops)].append((nutts*acc, nutts))
                corrects += nutts * acc
                samples += nutts
        total_corrects += corrects
        total_samples += samples
        accs.append(corrects / samples)

    result_aux = load_from_file(result_fpaths[0], noheader=False)
    fields = result_aux.pop(0)
    aggregated_result = [fields]
    for row in result_aux:
        aggregated_result.append(row[:3] + [0.0])
    for row_idx, (pattern_type, nprops, nutts, _) in enumerate(aggregated_result[1:]):
        nprops, nutts = int(nprops), int(nutts)
        stats = meta2stats[(pattern_type, nprops)]
        corrects = sum([corrects_formula for corrects_formula, _ in stats])
        nutts = sum([nutts_formula for _, nutts_formula in stats])
        acc = corrects / nutts if nutts != 0 else "no valid data"
        aggregated_result[row_idx+1] = [pattern_type, nprops, nutts, acc]

    result_fnames = [os.path.splitext(result_fpath)[0] for result_fpath in result_fpaths]
    aggregated_result_fpath = f"{os.path.commonprefix(result_fnames)}_aggregated.csv"
    save_to_file(aggregated_result, aggregated_result_fpath)
    accumulated_acc = total_corrects / total_samples
    accumulated_std = np.std(accs)
    print(f"total accuracy: {accumulated_acc}")
    print(f'standard deviation: {accumulated_std}')
    return accumulated_acc, accumulated_std


def evaluate_rer(out_lmks_str, true_lmks):
    out_lmks = out_lmks_str.split(" | ")
    return set(out_lmks) == set(true_lmks)


def evaluate_plan(out_traj, true_traj):
    return out_traj == true_traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_fpath", type=str, default="data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_111.pkl", help="path to pkl file storing train set")
    parser.add_argument("--test_dataset_fpath", type=str, default="data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_111.pkl", help="path to pkl file storing test set")
    parser.add_argument("--analysis_fpath", type=str, default="data/analysis_symbolic_batch12_perm.csv", help="path to dataset analysis")
    parser.add_argument("--model", type=str, default="gpt3_finetuned_symbolic_batch12_perm_utt_0.2_111", help="name of model to be evaluated")
    parser.add_argument("--nexamples", type=int, default=1, help="number of examples per instance for GPT-3")
    parser.add_argument("--aggregate", action="store_true", help="whether to aggregate results or compute new results.")
    args = parser.parse_args()
    dataset_name = Path(args.train_dataset_fpath).stem

    if args.aggregate:  # aggregate acc-per-formula result files
        result_dpath = "results/finetuned_gpt3/formula_holdout_batch12_perm"
        result_fpaths = [os.path.join(result_dpath, fname) for fname in os.listdir(result_dpath) if "acc" in fname and "csv" in fname and "aggregated" not in fname]
        filter_types = ["fair_visit"]
        accumulated_acc, accumulated_std = aggregate_results(result_fpaths, filter_types)
        print("Please verify results files")
        pprint(result_fpaths)
    else:
        if "gpt3" in args.model or "davinci" in args.model:  # gpt3 for finetuned gpt3, davinci for off-the-shelf gpt3
            dataset = load_from_file(args.train_dataset_fpath)
            test_dataset = load_from_file(args.test_dataset_fpath)
            valid_iter = test_dataset["valid_iter"]
            dataset["valid_meta"] = test_dataset["valid_meta"]
            if "utt" in args.train_dataset_fpath:  # results directory based on holdout type
                dname = "utt_holdout_batch12_perm"
            elif "formula" in args.train_dataset_fpath:
                dname = "formula_holdout_batch12_perm"
            elif "type" in args.train_dataset_fpath:
                dname = "type_holdout_batch12_perm"
            if "finetuned" in args.model:
                engine = load_from_file("model/gpt3_models.pkl")[args.model]
                valid_iter = [(f"Utterance: {utt}\nLTL:", ltl) for utt, ltl in valid_iter]
                result_dpath = os.path.join("results", "finetuned_gpt3", dname)
                os.makedirs(result_dpath, exist_ok=True)
                result_log_fpath = os.path.join(result_dpath, f"log_{args.model}.csv")  # fintuned model name already contains dataset name
                acc_fpath = os.path.join(result_dpath, f"acc_{args.model}.csv")
            else:
                engine = args.model
                prompt_fpath = os.path.join("data", "prompt_symbolic_batch12_perm", f"prompt_nexamples{args.nexamples}_{dataset_name}.txt")
                prompt = load_from_file(prompt_fpath)
                valid_iter = [(f"{prompt} {utt}\nLTL:", ltl) for utt, ltl in valid_iter]
                result_dpath = os.path.join("results", "pretrained_gpt3", dname)
                os.makedirs(result_dpath, exist_ok=True)
                result_log_fpath = os.path.join(result_dpath, f"log_{args.model}_{dataset_name}.csv")
                acc_fpath = os.path.join(result_dpath, f"acc_{args.model}_{dataset_name}.csv")
            dataset["valid_iter"] = valid_iter
            split_dataset_fpath = os.path.join("data", "gpt3", f"{dataset_name}.pkl")
            save_to_file(dataset, split_dataset_fpath)
            model = GPT3(engine, temp=0, max_tokens=128)
        else:
            raise ValueError(f"ERROR: model not recognized: {args.model}")

        logging.basicConfig(level=logging.DEBUG,
                            format='%(message)s',
                            handlers=[
                                logging.FileHandler(f'{os.path.splitext(result_log_fpath)[0]}.log', mode='w'),
                                logging.StreamHandler()
                            ]
        )

        evaluate_lang_from_file(model, split_dataset_fpath, args.analysis_fpath, result_log_fpath, acc_fpath)
