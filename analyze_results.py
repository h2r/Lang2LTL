import os
from pathlib import Path
import argparse
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

from utils import load_from_file, deserialize_props_str
from formula_sampler import TYPE2NPROPS, sample_formulas


def plot_cm(results_fpath, type2nprops, debug):
    print(f"plotting confusion matrix for:\n{Path(results_fpath)}")
    cm, all_types = construct_cm(results_fpath, type2nprops, debug)

    if "utt" in results_fpath:
        holdout_type = "Utterance Holdout"
    elif "ltl_formula" in results_fpath:
        holdout_type = "Formula Holdout"
    elif "ltl_type" in results_fpath:
        holdout_type = "Type Holdout"
    else:
        raise ValueError(f"ERROR: unrecognized holdout type in results file:\n{results_fpath}")

    df_cm = pd.DataFrame(cm, index=all_types, columns=all_types)
    plt.figure(figsize=(8, 6))
    plt.title(f"Misclassified Types in {holdout_type}")
    plt.xlabel("Prediction")
    plt.ylabel("True")
    sns.set(font_scale=0.8)
    sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
    fig_fpath = os.path.join(os.path.dirname(results_fpath), f"fig_{'_'.join(Path(results_fpath).stem.split('_')[1:])}.png")  # remove results ID: "log"
    plt.savefig(fig_fpath)
    plt.show()


def construct_cm(results_fpath, type2nprops, debug):
    formula2type, formula2prop = find_all_formulas(type2nprops, "noperm" in results_fpath)
    if debug:
        print(f"Total number of unique LTL formulas: {len(formula2type)}, {len(formula2prop)}")
        print(f"Number of LTL type: {len(set(formula2type.values()))}")

    # pprint(formula2type)
    # breakpoint()

    y_true, y_pred = [], []
    type2errs = defaultdict(list)
    total_errs = 0
    results = load_from_file(results_fpath)
    for idx, result in enumerate(results):
        if debug:
            print(f"result {idx}")
        result = result[1:]  # remove train_or_valid
        pattern_type, nprops, true_prop_perm_str, utt, true_ltl, output_ltl, is_correct = result
        if is_correct != "True":
            total_errs += 1
        true_prop_perm = deserialize_props_str(true_prop_perm_str)

        if is_correct == "Syntax Error":
            type2errs["syntax_errors"].append(result)
        else:
            if output_ltl in formula2type:
                pred_prop_perm = formula2prop[output_ltl]

                if formula2type[true_ltl] != formula2type[output_ltl]:
                    type2errs["misclassified_type"].append(result)
                    # print(f"Misclassified Type:\n{pattern_type}, {nprops}, {true_prop_perm_str}\n{utt}\n{true_ltl}\n{output_ltl}\n{pattern_type} {formula2type[output_ltl]}\n")
                    # breakpoint()
                    y_true.append(pattern_type)
                    y_pred.append(formula2type[output_ltl])
                elif len(true_prop_perm) != len(pred_prop_perm):
                    type2errs["incorrect_nprops"].append(result)
                    # print(f"Incorrect Nprops:\n{pattern_type}, {nprops}, {true_prop_perm_str}\n{utt}\n{true_ltl}\n{output_ltl}\n{true_prop_perm}\n{pred_prop_perm}")
                    # breakpoint()
                elif sorted(true_prop_perm) != sorted(pred_prop_perm):
                    type2errs["incorrect_props"].append(result)
                    # print(f"Incorrect Props:\n{pattern_type}, {nprops}, {true_prop_perm_str}\n{utt}\n{true_ltl}\n{output_ltl}\n{sorted(true_prop_perm)}, {sorted(pred_prop_perm)}")
                    # breakpoint()
                elif true_prop_perm != pred_prop_perm and is_correct != "True":  # diff prop order and not spot equivalent, e.g visit 2
                    type2errs["incorrect_orders"].append(result)
                    # print(f"Incorrect Prop Order:\n{pattern_type}, {nprops}, {true_prop_perm_str}\n{utt}\n{true_ltl}\n{output_ltl}\n")
                    # breakpoint()
                else:
                    y_true.append(pattern_type)
                    y_pred.append(formula2type[output_ltl])
            else:
                type2errs["unknow_types"].append(result)
                # print(f"Unknown Type:\n{pattern_type}, {nprops}, {true_prop_perm_str}\n{utt}\n{true_ltl}\n{output_ltl}\n")
                # breakpoint()

    print(f"total number of errors:\t{total_errs}/{len(results)}\t= {total_errs/len(results)}")
    type2errs_sorted = sorted(type2errs.items(), key=lambda kv: len(kv[1]), reverse=True)
    nerrs_caught = 0
    for typ, errs in type2errs_sorted:
        print(f"number of {typ}:\t{len(errs)}/{total_errs}\t= {len(errs)/total_errs}")
        nerrs_caught += len(errs)
    if total_errs != nerrs_caught:
        raise ValueError(f"total nerrors != nerrs_caught: {total_errs} != {nerrs_caught}")

    all_types = sorted(type2nprops.keys(), reverse=True)
    cm = confusion_matrix(y_true, y_pred, labels=all_types)

    return cm, all_types


def find_all_formulas(type2nprops, perm):
    formula2type, formula2prop = {}, {}
    for pattern_type, all_nprops in type2nprops.items():
        for nprops in all_nprops:
            formulas, props_perm = sample_formulas(pattern_type, nprops, False)
            if perm:
                formula2type[formulas[0]] = pattern_type
                formula2prop[formulas[0]] = list(props_perm[0])
            else:
                for formula, prop_perm in zip(formulas, props_perm):
                    formula2type[formula] = pattern_type
                    formula2prop[formula] = list(prop_perm)
    return formula2type, formula2prop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_fpath", type=str, default="results/finetuned_gpt3/utt_holdout_batch12_perm/log_gpt3_finetuned_symbolic_batch12_perm_utt_0.2_42.csv", help="fpath of results to analyze.")
    parser.add_argument("--split_fpath", type=str, default="data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_42.pkl", help="fpath to split dataset used to produce results.")
    parser.add_argument("--debug", action="store_true", help="True to print debug trace.")
    args = parser.parse_args()

    plot_cm(args.results_fpath, TYPE2NPROPS, args.debug)
