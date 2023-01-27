from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint

from utils import load_from_file, deserialize_props_str
from formula_sampler import TYPE2NPROPS, sample_formulas


def plot_cm(results_fpath, type2nprops):
    print(f"plotting confusion matrix:\n{Path(results_fpath)}")
    cm, all_types  = construct_cm(results_fpath, type2nprops)

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
    plt.title(holdout_type)
    plt.xlabel("Prediction")
    plt.ylabel("True")
    sns.set(font_scale=0.8)
    sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
    plt.savefig("results/fig.png")
    plt.show()


def construct_cm(results_fpath, type2nprops):
    all_types = sorted(type2nprops.keys(), reverse=True)

    results = load_from_file(results_fpath)
    y_true, y_pred = [], []
    syntax_errors = []
    incorrect_nprops = []
    incorrect_orders = []
    unknow_types = []

    formula2type = {}
    formula2prop = {}
    for pattern_type, all_nprops in type2nprops.items():
        for nprops in all_nprops:
            formulas, props_perm = sample_formulas(pattern_type, nprops, False)
            for formula, prop_perm in zip(formulas, props_perm):
                formula2type[formula] = pattern_type
                formula2prop[formula] = list(prop_perm)

    # print(len(formula2type))
    # print(len(formula2type))
    # pprint(len(set(formula2type.values())))

    for idx, result in enumerate(results):
        # print(f"result {idx}")
        result = result[1:]
        pattern_type, nprops, prop_perm_str, utt, true_ltl, output_ltl, is_correct = result
        nprops = int(nprops)
        true_prop_perm = deserialize_props_str(prop_perm_str)

        if is_correct == "Syntax Error":
            syntax_errors.append(result)
        else:
            if output_ltl in formula2type:
                pred_prop_perm = formula2prop[output_ltl]

                if len(true_prop_perm) != len(pred_prop_perm):
                    print(f"incorrect_nprops:\n{result}\n")
                    incorrect_nprops.append(result)
                    # breakpoint()
                elif true_prop_perm != pred_prop_perm and not is_correct:  # diff prop order and not spot equivalent
                    print(f"incorrect_orders:\n{result}\n")
                    incorrect_orders.append(result)
                    # breakpoint()
                else:
                    if not is_correct:
                        print(f"{pattern_type} {formula2type[output_ltl]}\n{result}\n")
                        # breakpoint()
                    y_true.append(pattern_type)
                    y_pred.append(formula2type[output_ltl])
            else:
                unknow_types.append(result)
                print(f"unknown_type\n{result}")
                # breakpoint()

    cm = confusion_matrix(y_true, y_pred, labels=all_types)

    print(f"number of syntax errors: {len(syntax_errors)}/{len(results)}")
    print(f"number of incorrect nprops: {len(incorrect_nprops)}/{len(results)}")
    print(f"number of incorrect orders: {len(incorrect_orders)}/{len(results)}")
    print(f"number of unknow types: {len(unknow_types)}/{len(results)}")

    return cm, all_types


def formula_to_type(all_types, nprops):
    """
    Construct mapping from all possible formulas with `nprops` propositions to their type, e.g. ordered_visit_3
    :param nprops:
    :return:
    """
    formula2type = {}
    for pattern_type in all_types:
        formula = sample_formulas(pattern_type, nprops, False)[0]
        formula2type[formula] = f"{pattern_type}_{nprops}"
    return formula2type


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_fpath", type=str, default="results/finetuned_gpt3/utt_holdout_batch12_perm/log_gpt3_finetuned_symbolic_batch12_perm_utt_0.2_42.csv", help="fpath of results to analyze.")
    parser.add_argument("--split_fpath", type=str, default="data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_42.pkl", help="fpath to split dataset used to produce results.")
    parser.add_argument("--what_test", action="store_true", help="True to find holdout formulas.")
    args = parser.parse_args()

    plot_cm(args.results_fpath, TYPE2NPROPS)
