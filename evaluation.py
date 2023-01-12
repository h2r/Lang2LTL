import os
from pathlib import Path
import argparse
import logging
from collections import defaultdict
import numpy as np
import spot

from gpt3 import GPT3
from dataset import load_split_dataset
from utils import load_from_file, save_to_file


def evaluate_lang(output_ltls, true_ltls):
    """
    Parse LTL formulas in infix or prefix (spot.formula) then check semantic equivalence (spot.are_equivalent).
    """
    accs = []
    for out_ltl, true_ltl in zip(output_ltls, true_ltls):
        try:  # output LTL formula may have syntax error
            is_correct = spot.are_equivalent(spot.formula(out_ltl), spot.formula(true_ltl))
            is_correct = "True" if is_correct else "False"
        except SyntaxError:
            logging.info(f"Syntax error: {true_ltl}\n{out_ltl}\n")
            is_correct = "Syntax Error"
        accs.append(is_correct)
    acc = np.mean([True if acc == "True" else False for acc in accs])
    return accs, acc


def evaluate_lang_single(model, valid_iter, valid_meta, analysis_fpath, result_log_fpath, acc_fpath, valid_iter_len):
    """
    Evaluate translation accuracy per LTL pattern type.
    """
    result_log = [["train_or_valid", "pattern_type", "nprops", "utterances", "true_ltl", "output_ltl", "is_correct"]]

    meta2accs = defaultdict(list)
    for idx, ((utt, true_ltl), (pattern_type, nprops)) in enumerate(zip(valid_iter, valid_meta)):
        train_or_valid = "valid" if idx < valid_iter_len else "train"  # TODO: remove after having enough data
        out_ltl = model.translate([utt])[0].strip()
        try:  # output LTL formula may have syntax error
            is_correct = spot.are_equivalent(spot.formula(out_ltl), spot.formula(true_ltl))
            is_correct = "True" if is_correct else "False"
        except SyntaxError:
            is_correct = "Syntax Error"
        logging.info(f"{idx}\n{pattern_type} | {nprops}\n{utt}\n{true_ltl}\n{out_ltl}\n{is_correct}\n")
        result_log.append([train_or_valid, pattern_type, nprops, utt, true_ltl, out_ltl, is_correct])
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
    result_aux = load_from_file(result_fpaths[0], noheader=False)
    fields = result_aux.pop(0)
    aggregated_result = [fields]
    for row in result_aux:
        aggregated_result.append(row[:3]+[0.0])

    total_corrects, total_samples = 0, 0
    for n, result_fpath in enumerate(result_fpaths):
        result = load_from_file(result_fpath, noheader=True)

        for row_idx, row in enumerate(result):
            if row[0] not in filter_types and row[3] != "no valid data":
                acc, nsamples = float(row[3]), int(row[2])
                aggregated_result[row_idx+1][3] += 1 / (n + 1) * (acc - aggregated_result[row_idx+1][3])  # running average
                total_corrects += nsamples * acc
                total_samples += nsamples

    aggregated_result_fpath = f"{os.path.commonprefix(result_fpaths)}.csv"
    save_to_file(aggregated_result, aggregated_result_fpath)
    print(f"total accuracy: {total_corrects / total_samples}")


def evaluate_plan(out_traj, true_traj):
    return out_traj == true_traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dataset_fpath", type=str, default="data/holdout_splits/split_symbolic_no_perm_batch1_utt_0.2_111.pkl", help="path to pkl file storing train, test set")
    parser.add_argument("--model", type=str, default="gpt3_finetuned_split_symbolic_no_perm_batch1_utt_0.2_111", help="name of model to be evaluated")
    parser.add_argument("--nexamples", type=int, default=1, help="number of examples per instance for GPT-3")
    args = parser.parse_args()
    dataset_name = Path(args.split_dataset_fpath).stem

    if "gpt3" in args.model or "davinci" in args.model:
        dataset = load_from_file(args.split_dataset_fpath)
        valid_iter = dataset["valid_iter"]
        if "finetuned" in args.model:
            engine = load_from_file("model/gpt3_models.pkl")[args.model]
            valid_iter = [(f"Utterance: {utt}\nLTL:", ltl) for utt, ltl in valid_iter]
            result_log_fpath = f"results/log_{args.model}.csv"  # finetuned model name also include dataset name
            acc_fpath = f"results/acc_{args.model}.csv"
        else:
            engine = args.model
            prompt_fpath = os.path.join("data", "symbolic_prompts_new", f"prompt_{args.nexamples}_{dataset_name}.txt")
            prompt = load_from_file(prompt_fpath)
            valid_iter = [(f"{prompt}Utterance: {utt}\nLTL:", ltl) for utt, ltl in valid_iter]
            result_log_fpath = f"results/log_{args.model}_{dataset_name}.csv"
            acc_fpath = f"results/acc_{args.model}_{dataset_name}.csv"
        dataset["valid_iter"] = valid_iter
        split_dataset_fpath = os.path.join("data", "gpt3", f"{dataset_name}.pkl")
        save_to_file(dataset, split_dataset_fpath)
        model = GPT3(engine, temp=0, max_tokens=64)
    else:
        raise ValueError(f"ERROR: model not recognized: {args.model}")

    analysis_fpath = "data/analysis_batch1.csv"
    evaluate_lang_from_file(model, split_dataset_fpath, analysis_fpath, result_log_fpath, acc_fpath)

    # To aggregate accuracy-per-formula result files, comment out all above in main, uncomment all below
    # result_fpaths = [
    #     "results/acc_gpt3_finetuned_split_symbolic_no_perm_batch1_utt_0.2_0.csv",
    #     "results/acc_gpt3_finetuned_split_symbolic_no_perm_batch1_utt_0.2_1.csv",
    #     "results/acc_gpt3_finetuned_split_symbolic_no_perm_batch1_utt_0.2_2.csv",
    #     "results/acc_gpt3_finetuned_split_symbolic_no_perm_batch1_utt_0.2_42.csv",
    #     "results/acc_gpt3_finetuned_split_symbolic_no_perm_batch1_utt_0.2_111.csv",
    # ]
    # filter_types = ["fair_visit"]
    # aggregate_results(result_fpaths, filter_types)
