import logging
from collections import defaultdict
import numpy as np
import spot

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
    acc_anaysis = [["LTL Template Type", "Number of Propositions", "Number of Utterances", "Accuracy"]]
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
    train_iter, train_meta, valid_iter, valid_meta = load_split_dataset(split_dataset_fpath)
    return evaluate_lang_single(model, valid_iter, valid_meta,
                                analysis_fpath, result_log_fpath, acc_fpath, len(valid_iter))


def evaluate_plan(out_traj, true_traj):
    return out_traj == true_traj