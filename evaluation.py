import numpy as np
import spot

from s2s_pt_transformer import construct_dataset
from utils import save_to_file


def evaluate(s2s, data_fpath, results_fpath):
    train_iter, valid_iter, _, _, _, _ = construct_dataset(data_fpath)
    results = [["train_or_valid", "utterances", "true_ltl", "output_ltl", "is_correct"]]
    accs = []

    for idx, (utt, true_ltl) in enumerate(valid_iter+train_iter):
        train_or_valid = "valid" if idx < len(valid_iter) else "train"
        out_ltl = s2s.translate([utt])[0]
        try:  # output LTL formula may have syntax error
            is_correct = spot.are_equivalent(spot.formula(out_ltl), spot.formula(true_ltl))
            is_correct = 'True' if is_correct else 'False'
        except SyntaxError:
            print(f'Syntax error in output LTL: {out_ltl}')
            is_correct = 'Syntax Error'
        results.append([train_or_valid, utt, true_ltl, out_ltl, is_correct])
        if train_or_valid == "valid":
            accs.append(is_correct)

    save_to_file(results, f"{results_fpath}.csv")
    print(np.mean([True if acc == 'True' else False for acc in accs]))
