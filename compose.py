import os
from pathlib import Path
import argparse
from collections import defaultdict
from itertools import product
import spot

from formula_sampler import PROPS, FEASIBLE_TYPES
from utils import deserialize_props_str, load_from_file, save_to_file

UNARY_OPERATORS = ["not", "next", "finally", "always"]
BINARY_OPERATORS = ["and", "or", "implies", "until"]


def compose(data_fpath, operators, base_types, base_nprops, perm=False):
    # Check arguments
    assert len(base_types) == len(base_nprops), f"{base_types} base formula types != {base_nprops} base proposition lists."

    for base_type in base_types:
        if base_type not in FEASIBLE_TYPES:
            raise ValueError(f"ERROR: {base_type} not a feasible LTL type.")

    # Load dataset
    dataset = load_from_file(data_fpath)
    meta2data = defaultdict(list)
    all_formulas = set()
    for pattern_type, props_str, utt, formula in dataset:
        props = deserialize_props_str(props_str)
        if not perm:
            if props == PROPS[:len(props)]:
                meta2data[(pattern_type, len(props))].append((utt, formula))
                all_formulas.add(formula)
        else:
            meta2data[(pattern_type, len(props))].append((utt, formula))
            all_formulas.add(formula)

    # Select base formulas and base utterances
    all_base_pairs = []
    for pattern_type, nprops in zip(base_types, base_nprops):
        utt_ltl_pairs = meta2data[(pattern_type, nprops)]
        all_base_pairs.append(utt_ltl_pairs)

    # Compose
    compositions = [["composed_utterance", "composed_formulas", "base_utterances", "base_formulas"]]
    for operator in operators:
        if operator in UNARY_OPERATORS:
            base_pairs = all_base_pairs.pop(0)
        elif operator in BINARY_OPERATORS:
            if len(all_base_pairs) < 2:
                raise ValueError(f"ERROR: {len(all_base_pairs)} base pairs not enough for binary operator {operator}.")
            base_pairs = [all_base_pairs.pop(0), all_base_pairs.pop(0)]
        else:
            raise ValueError(f"ERROR: unrecognized operator: {operator}.")

        if len(base_pairs) > 1:
            base_pairs_perm = list(product(*base_pairs))
        else:
            base_pairs_perm = base_pairs

        pairs_composed = []
        for compose_bases in base_pairs_perm:
            utts_base, formulas_base = zip(*compose_bases)  # unpack list of tuples into 2 lists
            if operator == "and":
                utt_composed, formula_composed = compose_and(utts_base, formulas_base)
            elif operator == "or":
                utt_composed, formula_composed = compose_or(utts_base, formulas_base)
            else:
                raise ValueError(f"ERROR: operator not yet supported: {operator}.")

            # Check composed formula for syntactical correctness and redundancy before adding to composed dataset
            try:
                spot.formula(formula_composed)
            except SyntaxError:
                raise SyntaxError(f"Syntax error in composed formula:\n{formula_composed}\n{utt_composed}")

            if formula_composed in all_formulas:
                print(f"Composed formula already exists:\n{formula_composed}\n{utt_composed}")
            else:
                pairs_composed.append((utt_composed, formula_composed))
                compositions.append([utt_composed, formula_composed, utts_base, formulas_base])
            all_base_pairs.insert(0, pairs_composed)

    return compositions


def compose_and(utts, formulas):
    utt_composed = f"{utts[0]}, in addition {utts[1]}"
    formula_composed = f"& {formulas[0]} {formulas[1]}"
    return utt_composed, formula_composed


def compose_or(utts, formulas):
    utt_composed = f"Either {utts[0]}, or {utts[1]}"
    formula_composed = f"| {formulas[0]} {formulas[1]}"
    return utt_composed, formula_composed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", type=str, default="data/symbolic_batch12_perm.csv", help="dataset used for composition.")
    parser.add_argument("--perm", action="store_true", help="True to permute propositions in composed formulas.")
    args = parser.parse_args()

    all_operators = [["and"]]  # operators used to compose base formulas
    all_base_types = [["sequenced_visit", "global_avoidance"]]  # LTL type for each base formula
    all_base_nprops = [[2, 1]]  # number of propositions for each base formula

    for operators, base_types, base_nprops in zip(all_operators, all_base_types, all_base_nprops):
        compositions = compose(args.data_fpath, operators, base_types, base_nprops, args.perm)
        save_fpath = os.path.join(os.path.dirname(args.data_fpath), f"composed_{Path(args.data_fpath).stem}.csv")
        save_to_file(compositions, save_fpath, mode='w')  # change mode to 'a' to append to existing composed dataset
