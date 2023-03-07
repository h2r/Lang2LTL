import os
from pathlib import Path
import argparse
import logging
from collections import defaultdict
from itertools import product
import spot

from formula_sampler import PROPS, FEASIBLE_TYPES
from utils import deserialize_props_str, load_from_file, save_to_file

UNARY_OPERATORS = ["not", "next", "finally", "always"]
BINARY_OPERATORS = ["and", "or", "implies", "until"]
FEASIBLE_OPERATORS = ["and", "or"]


def compose(data_fpath, all_operators, all_base_types, all_base_nprops, ignore_redundant):
    """
    Construct composed dataset.
    """
    # Load dataset
    dataset = load_from_file(data_fpath)
    meta2data = defaultdict(list)
    all_formulas_spot = set()
    for pattern_type, props_str, utt, formula in dataset:
        props = deserialize_props_str(props_str)
        meta2data[(pattern_type, len(props))].append((utt, formula))
        all_formulas_spot.add(spot.formula(formula))

    save_fpath = os.path.join(os.path.dirname(data_fpath), f"composed_{Path(args.data_fpath).stem}.csv")
    log_fpath = os.path.join(os.path.dirname(data_fpath), f"composed_{Path(args.data_fpath).stem}_stats.log")
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(log_fpath,mode='w'),
                            logging.StreamHandler()
                        ]
    )
    logger = logging.getLogger()

    compositions = [["composed_utterance", "composed_formulas", "base_utterances", "base_formulas"]]
    nattempts, ncomposed, npairs = 0, 0, 0
    for operators in all_operators:
        for base_types in all_base_types:
            for base_nprops in all_base_nprops:
                logger.info(f"Composing type {nattempts}:\n{operators}\n{base_types}\n{base_nprops}")
                compositions_single = compose_single(meta2data, all_formulas_spot, operators, base_types, base_nprops, ignore_redundant, logger)
                compositions.extend(compositions_single)
                nattempts += 1
                npairs += len(compositions_single)
                if compositions_single:
                    ncomposed += 1
    logger.info(f"Total number of composed types: {ncomposed}")
    logger.info(f"Total number of composed pairs: {npairs}")
    # save_to_file(compositions, save_fpath, mode='w')


def compose_single(meta2data, all_formulas_spot, operators, base_types, base_nprops, ignore_redundant, logger):
    """
    Construct a single composition permuting all base formulas and base utterances.
    e.g., ["and"],  ["sequenced_visit", "global_avoidance"], [2, 1]
    """
    # Check arguments
    assert len(base_types) == len(base_nprops), f"{base_types} base formula types != {base_nprops} base prop lists."

    for base_type in base_types:
        if base_type not in FEASIBLE_TYPES:
            raise ValueError(f"ERROR: {base_type} not a feasible LTL type.")

    # Select base formulas and based utterances based input arguments
    all_base_pairs = []
    for pattern_type, nprops in zip(base_types, base_nprops):
        utt_ltl_pairs = meta2data[(pattern_type, nprops)]
        all_base_pairs.append(utt_ltl_pairs)

    # Compose one operator at a time
    compositions = []
    for operator in operators:
        if operator in UNARY_OPERATORS:
            base_pairs = all_base_pairs.pop(0)  # replace clause 1 by composed formula
        elif operator in BINARY_OPERATORS:
            if len(all_base_pairs) < 2:
                raise ValueError(f"ERROR: {len(all_base_pairs)} base pairs not enough for binary operator {operator}.")
            base_pairs = [all_base_pairs.pop(0), all_base_pairs.pop(0)]  # replace clause 1 and 2 by composed formula
        else:
            raise ValueError(f"ERROR: unrecognized operator: {operator}.")

        if len(base_pairs) > 1:  # binary operator
            base_pairs_perm = list(product(*base_pairs))  # permute all possible operand 1 and all possible operand 2
        else:  # unary operator
            base_pairs_perm = base_pairs
        logger.info(f"Number of pairs to be composed: {len(base_pairs_perm)}\n")  # TODO: count only valid for <= 2 clauses

        if len(base_pairs_perm) == 0:  # nprops invalid for 1 type, e.g. ["and"], ["visit", "sequenced_visit"], [1, 1]
            return []
        else:
            return [0] * len(base_pairs_perm)
        #
        # pairs_composed = []
        # for pair_idx, composing_bases in enumerate(base_pairs_perm):
        #     utts_base, formulas_base = zip(*composing_bases)  # unpack list of tuples into 2 lists
        #     if operator == "and":
        #         utt_composed, formula_composed = compose_and(utts_base, formulas_base)
        #     elif operator == "or":
        #         utt_composed, formula_composed = compose_or(utts_base, formulas_base)
        #     else:
        #         raise ValueError(f"ERROR: operator not yet supported: {operator}.")
        #     logger.info(f"Composed pair {pair_idx}:\n{utts_base}\n{formulas_base}\n{utt_composed}\n{formula_composed}\n")
        #
        #     # Check composed formula for syntactical correctness and redundancy before adding to composed dataset
        #     try:
        #         formula_spot = spot.formula(formula_composed)
        #     except SyntaxError:
        #         raise SyntaxError(f"Syntax error in composed formula:\n{formula_composed}\n{utt_composed}")
        #
        #     if ignore_redundant and formula_spot in all_formulas_spot:
        #         logger.info(f"Composed formula already exists:\n{formula_spot} = {formula_composed}\n{utt_composed}\n{formulas_base}\n{utts_base}\n")
        #         return []
        #     else:
        #         pairs_composed.append((utt_composed, formula_composed))
        #         compositions.append([utt_composed, formula_composed, utts_base, formulas_base])
        #
        #     all_base_pairs.insert(0, pairs_composed)

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
    # python compose.py --ignore_redundant
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", type=str, default="data/symbolic_batch12_perm.csv", help="dataset used for composition.")
    parser.add_argument("--ignore_redundant", action="store_true", help="True to ignore composed formulas already exist.")
    args = parser.parse_args()

    nclauses = 2
    all_operators = list(product(FEASIBLE_OPERATORS, repeat=nclauses-1))  # operators used to compose base formulas
    all_base_types = list(product(FEASIBLE_TYPES, repeat=nclauses))  # LTL type for each base formula
    all_base_nprops = list(product(range(1, len(PROPS)+1), repeat=nclauses))  # nprops for each base formula

    # all_operators = [["and"]]
    # all_base_types = [["sequenced_visit", "global_avoidance"]]
    # all_base_nprops = [[2, 1]]  # perm: 64800; non-perm: 648

    compose(args.data_fpath, all_operators, all_base_types, all_base_nprops, args.ignore_redundant)
