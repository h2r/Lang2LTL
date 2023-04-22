import os
from pathlib import Path
import argparse
import logging
from collections import defaultdict
from itertools import product
from sklearn.model_selection import train_test_split, KFold
import spot

from utils import deserialize_props_str, load_from_file, save_to_file

UNARY_OPERATORS = ["not", "next", "finally", "always"]
BINARY_OPERATORS = ["and", "or", "implies", "until"]
FEASIBLE_OPERATORS = ["and", "or"]  # operators currently supported


def compose(data_fpath, nclauses, feasible_operators, ignore_repeat, size_formula, seed_formula, size_utt, seeds_utt):
    """
    Construct composed dataset.
    In one pass of base dataset, construct composed dataset for zero-shot transfer, formula and utterance holdout.
    """
    save_fpath_zeoshot = os.path.join(os.path.dirname(data_fpath), f"composed_zeroshot_{Path(args.data_fpath).stem}.pkl")
    save_fpath_formula = os.path.join(os.path.dirname(data_fpath), f"composed_formula_{Path(args.data_fpath).stem}.pkl")
    save_fpath_utt = os.path.join(os.path.dirname(data_fpath), f"composed_utt_{Path(args.data_fpath).stem}.pkl")
    log_fpath = os.path.join(os.path.dirname(data_fpath), f"composed_{Path(args.data_fpath).stem}.log")

    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(log_fpath, mode='w'),
                            logging.StreamHandler()
                        ]
    )
    logger = logging.getLogger()

    # Load base dataset
    ltl2utts, all_base_ltls, all_base_ltls_spot, ltl2meta = load_base_dataset(data_fpath, logger)

    # Construct composed dataset
    operator_seqs = list(product(feasible_operators, repeat=nclauses-1))  # all combs of operators to connect base formulas
    base_ltl_seqs = list(product(all_base_ltls, repeat=nclauses))  # all combs of base formulas

    composed_zeroshot = defaultdict(list)  # by itself serve as test set for zero-shot transfer
    formula2dataset = defaultdict(list)  # formula to (data, meta) pair
    composed_utt = [[] for _ in range(len(seeds_utt))]  # utt split dataset for each fold
    nattempts, npairs = 0, 0
    err2count = defaultdict(int)  # composed formula: syntax error, infeasible, repeated, correct
    for operator_seq in operator_seqs:
        for base_ltl_seq in base_ltl_seqs:
            base_ltl_seq = list(base_ltl_seq)
            logger.info(f"Composing {nattempts}:\n{operator_seq}\n{base_ltl_seq}\n{[ltl2meta[ltl] for ltl in base_ltl_seq]}")
            data, meta = compose_single(operator_seq, base_ltl_seq, ltl2utts, all_base_ltls_spot, ltl2meta, ignore_repeat, err2count, logger)

            composed_zeroshot["data"].extend(data)
            composed_zeroshot["meta"].extend(meta)

            if data and meta:
                formula2dataset[meta[0][0]].append((data, meta))  # key: composed ltl formula 'and-global_avoidance_3-visit_2'

                for seed_idx, seed_utt in enumerate(seeds_utt):
                    train_dataset, valid_dataset = train_test_split(list(zip(data, meta)), test_size=size_utt, random_state=seed_utt)
                    train_data, train_meta = zip(*train_dataset)
                    valid_data, valid_meta = zip(*valid_dataset)
                    if composed_utt[seed_idx]:
                        for item_idx, item in enumerate([train_data, train_meta, valid_data, valid_meta]):
                            composed_utt[seed_idx][item_idx] += item
                    else:
                        composed_utt[seed_idx] = [train_data, train_meta, valid_data, valid_meta, {"size": size_utt, "seed": seed_utt}]

            nattempts += 1
            npairs += len(data)

    logger.info(f"Composed types with syntax error: {err2count['syntax_err']}/{nattempts} = {err2count['syntax_err']/nattempts}")
    logger.info(f"Composed types that are infeasible: {err2count['infeasible']}/{nattempts} = {err2count['infeasible']/nattempts}")
    logger.info(f"Composed types being redundant: {err2count['repeat']}/{nattempts} = {err2count['repeat']/nattempts}")
    ncorrects = nattempts - err2count['syntax_err'] - err2count['infeasible'] - err2count['repeat']
    logger.info(f"Correct composed types: {ncorrects}/{nattempts} = {ncorrects /nattempts}")
    logger.info(f"Total number of composed pairs: {npairs}")

    # Save zero_shot transfer dataset
    save_to_file(composed_zeroshot, save_fpath_zeoshot)

    # Save utterance holdout dataset
    save_to_file(composed_utt, save_fpath_utt)

    # Construct and save formula holdout dataset
    composed_formula = []
    all_formulas = list(formula2dataset.keys())
    kf = KFold(n_splits=len(all_formulas) // size_formula, random_state=seed_formula, shuffle=True)
    for fold_idx, (train_indices, holdout_indices) in enumerate(kf.split(all_formulas)):
        holdout_formulas = [all_formulas[idx] for idx in holdout_indices]
        train_iter, train_meta, valid_iter, valid_meta = [], [], [], []  # meta data is (pattern_type, nprops) pairs
        for formula in all_formulas:
            for data, meta in formula2dataset[formula]:  # same type and nprops, diff perm of props considered same formula
                if formula in holdout_formulas:
                    valid_iter.extend(data)
                    valid_meta.extend(meta)
                else:
                    train_iter.extend(data)
                    train_meta.extend(meta)
        composed_formula.append((train_iter, train_meta, valid_iter, valid_meta, {"size": size_formula, "seed": seed_formula, "fold_idx": fold_idx}))
    save_to_file(composed_formula, save_fpath_formula)


def compose_single(operators, base_ltls, ltl2utts, all_ltls_spot, ltl2meta, ignore_repeat, err2count, logger):
    """
    Construct a single composed type permuting all composed utterances.
    e.g., ["and"],  ["sequenced_visit", "global_avoidance"], [2, 1]
    Same composed formula for all composed utterances.
    Short circuit: return empty lists if compose formula for 1 composed utterance incorrect or redundant.
    """
    # Select base formulas and based utterances based on input arguments
    all_base_triples = [[(utt, base_ltl, ltl2meta[base_ltl]) for utt in ltl2utts[base_ltl]] for base_ltl in base_ltls]

    # Compose one operator at a time
    for operator in operators:
        if operator in UNARY_OPERATORS:
            base_triples = all_base_triples.pop(0)  # replace clause 1 by composed formula
        elif operator in BINARY_OPERATORS:
            if len(all_base_triples) < 2:
                raise ValueError(f"ERROR: {len(all_base_triples)} bases not enough for binary operator {operator}.")
            base_triples = [all_base_triples.pop(0), all_base_triples.pop(0)]  # replace clause 1 and 2 by composed formula
        else:
            raise ValueError(f"ERROR: unrecognized operator: {operator}.")

        if len(base_triples) > 1:  # binary operator
            base_triples_comb = list(product(*base_triples))  # all combs of operand 1 and operand 2
        else:  # unary operator
            base_triples_comb = base_triples
        logger.info(f"Number of triples to be composed: {len(base_triples_comb)}\n")  # TODO: count only valid <= 2 clauses

        pairs_composed = []
        for pair_idx, composing_bases in enumerate(base_triples_comb):
            utts_base, ltls_base, metas = zip(*composing_bases)  # unpack list of tuples into 3 lists
            if operator == "and":
                utt_composed, ltl_composed = compose_and(utts_base, ltls_base)
            elif operator == "or":
                utt_composed, ltl_composed = compose_or(utts_base, ltls_base)
            else:
                raise ValueError(f"ERROR: operator not yet supported: {operator}.")
            # logger.info(f"Composed pair {pair_idx}:\n{utts_base}\n{formulas_base}\n{utt_composed}\n{formula_composed}\n")

            # Check composed formula for incorrect syntax, feasibility, redundancy before adding to composed dataset
            try:
                ltl_spot = spot.formula(ltl_composed)
            except SyntaxError:
                err2count["syntax_err"] += 1
                # logger.info(f"Syntax error in composed formula:\n{ltl_composed}\n{utt_composed}")
                return [], []
            if ignore_repeat and ltl_spot in all_ltls_spot:
                err2count["repeat"] += 1
                # logger.info(f"Repeat composed formula already exists in base dataset:\n{ltl_spot} = {ltl_composed}\n{utt_composed}\n{ltls_base}\n{utts_base}\n")
                return [], []
            elif spot.are_equivalent(ltl_spot, spot.formula("False")):
                err2count["infeasible"] += 1
                # logger.info(f"Infeasible composed formula:\n{ltl_composed}\n{utt_composed}")
                return [], []
            else:
                # logger.info(f"Correct composed formula:\n{ltl_composed}\n{metas}")
                pairs_composed.append((utt_composed, ltl_composed, metas))

        all_base_triples.insert(0, pairs_composed)  # continue composing composed formula with remaining base formulas

    dataset_composed = all_base_triples[0]
    # type_composed = '-'.join(list(operators) + base_ltls)
    data, meta = [], []
    for utt, ltl, metas in dataset_composed:
        data.append((utt, ltl))
        info_composed = '-'.join(list(operators) + [f"{pattern_type}_{len(props)}" for pattern_type, props in metas])
        info_bases = '-'.join(list(operators) + [f"{pattern_type}_{'_'.join(props)}" for pattern_type, props in metas])
        meta.append((info_composed, info_bases))  # two differ only for perm dataset ('and-global_avoidance_3-visit_2', 'and-patrolling_d_a_b-visit_a_b')
    return data, meta


def compose_and(utts, formulas):
    utt_composed = f"{utts[0]}, in addition {utts[1]}"
    formula_composed = f"& {formulas[0]} {formulas[1]}"
    return utt_composed, formula_composed


def compose_or(utts, formulas):
    utt_composed = f"Either {utts[0]}, or {utts[1]}"
    formula_composed = f"| {formulas[0]} {formulas[1]}"
    return utt_composed, formula_composed


def load_base_dataset(data_fpath, logger):
    dataset = load_from_file(data_fpath)
    ltl2utts = defaultdict(list)
    all_base_ltls = set()
    all_base_ltls_spot = set()
    ltl2metas = defaultdict(list)
    for pattern_type, props_str, utt, ltl in dataset:
        props = deserialize_props_str(props_str)

        ltl2utts[ltl].append(utt)
        all_base_ltls.add(ltl)
        all_base_ltls_spot.add(spot.formula(ltl))

        ltl2metas[ltl].append((pattern_type, tuple(props)))

    for ltl, metas in ltl2metas.items():
        metas = set(metas)
        if len(metas) > 1:
            logger.info(f"Duplicate base formulas:\n{metas}\n{ltl}\n")

    ltl2meta = {ltl: meta[0] for ltl, meta in ltl2metas.items()}
    return ltl2utts, all_base_ltls, all_base_ltls_spot, ltl2meta


def get_valid_composed_formulas(data_fpath, nclauses, feasible_operators):
    """
    A valid composed formula does not have syntax, infeasibility, or repeat existing base formula.
    """
    log_fpath = os.path.join(os.path.dirname(data_fpath), f"composed_valid_formulas_stats_{Path(data_fpath).stem}.log")
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(log_fpath, mode='w'),
                            logging.StreamHandler()
                        ]
    )
    logger = logging.getLogger()

    # Load base dataset
    _, all_base_ltls, all_base_ltls_spot, _ = load_base_dataset(data_fpath, logger)

    # Construct composed formula; count errors
    operator_seqs = list(product(feasible_operators, repeat=nclauses-1))  # all combs of operators to connect base formulas
    base_ltl_seqs = list(product(all_base_ltls, repeat=nclauses))  # all combs of base formulas
    err2count = defaultdict(int)  # composed formula: correct, syntax error, infeasible, repeat
    nattempts = 0
    composed_ltls = set()
    for operator_seq in operator_seqs:
        for base_ltl_seq in base_ltl_seqs:
            base_ltl_seq = list(base_ltl_seq)
            nattempts += 1
            # Compose one operator at a time
            for operator in operator_seq:
                if operator in UNARY_OPERATORS:
                    base_ltls = [base_ltl_seq.pop(0)]  # replace clause 1 by composed formula
                elif operator in BINARY_OPERATORS:
                    if len(base_ltl_seq) < 2:
                        raise ValueError(f"ERROR: {len(base_ltl_seq)} not enough base ltls for binary operator {operator}.")
                    base_ltls = [base_ltl_seq.pop(0), base_ltl_seq.pop(0)]  # replace clause 1 and 2 by composed formula
                else:
                    raise ValueError(f"ERROR: unrecognized operator: {operator}.")

                if operator == "and":
                    op = "&"
                elif operator == "or":
                    op = "|"
                else:
                    raise ValueError(f"ERROR: operator not yet supported: {operator}.")
                ltl_composed = f"{op} {base_ltls[0]} {base_ltls[1]}"

                # Check composed formula for incorrect syntax, feasibility, redundancy before adding to composed dataset
                try:
                    ltl_spot = spot.formula(ltl_composed)
                except SyntaxError:
                    err2count["syntax_err"] += 1
                    logger.info(f"Syntax error in composed formula:\n{ltl_composed}")
                    break
                if ltl_spot in all_base_ltls_spot:
                    err2count["repeat"] += 1
                    logger.info(f"Repeated composed formula already exists in base dataset:\n{ltl_spot} = {ltl_composed}\n{base_ltls}\n")
                    break
                elif spot.are_equivalent(ltl_spot, spot.formula("False")):
                    err2count["infeasible"] += 1
                    logger.info(f"Infeasible composed formula:\n{ltl_composed}")
                    break
                else:
                    err2count["correct"] += 1
                    logger.info(f"Correct composed formula:\n{ltl_composed}")

                base_ltl_seq.insert(0, ltl_composed)  # continue composing composed formula with remaining base formulas

            if base_ltl_seq:
                composed_ltls.add(base_ltl_seq[0])

    logger.info(f"composed_ltls: {composed_ltls}")
    logger.info(f"total attempts: {nattempts}")
    logger.info(f"total number of composed formulas: {len(composed_ltls)}")
    logger.info(f"error2count: {err2count} {sum(err2count.values())}")


if __name__ == "__main__":
    # python compose.py --ignore_repeat
    # python compose.py --get_formula_stats
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", type=str, default="data/symbolic_batch12_noperm.csv", help="base dataset to compose.")
    parser.add_argument("--nclauses", type=int, default=2, help="number of clauses in composed formula.")
    parser.add_argument("--get_formula_stats", action="store_true", help="True to only compute stats on composed formulas.")
    parser.add_argument("--ignore_repeat", action="store_true", help="True to ignore composed formulas already exist.")
    parser.add_argument("--size_formula", type=int, default=900, help="fold size for formula holdout.")
    parser.add_argument("--seed_formula", type=int, default=42, help="seed for formula holdout.")
    parser.add_argument("--size_utt", type=float, default=0.2, help="fold size for utterance holdout.")
    parser.add_argument("--seeds_utt", type=int, default=[0, 1, 2, 42, 111], help="seed for utterance holdout.")
    args = parser.parse_args()

    if args.get_formula_stats:
        get_valid_composed_formulas(args.data_fpath, args.nclauses, FEASIBLE_OPERATORS)
    else:
        compose(args.data_fpath, args.nclauses, FEASIBLE_OPERATORS, args.ignore_repeat,
                args.size_formula, args.seed_formula, args.size_utt, args.seeds_utt)
