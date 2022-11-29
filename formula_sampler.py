"""
Sample formulas from a set of specification patterns with permutation of varying numbers of propositions.
Please refer to Specification Patterns for Robotic Missions for the definition of each pattern type.
https://arxiv.org/pdf/1901.02077.pdf
"""
import argparse
from itertools import permutations
from pprint import pprint
import spot

from utils import prefix_to_infix


def sample_formulas(pattern_type, nprops):
    """
    :param pattern_type: type of LTL specification pattern
    :param nprops: number of proposition in LTL formulas
    :return: sampled formulas with `nprops` propositions of `pattern_type` and permutation of propositions
    """
    props_all = [chr(ord("a")+i) for i in range(nprops)]
    props_perm = list(permutations(props_all))

    if pattern_type == "visit":
        pattern_sampler = visit
    elif pattern_type == "sequenced_visit":
        pattern_sampler = sequenced_visit
    elif pattern_type == "ordered_visit":
        pattern_sampler = ordered_visit
    # elif pattern_type == "strict_ordered_visit":
    #     pattern_sampler = strict_ordered_visit_fixed
    # elif pattern_type == "fair_visit":
    #     pattern_sampler = fair_visit
    elif pattern_type == "patrolling":
        pattern_sampler = patrolling
    elif pattern_type == "sequenced_patrolling":
        pattern_sampler = sequenced_patrolling
    elif pattern_type == "ordered_patrolling":
        pattern_sampler = ordered_patrolling
    else:
        raise TypeError(f"ERROR: unrecognized pattern type {pattern_type}")

    formulas = [pattern_sampler(list(props)) for props in props_perm]
    if args.debug:
        formulas = [spot.formula(pattern_sampler(list(props))) for props in props_perm]

    return formulas, props_perm


def visit(props):
    if len(props) == 1:
        return f"F {props[0]}"
    return f"& F {props.pop(0)} " + visit(props)


def sequenced_visit(props):
    if len(props) == 1:
        return f"F {props[0]}"
    return f"F & {props.pop(0)} " + sequenced_visit(props)


def ordered_visit(props):
    if len(props) == 1:
        return f"F {props[0]}"
    return f"& U ! {props[1]} {props.pop(0)} " + ordered_visit(props)


def strict_ordered_visit_fixed(props):
    """
    Different to what presented in the paper, after visiting a predecessor,
    trace does not need to exit it at the immediate next time step.
    """
    return


def fair_visit(props):
    return


def patrolling(props):
    if len(props) == 1:
        return f"G F {props[0]}"
    return f"& G F {props.pop(0)} " + patrolling(props)


def sequenced_patrolling(props):
    """
    Sequenced patrolling formulas are the same as patrolling formulas.
    e.g. G(F(a & F(b & Fc))) == GFc & GFa & GFb
    """
    return f"G " + sequenced_visit(props)


def ordered_patrolling(props):
    """
    1st part of ordered patrolling formula is the same as sequenced visit formula.
    """
    formula = sequenced_patrolling(props[:])
    if len(props) > 1:
        # props.append(props[0])  # incorrect definition in paper: circle proposition list back to first proposition
        formula = f"& {formula} {ordered_patrolling_constraints(props)}"
    return prefix_to_infix(formula)


def ordered_patrolling_constraints(props):
    """
    2nd and 3rd parts of ordered patrolling formula.
    """
    assert len(props) >= 2, f"length of props for ordered_patrolling_constraints must be at least 2, got {len(props)}"
    if len(props) == 2:
        a, b = props[0], props[1]
        return f"& U ! {b} {a} G -> {b} X U ! {b} {a}"
    b, a = props[1], props.pop(0)
    return f"& U ! {b} {a} & G -> {b} X U ! {b} {a} " + ordered_patrolling_constraints(props)


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument("--debug", action="store_true", help="include to turn on debug mode.")
    args = paser.parse_args()

    # props = ['a', 'b', 'c']
    # formula = ordered_patrolling_constraints(props)
    # print(formula)
    # formula = spot.formula(prefix_to_infix(formula))
    # print(formula)

    formulas, props_perm = sample_formulas("ordered_patrolling", 2)
    pprint(list(zip(formulas, props_perm)))
