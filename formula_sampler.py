"""
Sample formulas from a set of specification patterns with permutation of varying numbers of propositions.
Please refer to Specification Patterns for Robotic Missions for the definition of each pattern type.
https://arxiv.org/pdf/1901.02077.pdf
"""
from itertools import permutations
from pprint import pprint
import spot


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
    elif pattern_type == "strict_ordered_visit":
        pattern_sampler = strict_ordered_visit_fixed
    elif pattern_type == "fair_visit":
        pattern_sampler = fair_visit
    elif pattern_type == "patrolling":
        pattern_sampler = patrolling
    elif pattern_type == "sequenced_patrolling":
        pattern_sampler = sequenced_patrolling
    else:
        raise TypeError(f"ERROR: unrecognized pattern type {pattern_type}")

    return [spot.formula(pattern_sampler(list(props))) for props in props_perm], props_perm


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
    Compared to what presented in the paper, after visit predecessor,
    trace does not need to exit it at immediate next time step.
    """
    return


def fair_visit(props):
    return


def patrolling(props):
    if len(props) == 1:
        return f"G F {props[0]}"
    return f"& G F {props.pop(0)} " + patrolling(props)


def sequenced_patrolling(props):
    return f"G " + sequenced_visit(props)


if __name__ == '__main__':
    formulas, props_perm = sample_formulas("sequenced_patrolling", 3)
    pprint(list(zip(formulas, props_perm)))
