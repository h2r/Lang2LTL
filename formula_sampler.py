"""
Sample formulas from a set of specification patterns with permutation of varying numbers of propositions.
https://arxiv.org/pdf/1901.02077.pdf
"""
from itertools import permutations
from pprint import pprint
import spot


def sample_formulas(pattern_type, nprops):
    props = [chr(ord("a")+i) for i in range(nprops)]
    props_perm = list(permutations(props))

    if pattern_type == "visit":
        pattern_fn = visit
    elif pattern_type == "sequenced_visit":
        pattern_fn = sequenced_visit
    else:
        raise TypeError(f"ERROR: unrecognized pattern type {pattern_type}")

    return [spot.formula(pattern_fn(list(prop))) for prop in props_perm]


def visit(props):
    if len(props) == 1:
        return f"F {props[0]}"
    return f"& F {props.pop(0)} " + visit(props)


def sequenced_visit(props):
    if len(props) == 1:
        return f"F {props[0]}"
    return f"F & {props.pop(0)} " + sequenced_visit(props)


if __name__ == '__main__':
    formulas = sample_formulas("sequenced_visit", 3)
    pprint(formulas)
