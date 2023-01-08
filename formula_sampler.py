"""
Sample formulas from a set of specification patterns with permutation of varying numbers of propositions.
Please refer to Specification Patterns for Robotic Missions for the definition of each pattern type.
https://arxiv.org/pdf/1901.02077.pdf
Pattern sampling functions with post fixed means the function is different to what presented in the paper,
after visiting a predecessor, trace does not need to exit DFA state at the immediate next time step.
"""
import argparse
from itertools import permutations
from pprint import pprint
import spot


PROPS = ["a", "b", "c", "d", "h", "j", "k", "l", "n", "o", "p", "q", "r", "s", "y", "z"]  # 16
ALL_TYPES = [
    "visit", "sequenced_visit", "ordered_visit", "strictly_ordered_visit", "fair_visit", "patrolling",
]


def sample_formulas(pattern_type, nprops, debug):
    """
    :param pattern_type: type of LTL specification pattern
    :param nprops: number of proposition in LTL formulas
    :return: sampled formulas with `nprops` propositions of `pattern_type` and permutation of propositions
    """
    props_all = PROPS[:nprops]  # props_all = [chr(ord("a")+i) for i in range(nprops)]
    props_perm = list(permutations(props_all))

    if pattern_type == "visit":
        pattern_sampler = finals
    elif pattern_type == "sequenced_visit":
        pattern_sampler = sequenced_visit
    elif pattern_type == "ordered_visit":
        pattern_sampler = ordered_visit
    elif pattern_type == "strictly_ordered_visit":
        pattern_sampler = strict_ordered_visit_fixed
    elif pattern_type == "fair_visit":
        pattern_sampler = fair_visit_fixed
    elif pattern_type == "patrolling":
        pattern_sampler = patrol
    elif pattern_type == "sequenced_patrolling":
        pattern_sampler = sequenced_patrol
    elif pattern_type == "ordered_patrolling":
        pattern_sampler = ordered_patrol_fixed
    elif pattern_type == "strictly_ordered_patrolling":
        pattern_sampler = strict_ordered_patrol_fixed
    elif pattern_type == "fair_patrolling":
        pattern_sampler = fair_patrol_fixed
    elif pattern_type == "past_avoidance":
        pattern_sampler = past_avoid
    elif pattern_type == "global_avoidance":
        pattern_sampler = global_avoid
    elif pattern_type == "future_avoidance":
        pattern_sampler = future_avoid
    else:
        raise TypeError(f"ERROR: unrecognized pattern type {pattern_type}")

    if debug:
        formulas = [spot.formula(pattern_sampler(list(props))) for props in props_perm]
    else:
        formulas = [pattern_sampler(list(props)) for props in props_perm]

    return formulas, props_perm


def sequenced_visit(props):
    if len(props) == 1:
        return f"F {props[0]}"
    return f"F & {props.pop(0)} " + sequenced_visit(props)


def ordered_visit(props):
    if len(props) == 1:
        return f"F {props[0]}"
    return f"& U ! {props[1]} {props.pop(0)} " + ordered_visit(props)


def strict_ordered_visit_fixed(props):
    formula = ordered_visit(props[:])
    if len(props) > 1:
        formula = f"& {formula} {strict_ordered_visit_constraint3(props)}"
    return formula


def strict_ordered_visit_constraint3(props):
    assert len(props) > 1, f"length of props for strict_ordered_visit_constraint3 must be > 1, got {len(props)}"
    if len(props) == 2:
        a, b = props[0], props[1]
        return f"U ! {a} U {a} U !{a} {b}"
    b, a = props[1], props.pop(0)
    return f"& U ! {a} U {a} U !{a} {b} " + strict_ordered_visit_constraint3(props)


def fair_visit_fixed(props):
    formula = finals(props[:])
    if len(props) > 1:
        props.append(props[0])  # proposition list circles back to 1st proposition for 2nd constraint
        formula = f"& {formula} {fair_visit_constraint2(props)}"
    return formula


def fair_visit_constraint2(props):
    """
    2nd term of fair visit formula.
    """
    assert len(props) > 1, f"length of props for fair_visit_constraint2 must be > 1, got {len(props)}"
    if len(props) == 2:
        a, b = props[0], props[1]
        return f"G i {a} W {a} & ! {a} W ! {a} {b}"
    b, a = props[1], props.pop(0)
    return f"& G i {a} W {a} & ! {a} W ! {a} {b} " + fair_visit_constraint2(props)


def patrol(props):
    if len(props) == 1:
        return f"G F {props[0]}"
    return f"& G F {props.pop(0)} " + patrol(props)


def sequenced_patrol(props):
    """
    Sequenced patrolling formulas are the same as patrolling formulas.
    e.g. G(F(a & F(b & Fc))) == GFc & GFa & GFb
    """
    return f"G " + sequenced_visit(props)


def ordered_patrol_fixed(props):
    formula = sequenced_patrol(props[:])
    if len(props) > 1:
        formula = f"& {formula} {utils(props[:])}"
        formula = f"& {formula} {ordered_patrol_constraint3(props)}"
    return formula


def ordered_patrol_constraint3(props):
    """
    3rd term of ordered patrolling formula.
    """
    assert len(props) > 1, f"length of props for ordered_patrol_constraint3 must be > 1, got {len(props)}"
    if len(props) == 2:
        a, b = props[0], props[1]
        return f"G i {b} W {b} & ! {b} W ! {b} {a}"
    b, a = props[1], props.pop(0)
    return f"& G i {b} W {b} & ! {b} W ! {b} {a} " + ordered_patrol_constraint3(props)


def fair_patrol_fixed(props):
    formula = f"G " + finals(props[:])
    if len(props) > 1:
        props.append(props[0])  # proposition list circles back to 1st proposition for 2nd constraint
        formula = f"& {formula} " + fair_visit_constraint2(props)
    return formula


def strict_ordered_patrol_fixed(props):
    formula = ordered_patrol_fixed(props[:])
    if len(props) > 1:
        formula = f"& {formula} {strict_ordered_patrol_constraint4(props)}"
    return formula


def strict_ordered_patrol_constraint4(props):
    """
    4th term of strict ordered patrolling formula.
    """
    assert len(props) > 1, f"length of props for strict_ordered_patrol_constraint4 must be > 1, got {len(props)}"
    if len(props) == 2:
        a, b = props[0], props[1]
        return f"G i {a} W {a} & ! {a} W ! {a} {b}"
    b, a = props[1], props.pop(0)
    return f"& G i {a} W {a} & ! {a} W ! {a} {b} " + strict_ordered_patrol_constraint4(props)


def past_avoid(props):
    assert len(props) == 2, f"length of props for past_avoid must be 2, got {len(props)}"
    return f"U ! {props[0]} {props[1]}"


def global_avoid(props):
    assert len(props) == 1, f"length of props for global_avoid must be 1, got {len(props)}"
    return f"G ! {props[0]}"


def future_avoid(props):
    assert len(props) == 2, f"length of props for future_avoid must be 2, got {len(props)}"
    return f"G i {props[0]} G ! {props[1]}"


def finals(props):
    """
    Conjunction of finals.
    """
    if len(props) == 1:
        return f"F {props[0]}"
    return f"& F {props.pop(0)} " + finals(props)


def utils(props):
    """
    Conjunction of utils.
    """
    assert len(props) > 1, f"length of props for conjunction of utils must be > 1, got {len(props)}"
    if len(props) == 2:
        a, b = props[0], props[1]
        return f"U ! {b} {a}"
    b, a = props[1], props.pop(0)
    return f"& U ! {b} {a} " + utils(props)


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument("--pattern_type", type=str, default="strictly_ordered_patrolling", help="type of specification pattern.")
    paser.add_argument("--nprops", type=int, default=3, help="number of propositions.")
    paser.add_argument("--debug", action="store_true", help="include to show LTL formulas in Spot instead of string.")
    args = paser.parse_args()

    formulas, props_perm = sample_formulas(args.pattern_type, args.nprops, args.debug)
    pprint(list(zip(formulas, props_perm)))

    # props = ['a', 'b', 'c']
    # formula = ordered_patrol_constraint3(props)
    # print(formula)
    # if args.debug:
    #     formula = spot.formula(formula)
    #     print(formula)
