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
    "visit", "sequenced_visit", "ordered_visit", "strictly_ordered_visit", "fair_visit", "patrolling",  # batch1
]


def sample_formulas(pattern_type, nprops, debug):
    """
    :param pattern_type: type of LTL specification pattern
    :param nprops: number of proposition in LTL formulas
    :return: sampled formulas with `nprops` propositions of `pattern_type` and permutation of propositions
    """
    if "restricted_avoidance" in pattern_type:
        props_all = PROPS[:1] * nprops
    else:
        props_all = PROPS[:nprops]  # props_all = [chr(ord("a")+i) for i in range(nprops)]
    props_perm = sorted(set(permutations(props_all)))  # remove repetitions from perm, e.g. props_all = ['a', 'a']

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
    elif pattern_type == "upper_restricted_avoidance":
        pattern_sampler = upper_restricted_avoid_fixed
    elif pattern_type == "lower_restricted_avoidance":
        pattern_sampler = lower_restricted_avoid_fixed
    elif pattern_type == "exact_restricted_avoidance":
        pattern_sampler = exact_restricted_avoid_fixed
    elif pattern_type == "instantaneous_reaction":
        pattern_sampler = instantaneous_reaction
    elif pattern_type == "delayed_reaction":
        pattern_sampler = delayed_reaction
    elif pattern_type == "prompt_reaction":
        pattern_sampler = prompt_reaction
    elif pattern_type == "bound_reaction":
        pattern_sampler = bound_reaction
    elif pattern_type == "bound_delay":
        pattern_sampler = bound_delay
    elif pattern_type == "wait":
        pattern_sampler = wait
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
        return f"U ! {a} U {a} U ! {a} {b}"
    b, a = props[1], props.pop(0)
    return f"& U ! {a} U {a} U ! {a} {b} " + strict_ordered_visit_constraint3(props)


def fair_visit_fixed(props):  # TODO: to be fixed
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
        formula = f"& & {formula} {utils(props[:])} {ordered_patrol_constraint3(props)}"
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


def fair_patrol_fixed(props):  # TODO: to be fixed
    formula = f"G " + finals(props[:])
    if len(props) > 1:
        props.append(props[0])  # proposition list circles back to 1st proposition for 2nd constraint
        formula = f"& {formula} " + fair_visit_constraint2(props)
    return formula


def past_avoid(props):
    assert len(props) == 2, f"length of props for past_avoid must be 2, got {len(props)}"
    return f"U ! {props[0]} {props[1]}"


def global_avoid(props):
    return always(props, negate=True)


def future_avoid(props):
    assert len(props) == 2, f"length of props for future_avoid must be 2, got {len(props)}"
    return f"G i {props[0]} G ! {props[1]}"


def upper_restricted_avoid_fixed(props):
    props.append(props[0])
    return f"! {lower_restricted_avoid_fixed(props)}"


def lower_restricted_avoid_fixed(props):
    if len(props) == 1:
        return finals(props[0])
    a = props.pop(0)
    return f"F & {a} U {a} & ! {a} U ! {a} {lower_restricted_avoid_fixed(props)}"


def exact_restricted_avoid_fixed(props):  # TODO: to be fixed
    if len(props) == 1:
        return f"U ! {props[0]} & {props[0]} U {props[0]} G ! {props[0]}"
    a = props.pop(0)
    return f"U ! {a} & {a} U {a} {exact_restricted_avoid_fixed(props)}"


def instantaneous_reaction(props):
    assert len(props) == 2, f"length of props for instantaneous_reaction must be 2, got {len(props)}"
    return f"G i {props[0]} {props[1]}"


def delayed_reaction(props):
    assert len(props) == 2, f"length of props for delayed_reaction must be 2, got {len(props)}"
    return f"G i {props[0]} F {props[1]}"


def prompt_reaction(props):
    assert len(props) == 2, f"length of props for prompt_reaction must be 2, got {len(props)}"
    return f"G i {props[0]} X {props[1]}"


def bound_reaction(props):
    assert len(props) == 2, f"length of props for bound_reaction must be 2, got {len(props)}"
    return f"G e {props[0]} {props[1]}"


def bound_delay(props):
    assert len(props) == 2, f"length of props for bound_delay must be 2, got {len(props)}"
    return f"G e {props[0]} X {props[1]}"


def wait(props):
    assert len(props) == 2, f"length of props for wait must be 2, got {len(props)}"
    return f"U {props[0]} {props[1]}"


def finals(props):
    """
    Conjunction of finals.
    """
    if len(props) == 1:
        return f"F {props[0]}"
    return f"& F {props.pop(0)} " + finals(props)


def always(props, negate=False):
    """
    Conjunction of always.
    """
    if negate:
        operator = "G !"
    else:
        operator = "G"
    if len(props) == 1:
        return f"{operator} {props[0]}"
    return f"& {operator} {props.pop(0)} {always(props, negate)}"


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
    paser.add_argument("--nprops", type=int, default=2, help="number of propositions.")
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
