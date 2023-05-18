import os
from utils import load_from_file, props_in_utt
from lang2ltl import rer, translate_modular, PROPS


def sym_trans_example(in_symbolic_utts, sym_trans, convert_rule, props):
    if sym_trans == "gpt3_finetuned":
        translation_engine = f"gpt3_finetuned_symbolic_batch12_perm_utt_0.2_42"
        translation_engine = load_from_file("model/gpt3_models.pkl")[translation_engine]
    elif "t5" in sym_trans:
        model_fpath = os.path.join("model", "t5-base", f"checkpoint-{load_from_file('model/models.pkl')['t5-base']}")
        translation_engine = model_fpath
    else:
        raise ValueError(f"ERROR: unrecognized symbolic translation model: {sym_trans}")

    for in_symbolic_utt in in_symbolic_utts:
        res = props_in_utt(in_symbolic_utt, props)
        symbolic_utts, symbolic_ltls, output_ltls, placeholder_maps = translate_modular([in_symbolic_utt], [res], sym_trans,
                                                                                        translation_engine, convert_rule,
                                                                                        props)
        print(f"Input symbolic utt: {in_symbolic_utt}\nSymbolic ltl: {symbolic_ltls[0]}\nOutput ltl: {output_ltls[0]}\nPlaceholder map: {placeholder_maps[0]}\n")


def multimodal_example(utt, rer_model, rer_engine, rer_prompt, sym_trans, convert_rule, props):
    """
    Extract referring expressions to use by a language and vision model to resolve propositions.
    :param utt: input utterance.
    :param rer_model: referring expression recognition module, e.g., "gpt3", "gpt4", "llama-7B".
    :param rer_engine: GPT engine for RER, e.g., "text-davinci-003", "gpt4".
    :param rer_prompt: prompt for GPT RER.
    :param sym_trans: symbolic translation module, e.g., "t5-base", "gpt3_finetuned", "gpt3_pretrained".
    :param convert_rule: conversion rule from referring expressions to propositions, e.g., "lang2ltl", "cleanup".
    :param model_fpath: pretrained model weights for symbolic translation.
    """
    res, utt2names = rer(rer_model, rer_engine, rer_prompt, [utt])

    if sym_trans == "gpt3_finetuned":
        translation_engine = f"gpt3_finetuned_symbolic_batch12_perm_utt_0.2_42"
        translation_engine = load_from_file("model/gpt3_models.pkl")[translation_engine]
    elif "t5" in sym_trans:
        model_fpath = os.path.join("model", "t5-base", f"checkpoint-{load_from_file('model/models.pkl')['t5-base']}")
        translation_engine = model_fpath
    else:
        raise ValueError(f"ERROR: unrecognized symbolic translation model: {sym_trans}")

    symbolic_utts, symbolic_ltls, output_ltls, placeholder_maps = translate_modular([utt], [res], sym_trans, translation_engine, convert_rule, props)
    print(f"Input utt: {utt}\n\nSymbolic utt: {symbolic_utts}\n\nSymbolic ltl: {symbolic_ltls}\n\nOutput ltl: {output_ltls}\n\nPlaceholder map: {placeholder_maps}")


if __name__ == "__main__":
    utt = "go to the red brick building without passing by the CS department"
    multimodal_example(utt,
                       rer_model="gpt3",
                       rer_engine="text-davinci-003",
                       rer_prompt="data/osm/rer_prompt_16.txt",
                       sym_trans="gpt3_finetuned",
                       convert_rule="lang2ltl",
                       props=PROPS)

    in_symbolic_utts = [
                        "go to a",
                        "go to b then move to c",
                        "do not go to b until you reach a then go to b"
    ]
    sym_trans_example(in_symbolic_utts=in_symbolic_utts,
                      sym_trans="gpt3_finetuned",
                      convert_rule="lang2ltl",
                      props=PROPS)
