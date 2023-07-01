import os
import logging
from openai.embeddings_utils import cosine_similarity

from gpt import GPT3, GPT4
from get_embed import generate_embeds
from s2s_sup_tcd import Seq2Seq
from s2s_hf_transformers import HF_MODELS
from formula_sampler import ALL_PROPS
from utils import load_from_file, save_to_file, build_placeholder_map, substitute

SHARED_DPATH = os.path.join(os.path.expanduser('~'), "data", "shared", "lang2ltl")  # group's data folder on cluster


def lang2ltl(utt, obj2sem, keep_keys,
             data_dpath=f"{SHARED_DPATH}/data", exp_name="lang2ltl-api",
             rer_model="gpt4", rer_engine="gpt-4", rer_prompt_fpath=f"{SHARED_DPATH}/data/rer_prompt_diverse_16.txt",
             embed_model="gpt3", embed_engine="text-embedding-ada-002", ground_model="gpt3", topk=2, update_embed=True,
             model_dpath=f"{SHARED_DPATH}/model_3000000", sym_trans_model="t5-base", convert_rule="lang2ltl", props=ALL_PROPS,
    ):
    if sym_trans_model in HF_MODELS:
        model_fpath = os.path.join(model_dpath, "t5-base", "checkpoint-best")
        translation_engine = model_fpath
    elif sym_trans_model == "gpt3_finetuned":
        translation_engine = f"gpt3_finetuned_symbolic_batch12_perm_utt_0.2_42"
        translation_engine = load_from_file(os.path.join(model_dpath, "gpt3_models.pkl"))[translation_engine]
    else:
        raise ValueError(f"ERROR: unrecognized symbolic translation model: {sym_trans_model}")

    logging.info(f"RER engine: {rer_engine}")
    logging.info(f"Embedding engine: {embed_model} {embed_engine}")
    logging.info(f"Symbolic translation engine: {translation_engine}\n")
    logging.info(f"Input Utterance to be translated:\n{utt}\n")

    res, utt2res = rer(rer_model, rer_engine, rer_prompt_fpath, [utt])
    logging.info(f"\nExtracted Referring Expressions (REs):\n{res}\n")

    obj2embed, obj2embed_fpath = generate_embeds(embed_model, data_dpath, obj2sem, keep_keys=keep_keys, embed_engine=embed_engine, exp_name=exp_name, update_embed=update_embed)
    logging.info(f"Generated Database of Embeddings for:\n{obj2sem}\nsaved at:\n{obj2embed_fpath}\n")

    re2embed_dpath = os.path.join(data_dpath, "re_embeds")
    os.makedirs(re2embed_dpath, exist_ok=True)
    re2embed_fpath = os.path.join(re2embed_dpath, f"re2embed_{exp_name}_{embed_model}-{embed_engine}.pkl")
    re2grounds = ground_res(res, re2embed_fpath, obj2embed_fpath, ground_model, embed_engine, topk)
    logging.info(f"Groundings for REs:\n{re2grounds}\n")

    ground_utts, objs_per_utt = ground_utterances([utt], utt2res, re2grounds)
    logging.info(f"Grounded Input Utterance:\n{ground_utts[0]}\ngroundings: {objs_per_utt[0]}\n")

    sym_utts, sym_ltls, out_ltls, placeholder_maps = translate_grounded_utts(ground_utts, objs_per_utt, sym_trans_model, translation_engine, convert_rule, props)
    logging.info(f"Placeholder Map:\n{placeholder_maps[0]}\n")
    logging.info(f"Symbolic Utterance:\n{sym_utts[0]}\n")
    logging.info(f"Translated Symbolic LTL Formula:\n{sym_ltls[0]}\n")
    logging.info(f"Grounded LTL Formula:\n{out_ltls[0]}\n\n\n")

    return out_ltls[0]


def rer(rer_model, rer_engine, rer_prompt, input_utts):
    """
    Referring Expression Recognition: extract name entities from input utterances.
    """
    rer_prompt = load_from_file(rer_prompt)

    if rer_model == "gpt3":
        rer_module = GPT3(rer_engine)
    elif rer_model == "gpt4":
        rer_module = GPT4(rer_engine)
    else:
        raise ValueError(f"ERROR: RER module not recognized: {rer_model}")

    names, utt2names = set(), []  # name entity list names should not have duplicates
    for idx_utt, utt in enumerate(input_utts):
        logging.info(f"Extracting referring expressions from utterance: {idx_utt}/{len(input_utts)}")
        names_per_utt = [name.strip() for name in rer_module.extract_re(query=f"{rer_prompt.strip()} {utt}\nPropositions:")]
        names_per_utt = list(set(names_per_utt))  # remove duplicated RE

        # extra_names = []  # make sure both 'name' and 'the name' are in names_per_utt to mitigate RER error
        # for name in names_per_utt:
        #     name_words = name.split()
        #     if name_words[0] == "the":
        #         extra_name = " ".join(name_words[1:])
        #     else:
        #         name_words.insert(0, "the")
        #         extra_name = " ".join(name_words)
        #     if extra_name not in names_per_utt:
        #         extra_names.append(extra_name)
        # names_per_utt += extra_names

        names.update(names_per_utt)
        utt2names.append((utt, names_per_utt))

    return names, utt2names


def ground_res(res, re2embed_fpath, obj_embed, ground_model, embed_engine, topk):
    """
    Find groundings (objects in given environment) of referring expressions (REs) extracted from input utterances.
    """
    obj2embed = load_from_file(obj_embed)  # load embeddings of known objects in given environment
    if os.path.exists(re2embed_fpath):  # load cached embeddings of referring expressions
        re2embed = load_from_file(re2embed_fpath)
    else:
        re2embed = {}

    if ground_model == "gpt3":
        ground_module = GPT3(embed_engine)
    else:
        raise ValueError(f"ERROR: grounding module not recognized: {ground_model}")

    re2grounds = {}
    is_new_embed = False
    for re in res:
        logging.info(f"grounding referring expression: {re}")
        if re in re2embed:  # use cached RE embedding if exists
            logging.info(f"use cached RE embedding: {re}")
            re_embed = re2embed[re]
        else:
            re_embed = ground_module.get_embedding(re)
            re2embed[re] = re_embed
            is_new_embed = True

        sims = {o: cosine_similarity(e, re_embed) for o, e in obj2embed.items()}
        sims_sorted = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        re2grounds[re] = list(dict(sims_sorted[:topk]).keys())

        if is_new_embed:
            save_to_file(re2embed, re2embed_fpath)

    return re2grounds


def ground_utterances(input_strs, utt2res, re2grounds):
    """
    Replace referring expressions in input utterances with best matching objects in given env.
    """
    grounding_maps = []  # name to grounding map per utterance
    for _, res in utt2res:
        grounding_maps.append({re: re2grounds[re][0] for re in res})

    output_strs, subs_per_str = substitute(input_strs, grounding_maps, is_utt=True)

    return output_strs, subs_per_str


def translate_grounded_utts(ground_utts, objs_per_utt, sym_trans_model, translation_engine, convert_rule, props, trans_modular_prompt=None):
    """
    Translation language to LTL modular approach.
    :param ground_utts: Input utterances with name entities grounded to objects in given environment.
    :param objs_per_utt: grounding objects for each input utterance.
    :param sym_trans_model: symbolic translation model, gpt3_finetuned, gpt3_pretrained, t5-base.
    :param translation_engine: pretrained T5 model weights, finetuned or pretrained GPT-3 engine to use for translation.
    :param convert_rule: referring expression to proposition conversion rule.
    :param props: all possible propositions.
    :param trans_modular_prompt: prompt for pretrained GPT-3.
    :return: output grounded LTL formulas, corresponding intermediate symbolic LTL formulas, placeholder maps
    """
    if sym_trans_model in HF_MODELS:
        trans_module = Seq2Seq(translation_engine, sym_trans_model)
    elif "gpt3" in sym_trans_model:
        trans_module = GPT3(translation_engine)
        if "ft" in translation_engine:
            trans_modular_prompt = ""
        elif "text-davinci" in translation_engine:
            trans_modular_prompt = load_from_file(trans_modular_prompt)
        else:
            raise ValueError(f"ERROR: Unrecognized translation engine: {translation_engine}")
    else:
        raise ValueError(f"ERROR: translation module not recognized: {sym_trans_model}")

    placeholder_maps, placeholder_maps_inv = [], []
    for objs in objs_per_utt:
        placeholder_map, placeholder_map_inv = build_placeholder_map(objs, convert_rule, props)
        placeholder_maps.append(placeholder_map)
        placeholder_maps_inv.append(placeholder_map_inv)
    symbolic_utts, _ = substitute(ground_utts, placeholder_maps, is_utt=True)  # replace names by symbols

    symbolic_ltls = []
    for idx, sym_utt in enumerate(symbolic_utts):
        logging.info(f"Symbolic Translation: {idx}/{len(symbolic_utts)}")
        query = sym_utt.translate(str.maketrans('', '', ',.'))
        if "gpt3" in sym_trans_model:
            query = f"Utterance: {query}\nLTL:"  # query format for finetuned GPT-3
            ltl = trans_module.translate(query, trans_modular_prompt)[0]
        else:
            ltl = trans_module.type_constrained_decode([query])[0]
        symbolic_ltls.append(ltl)
    output_ltls, _ = substitute(symbolic_ltls, placeholder_maps_inv, is_utt=False)  # replace symbols by props

    return symbolic_utts, symbolic_ltls, output_ltls, placeholder_maps
