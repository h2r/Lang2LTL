import os
import time
import logging
from openai.embeddings_utils import cosine_similarity

from get_embed import store_embeds
from gpt import GPT3, GPT4
from s2s_sup import Seq2Seq, T5_MODELS
from s2s_pt_transformer import construct_dataset_meta
from dataset_symbolic import load_split_dataset
from utils import load_from_file, save_to_file, build_placeholder_map, substitute

PROPS = ["a", "b", "c", "d", "h", "j", "k", "l", "n", "o", "p", "q", "r", "s", "y", "z"]


def lang2ltl(utt, lmk2sem, result_dpath,
             embed_model="gpt3", embed_engine="text-embedding-ada-002",
             rer_model="gpt3", rer_engine="text-davinci-003", rer_prompt_fpath="data/osm/rer_prompt_16.txt", update_embed=False,
             ground_model="gpt3", topk=2,
             sym_trans_model="gpt3_finetuned", translation_engine="gpt3_finetuned_symbolic_batch12_perm_utt_0.2_42", finetuned_models="model/gpt3_models.pkl", convert_rule="lang2ltl", props=PROPS,
    ):
    translation_engine = load_from_file(finetuned_models)[translation_engine]

    logging.info(f"RER engine: {rer_engine}")
    logging.info(f"Embedding engine: {embed_engine}")
    logging.info(f"Symbolic translation engine: {translation_engine}\n")
    logging.info(f"Input Utterance to be translated:\n{utt}\n")

    obj2embed_fpath, obj2embed = store_embeds(embed_model, result_dpath, lmk2sem, [], embed_engine, update_embed)
    logging.info(f"\nGenerated Embeddings for:\n{lmk2sem}\nsaved at:\n{obj2embed_fpath}\n")

    res, utt2res = rer(rer_model, rer_engine, rer_prompt_fpath, [utt])
    logging.info(f"\nExtracted Referring Expressions (REs):\n{res}\n")

    re2embed_dpath = os.path.join(result_dpath, "lmk_name_embeds")
    os.makedirs(re2embed_dpath, exist_ok=True)
    re2embed_fpath = os.path.join(re2embed_dpath, f"name2embed_lang2ltl_api_{embed_engine}.pkl")
    name2grounds = ground_names(res, re2embed_fpath, obj2embed_fpath, ground_model, embed_engine, topk)
    logging.info(f"Groundings for REs:\n{name2grounds}\n")

    ground_utts, objs_per_utt = ground_utterances([utt], utt2res, name2grounds)
    logging.info(f"Grounded Input Utterance:\n{ground_utts[0]}\ngroundings: {objs_per_utt[0]}\n")

    sym_utts, sym_ltls, out_ltls, placeholder_maps = translate_modular(ground_utts, objs_per_utt, sym_trans_model, translation_engine, convert_rule, props)
    logging.info(f"Placeholder Map:\n{placeholder_maps[0]}\n")
    logging.info(f"Symbolic Utterance:\n{sym_utts[0]}\n")
    logging.info(f"Translated Symbolic LTL Formula:\n{sym_ltls[0]}\n")
    logging.info(f"Grounded LTL Formula:\n{out_ltls[0]}\n")

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


def ground_names(names, name_embed, obj_embed, ground_model, embed_engine, topk):
    """
    Find groundings (objects in given environment) of referring expressions (REs) in input utterances.
    """
    obj2embed = load_from_file(obj_embed)  # load embeddings of known objects in given environment
    if os.path.exists(name_embed):  # load cached embeddings of name entities
        name2embed = load_from_file(name_embed)
    else:
        name2embed = {}

    if ground_model == "gpt3":
        ground_module = GPT3(embed_engine)
    # elif embed_model == "bert":
    #     ground_module = BERT()
    else:
        raise ValueError(f"ERROR: grounding module not recognized: {ground_model}")

    name2grounds = {}
    is_embed_added = False
    for name in names:
        logging.info(f"grounding landmark: {name}")
        if name in name2embed:  # use cached embedding if exists
            logging.info(f"use cached embedding: {name}")
            embed = name2embed[name]
        else:
            embed = ground_module.get_embedding(name)
            name2embed[name] = embed
            is_embed_added = True

        sims = {n: cosine_similarity(e, embed) for n, e in obj2embed.items()}
        sims_sorted = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        name2grounds[name] = list(dict(sims_sorted[:topk]).keys())

        if is_embed_added:
            save_to_file(name2embed, name_embed)

    return name2grounds


def ground_utterances(input_strs, utt2names, name2grounds):
    """
    Replace name entities in input strings (e.g. utterances, LTL formulas) with objects in given environment.
    """
    grounding_maps = []  # name to grounding map per utterance
    for _, names in utt2names:
        grounding_maps.append({name: name2grounds[name][0] for name in names})

    output_strs, subs_per_str = substitute(input_strs, grounding_maps, is_utt=True)

    # breakpoint()

    return output_strs, subs_per_str


def translate_modular(ground_utts, objs_per_utt, sym_trans_model, translation_engine, convert_rule, props, trans_modular_prompt=None, s2s_sup_data=None):
    """
    Translation language to LTL modular approach.
    :param ground_utts: Input utterances with name entities grounded to objects in given environment.
    :param objs_per_utt: grounding objects for each input utterance.
    :param sym_trans_model: symbolic translation model, gpt3_finetuned, gpt3_pretrained, t5-base, t5-small, pt_transformer
    :param translation_engine: finetuned or pretrained GPT-3 engine to use for translation.
    :param convert_rule: referring expression to proposition conversion rule.
    :param props: all possible propositions.
    :param trans_modular_prompt: prompt for pretrained GPT-3.
    :param s2s_sup_data: file path to train and test data for supervised seq2seq.
    :return: output grounded LTL formulas, corresponding intermediate symbolic LTL formulas, placeholder maps
    """
    if "ft" in translation_engine:
        trans_modular_prompt = ""
    elif "text-davinci" in translation_engine:
        trans_modular_prompt = load_from_file(trans_modular_prompt)
    else:
        raise ValueError(f"ERROR: Unrecognized translation engine: {translation_engine}")

    if "gpt3" in sym_trans_model:
        trans_module = GPT3(translation_engine)
    elif sym_trans_model in T5_MODELS:
        trans_module = Seq2Seq(sym_trans_model)
    elif sym_trans_model == "pt_transformer":
        train_iter, _, _, _ = load_split_dataset(s2s_sup_data)
        vocab_transform, text_transform, src_vocab_size, tar_vocab_size = construct_dataset_meta(train_iter)
        model_params = f"model/s2s_{sym_trans_model}.pth"
        trans_module = Seq2Seq(sym_trans_model,
                               vocab_transform=vocab_transform, text_transform=text_transform,
                               src_vocab_sz=src_vocab_size, tar_vocab_sz=tar_vocab_size, fpath_load=model_params)
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
        query = f"Utterance: {query}\nLTL:"  # query format for finetuned GPT-3
        ltl = trans_module.translate(query, trans_modular_prompt)[0]
        # try:
        #     spot.formula(ltl)
        # except SyntaxError:
        #     ltl = feedback_module(trans_module, query, trans_modular_prompt, ltl)
        symbolic_ltls.append(ltl)

    output_ltls, _ = substitute(symbolic_ltls, placeholder_maps_inv, is_utt=False)  # replace symbols by props

    return symbolic_utts, symbolic_ltls, output_ltls, placeholder_maps


if __name__ == "__main__":
    result_dpath = os.path.join("results", "lang2ltl_api")
    os.makedirs(result_dpath, exist_ok=True)
    result_fpath = os.path.join(result_dpath, f"log_{time.time()}.log")
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(result_fpath, mode='w'),
                            logging.StreamHandler()
                        ]
    )

    utt = "Go to bookshelf first, then workstation A, then go to counter, then back to workstation A."
    lmk2sem = {
        "bookshelf": {},
        "desk A": {},
        "kitchen counter": {},
        "desk B": {},
    }
    out_ltl = lang2ltl(utt, lmk2sem, result_dpath)
