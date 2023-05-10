import os
import argparse
from utils import load_from_file
from lang2ltl import rer, translate_modular, PROPS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rer", type=str, default="gpt3", choices=["gpt3", "gpt4", "llama-7B"], help="Referring Expressoin Recognition module.")
    parser.add_argument("--rer_engine", type=str, default="text-davinci-003", choices=["text-davinci-003", "gpt4", "llama-7B"], help="pretrained LLM for RER.")
    parser.add_argument("--rer_prompt", type=str, default="data/osm/rer_prompt_16.txt", help="path to RER prompt.")
    parser.add_argument("--sym_trans", type=str, default="gpt3_finetuned", choices=["t5-base", "gpt3_finetuned", "gpt3_pretrained"], help="symbolic translation module.")
    parser.add_argument("--model_dpath", type=str, default=None, help="directory to model checkpoints.")
    parser.add_argument("--model2ckpt_fpath", type=str, default=None, help="best checkpoint for models.")
    parser.add_argument("--convert_rule", type=str, default="lang2ltl", choices=["lang2ltl", "cleanup"], help="name to prop conversion rule.")
    parser.add_argument("--checkpoint", type=str, default=None, help="#checkpoint for t5 model")
    args = parser.parse_args()

    if args.sym_trans == "gpt3_finetuned":
        translation_engine = f"gpt3_finetuned_symbolic_batch12_perm_utt_0.2_42"
        translation_engine = load_from_file("model/gpt3_models.pkl")[translation_engine]
    elif "t5" in args.sym_trans:
        checkpoint = load_from_file(args.model2ckpt_fpath)[args.sym_trans]
        translation_engine = os.path.join(args.model_dpath, args.sym_trans, f"checkpoint-{checkpoint}")
    else:
        raise ValueError(f"ERROR: unrecognized symbolic translation model: {args.sym_trans}")

    utt = "go to the red brick building without passing by the CS department"

    res, utt2names = rer(args.rer, args.rer_engine, args.rer_prompt, [utt])
    symbolic_utts, symbolic_ltls, output_ltls, placeholder_maps = translate_modular([utt], [res], args.sym_trans, translation_engine, args.convert_rule, PROPS)
    print(f"Input utt: {utt}\n\nSymbolic utt: {symbolic_utts}\n\nSymbolic ltl: {symbolic_ltls}\n\nOutput ltl: {output_ltls}\n\nPlaceholder map: {placeholder_maps}")
