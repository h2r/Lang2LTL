import argparse
from utils import load_from_file
from lang2ltl import rer, ground_utterances, translate_modular

PROPS = ["a", "b", "c", "d", "h", "j", "k", "l", "n", "o", "p", "q", "r", "s", "y", "z"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rer", type=str, default="gpt3", choices=["gpt3", "gpt4", "llama-7B"], help="Referring Expressoin Recognition module.")
    parser.add_argument("--rer_engine", type=str, default="text-davinci-003", choices=["text-davinci-003", "gpt4", "llama-7B"], help="pretrained LLM for RER.")
    parser.add_argument("--rer_prompt", type=str, default="data/osm/rer_prompt_16.txt", help="path to RER prompt.")
    parser.add_argument("--sym_trans", type=str, default="gpt3_finetuned", choices=["gpt3_finetuned", "gpt3_pretrained", "t5-base", "t5-small", "pt_transformer"], help="symbolic translation module.")
    parser.add_argument("--convert_rule", type=str, default="lang2ltl", choices=["lang2ltl", "cleanup"], help="name to prop conversion rule.")
    parser.add_argument("--checkpoint", type=str, default=None, help="#checkpoint for t5 model")
    args = parser.parse_args()

    translation_engine = f"gpt3_finetuned_symbolic_batch12_perm_utt_0.2_42"
    translation_engine = load_from_file("model/gpt3_models.pkl")[translation_engine]

    # utt = 'go to the burger queen and then to cafe on Thayer'
    utt = 'go to the red brick building without passing by the CS department'

    res, utt2names = rer(args.rer, args.rer_engine, args.rer_prompt, [utt])
    symbolic_utts, symbolic_ltls, output_ltls, placeholder_maps = translate_modular([utt], [res], args.sym_trans, translation_engine, args.convert_rule, PROPS, checkpoint=args.checkpoint)
    print(f'orginal utt: {utt}\n\nsymbolic utt: {symbolic_utts}\n\nsymbolic ltl: {symbolic_ltls}\n\nfinal ltl: {output_ltls}\n\nplaceholder_maps: {placeholder_maps}')