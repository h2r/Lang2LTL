"""
Infer trained model.
"""
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
import spot

from s2s_hf_transformers import T5_PREFIX, T5_MODELS
from s2s_pt_transformer import Seq2SeqTransformer, \
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD, DIM_FFN_HID
from s2s_pt_transformer import translate as transformer_translate
from s2s_pt_transformer import construct_dataset as transformer_construct_dataset
from utils import save_to_file


class Seq2Seq:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type

        if args.model in T5_MODELS:  # https://huggingface.co/docs/transformers/model_doc/t5
            self.tokenizer = AutoTokenizer.from_pretrained(args.model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        elif model_type == "pt_transformer":
            self.model = Seq2SeqTransformer(kwargs["src_vocab_sz"], kwargs["tar_vocab_sz"],
                                            NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD,
                                            DIM_FFN_HID)
            self.model_translate = transformer_translate
            self.vocab_transform = vocab_transform
            self.text_transform = text_transform
            self.model.load_state_dict(torch.load(kwargs["fpath_load"]))
        else:
            raise ValueError(f'ERROR: unrecognized model: {model_type}')

    def translate(self, queries):
        if self.model_type in T5_MODELS:
            inputs = [f"{T5_PREFIX}{query}" for query in queries]  # add prefix
            inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
            output_tokens = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,
            )
            ltls = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        elif self.model_type == "pt_transformer":
            ltls = [self.model_translate(self.model, self.vocab_transform, self.text_transform, queries[0])]
        else:
            raise ValueError(f'ERROR: unrecognized model, {self.model_type}')
        return ltls

    def parameters(self):
        return self.model.parameters()


def evaluate(s2s, results_fpath):
    _, valid_iter, _, _, _, _ = transformer_construct_dataset(args.data)
    results = [["utterances", "true_ltl", "output_ltl", "is_correct"]]
    accs = []

    for utt, true_ltl in valid_iter:
        out_ltl = s2s.translate([utt])[0]
        try:  # output LTL formula may have syntax error
            is_correct = spot.are_equivalent(spot.formula(out_ltl), spot.formula(true_ltl))
            is_correct = 'True' if is_correct else 'False'
        except SyntaxError:
            print(f'Syntax error in output LTL: {out_ltl}')
            is_correct = 'Syntax Error'
        accs.append(is_correct)
        results.append([utt, true_ltl, out_ltl, is_correct])

    save_to_file(results, f"{results_fpath}.csv")
    print(np.mean([True if acc == 'True' else False for acc in accs]))


def count_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/symbolic_pairs_no_perm.csv', help='file path to train and test data for supervised seq2seq')
    parser.add_argument('--model', type=str, default="t5-base", choices=["t5-base", "pt_transformer"], help='name of supervised seq2seq model')
    parser.add_argument('--utt', type=str, default="visit a while avoiding b then go to b", help='utterance to translate')
    args = parser.parse_args()

    if args.model in T5_MODELS:  # pretrained T5 from Hugging Face
        s2s = Seq2Seq(args.model)
    elif args.model == "pt_transformer":  # pretrained seq2seq transformer
        _, _, vocab_transform, text_transform, src_vocab_size, tar_vocab_size = transformer_construct_dataset(args.data)
        model_params = f"model/s2s_{args.model}.pth"
        s2s = Seq2Seq(args.model,
                      vocab_transform=vocab_transform, text_transform=text_transform,
                      src_vocab_sz=src_vocab_size, tar_vocab_sz=tar_vocab_size, fpath_load=model_params)
    else:
        raise TypeError(f"ERROR: unrecognized model, {args.model}")

    print(f"number of trainable parameters in {args.model}: {count_params(s2s)}")

    evaluate(s2s, f"results/s2s_{args.model}_batch_1_results")

    # ltls = s2s.translate([args.utt])
    # print(args.utt)
    # print(ltls)
