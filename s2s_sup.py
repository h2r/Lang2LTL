"""
Infer trained model.
"""
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration

from s2s_hf_transformers import T5_PREFIX, T5_MODELS
from s2s_pt_transformer import Seq2SeqTransformer, \
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD, DIM_FFN_HID
from dataset import construct_dataset
from s2s_pt_transformer import translate as pt_transformer_translate
from s2s_pt_transformer import construct_dataset_meta as pt_transformer_construct_dataset_meta
from evaluation import evaluate_lang_from_file
from utils import count_params


class Seq2Seq:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type

        if args.model in T5_MODELS:  # https://huggingface.co/docs/transformers/model_doc/t5
            model_dir = f"model/{model_type}"
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        elif model_type == "pt_transformer":
            self.model = Seq2SeqTransformer(kwargs["src_vocab_sz"], kwargs["tar_vocab_sz"],
                                            NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD,
                                            DIM_FFN_HID)
            self.model_translate = pt_transformer_translate
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/symbolic_no_perm_batch1.csv', help='file path to train and test data for supervised seq2seq')
    parser.add_argument('--model', type=str, default="t5-base", choices=["t5-base", "t5-small", "pt_transformer"], help='name of supervised seq2seq model')
    parser.add_argument('--holdout_type', type=str, default='utt', help='type of holdout testing')
    parser.add_argument('--test_size', type=float, default=0.2, help='train test split ratio. used only when holdout_type=utt')
    parser.add_argument('--seed', type=int, default=42, help='random state for train test split. used only when holdout_type=utt')
    args = parser.parse_args()

    if args.holdout_type == "ltl_type":
        kwargs = {"holdout_types": ["sequenced_visit"]}
    elif args.holdout_type == "ltl_instance":
        kwargs = {"holdout_instances": [("sequenced_visit", 3)]}
    elif args.holdout_type == "utt":
        kwargs = {"test_size": args.test_size, "seed": args.seed}
    else:
        raise ValueError(f"ERROR unrecognized holdout type: {args.holdout_type}.")
    train_iter, train_meta, valid_iter, valid_meta = construct_dataset(args.data, args.holdout_type, **kwargs)

    if args.model in T5_MODELS:  # pretrained T5 from Hugging Face
        s2s = Seq2Seq(args.model)
    elif args.model == "pt_transformer":  # pretrained seq2seq transformer implemented in PyTorch
        vocab_transform, text_transform, src_vocab_size, tar_vocab_size = pt_transformer_construct_dataset_meta(train_iter)
        model_params = f"model/s2s_{args.model}_batch1.pth"
        s2s = Seq2Seq(args.model,
                      vocab_transform=vocab_transform, text_transform=text_transform,
                      src_vocab_sz=src_vocab_size, tar_vocab_sz=tar_vocab_size, fpath_load=model_params)
    else:
        raise TypeError(f"ERROR: unrecognized model, {args.model}")
    print(f"number of trainable parameters in {args.model}: {count_params(s2s)}")

    results_fpath = f"results/s2s_{args.model}_{Path(args.data).stem}_log.csv"
    analysis_fpath = "data/analysis_batch1.csv"
    acc_fpath = f"results/s2s_{args.model}_{Path(args.data).stem}_acc.csv"
    evaluate_lang_from_file(s2s, args.holdout_type, args.data, results_fpath, analysis_fpath, acc_fpath, **kwargs)
