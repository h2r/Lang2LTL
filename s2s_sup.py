"""
Infer trained model.
"""
import argparse
import logging
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration

from s2s_hf_transformers import T5_PREFIX, T5_MODELS
from s2s_pt_transformer import Seq2SeqTransformer, \
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD, DIM_FFN_HID
from s2s_pt_transformer import translate as pt_transformer_translate
from s2s_pt_transformer import construct_dataset_meta as pt_transformer_construct_dataset_meta
from dataset_symbolic import load_split_dataset
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
    parser.add_argument('--split_dataset_fpath', type=str, default='data/split_symbolic_no_perm_batch1_ltl_instance_0.2_42.pkl',
                        help='complete file path or prefix of file paths to train test split dataset')
    parser.add_argument('--model', type=str, default="t5-base", choices=["t5-base", "t5-small", "pt_transformer"],
                        help='name of supervised seq2seq model')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(f'results/s2s_{args.model}_{Path(args.split_dataset_fpath).stem}.log', mode='w'),
                            logging.StreamHandler()
                        ]
    )

    if "pkl" in args.split_dataset_fpath:  # complete file path, e.g. data/split_symbolic_no_perm_batch1_utt_0.2_42.pkl
        split_dataset_fpaths = [args.split_dataset_fpath]
    else:  # prefix of file paths, e.g. split_symbolic_no_perm_batch1_utt
        split_dataset_fpaths = [os.path.join("data", fpath) for fpath in os.listdir("data") if args.split_dataset_fpath in fpath]

    for split_dataset_fpath in split_dataset_fpaths:
        # Load train, test data
        train_iter, train_meta, valid_iter, valid_meta = load_split_dataset(split_dataset_fpath)

        # Load trained model
        if args.model in T5_MODELS:  # pretrained T5 from Hugging Face
            s2s = Seq2Seq(args.model)
        elif args.model == "pt_transformer":  # pretrained seq2seq transformer implemented in PyTorch
            vocab_transform, text_transform, src_vocab_size, tar_vocab_size = pt_transformer_construct_dataset_meta(train_iter)
            model_params = f"model/s2s_{args.model}_{Path(split_dataset_fpath).stem}.pth"
            s2s = Seq2Seq(args.model,
                          vocab_transform=vocab_transform, text_transform=text_transform,
                          src_vocab_sz=src_vocab_size, tar_vocab_sz=tar_vocab_size, fpath_load=model_params)
        else:
            raise TypeError(f"ERROR: unrecognized model, {args.model}")
        logging.info(f"Number of trainable parameters in {args.model}: {count_params(s2s)}")
        logging.info(f"Number of training samples: {len(train_iter)}")
        logging.info(f"Number of validation samples: {len(valid_iter)}\n")

        # Evaluation
        result_log_fpath = f"results/s2s_{args.model}_{Path(split_dataset_fpath).stem}_log.csv"
        analysis_fpath = "data/analysis_batch1.csv"
        acc_fpath = f"results/s2s_{args.model}_{Path(split_dataset_fpath).stem}_acc.csv"
        evaluate_lang_from_file(s2s, split_dataset_fpath, analysis_fpath, result_log_fpath, acc_fpath)
