"""
Infer trained model with type constrained decoding.
"""
import os
import argparse
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from formula_sampler import PROPS
from s2s_hf_transformers import T5_PREFIX, HF_MODELS
from s2s_pt_transformer import Seq2SeqTransformer, \
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD, DIM_FFN_HID
from s2s_pt_transformer import translate as pt_transformer_translate
from s2s_pt_transformer import construct_dataset_meta as pt_transformer_construct_dataset_meta
from dataset_symbolic import load_split_dataset
from eval import evaluate_sym_trans
from utils import count_params

S2S_MODELS = HF_MODELS.extend(["pt_transformer"])
UNARY_OPERATORS = ['!', "F", "G", "X"]
BINARY_OPERATORS = ['&', '|', 'U', 'i', 'e', 'M']
END_TOKEN = '</s>'
MAX_LENGTH = 256
MAX_DEPTH = 21
CHECK_DEPTH = 100


def is_valid(formula: list[str], next_token):
    if len(formula) == 0:
        return True
    else:
        if formula[0] in UNARY_OPERATORS:
            prop_counter = 1
        elif formula[0] in BINARY_OPERATORS:
            prop_counter = 2
        else:
            prop_counter = 0

        for i in range(1, len(formula)):
            if formula[i] not in UNARY_OPERATORS and formula[i] not in BINARY_OPERATORS:
                prop_counter -= 1
            elif formula[i] in BINARY_OPERATORS:
                prop_counter += 1

        if next_token in UNARY_OPERATORS:
            return prop_counter > 0
        elif next_token in BINARY_OPERATORS:
            return prop_counter > 0
        elif next_token == END_TOKEN:
            return prop_counter == 0
        else:
            return prop_counter > 0


class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def depth(root):
    """
    Returns the depth of a binary tree
    """
    if root is None:
        return 0
    leftAns = depth(root.left)
    rightAns = depth(root.right)
    return max(leftAns, rightAns) + 1


def build_tree(ltl):
    """
    Construct a binary tree from LTL formula string and return as p, partial formula supported
    """
    if ltl == '':  # its the end of the expression
        return Node(''), ''
    if ltl[0] in PROPS:  # the character is a proposition
        return Node(ltl[0]), ltl[1:]
    elif ltl[0] in BINARY_OPERATORS:
        # Create a node with ltl[0] as the data and both the children set to null
        p = Node(ltl[0])
        # Build the left sub-tree
        p.left, q = build_tree(ltl[1:])
        # Build the right sub-tree
        p.right, q = build_tree(q)
        return p, q


class Seq2Seq:
    def __init__(self, model_dpath, model_name, **kwargs):
        self.model_name = model_name

        if "t5" in model_name or "bart" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dpath)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dpath)
        elif model_name == "pt_transformer":
            self.model = Seq2SeqTransformer(kwargs["src_vocab_sz"], kwargs["tar_vocab_sz"],
                                            NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD,
                                            DIM_FFN_HID)
            self.model_translate = pt_transformer_translate
            self.vocab_transform = vocab_transform
            self.text_transform = text_transform
            self.model.load_state_dict(torch.load(kwargs["fpath_load"]))
        else:
            raise ValueError(f'ERROR: unrecognized model: {model_name}')

    def translate(self, queries):
        if "t5" in self.model_name or "bart" in self.model_name:
            inputs = [f"{T5_PREFIX}{query}" for query in queries]  # add prefix
            inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
            output_tokens = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,
                max_new_tokens=256,
            )
            ltls = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        elif self.model_name == "pt_transformer":
            ltls = [self.model_translate(self.model, self.vocab_transform, self.text_transform, queries[0])]
        else:
            raise ValueError(f'ERROR: unrecognized model, {self.model_name}')
        return ltls

    def type_constrained_decode(self, utts):
        """
        type constrained decoding based on LTL syntax:
        mask_and_regen() :: next_token is invalid, mask it and pick the second highest one
        add_and_gen_new() :: next_token is valid, append it to the partial formula and generate a new one

        Logic: 
        if len(ltl) == 0: mask_and_regen(next_token) if next_token == whitespace
            |
        elif len(ltl) > CHECK_DEPTH: if depth(ltl) > MAX_DEPTH: only props and <EOS> allowed
            |                                       |
            |                                       else: regular_generation(next_token)
            |
        elif not next_token == whitespace: regular_generation(next_token)
            |
        elif next_token == whitespace: mask_and_regen(next_token) if ltl[-1] == whitespace else add_and_gen_new(next_token)
            |
        else: mask_and_regen(next_token)
        """
        def mask_and_regen(lm_logits, next_decoder_input_ids):
            lm_logits[:, -1:, next_decoder_input_ids.item()] = float('-inf')
            next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
            next_token = self.tokenizer.decode(next_decoder_input_ids[0], skip_special_tokens=False)
            return next_token, next_decoder_input_ids, lm_logits

        def add_and_gen_new(lm_logits, next_decoder_input_ids, decoder_input_ids, next_token, whitespace):
            whitespace = next_token == ''
            decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
            lm_logits = self.model(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True).logits
            next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
            next_token = self.tokenizer.decode(next_decoder_input_ids[0], skip_special_tokens=False)
            return next_token, next_decoder_input_ids, decoder_input_ids, lm_logits, whitespace

        if "t5" in self.model_name or "bart" in self.model_name:
            inputs = [f"{T5_PREFIX}{utt}" for utt in utts]  # add prefix
            inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids
            decoder_input_ids = self.tokenizer("<pad>", add_special_tokens=False, return_tensors="pt").input_ids
            outputs = self.model(input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
            encoded_sequence = (outputs.encoder_last_hidden_state,)
            lm_logits = outputs.logits
            next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
            next_token = self.tokenizer.decode(next_decoder_input_ids[0], skip_special_tokens=False)
            token_list = []
            whitespace = False
            while '</s>' not in token_list and len(token_list) < MAX_LENGTH:
                # decode the first token
                if len(token_list) == 0:
                    if next_token in PROPS + UNARY_OPERATORS + BINARY_OPERATORS:
                        token_list.append(next_token)
                        next_token, next_decoder_input_ids, decoder_input_ids, lm_logits, whitespace = add_and_gen_new(lm_logits, next_decoder_input_ids, decoder_input_ids, next_token, whitespace)
                    elif next_token == '':
                        next_token, next_decoder_input_ids, decoder_input_ids, lm_logits, whitespace = add_and_gen_new(lm_logits, next_decoder_input_ids, decoder_input_ids, next_token, whitespace)
                    else:
                        next_token, next_decoder_input_ids, lm_logits = mask_and_regen(lm_logits, next_decoder_input_ids)
                # after the first token
                # only check depth after certain # of operators & props
                elif len(token_list) > CHECK_DEPTH:
                    no_uni_list = ''.join([s for s in token_list if not s in UNARY_OPERATORS])
                    partial_tree, _ = build_tree(no_uni_list)
                    # if max_depth is reached
                    if depth(partial_tree) > MAX_DEPTH:
                        # only output props and <EOS>
                        if not (next_token in PROPS or next_token == END_TOKEN or next_token == ''):
                            next_token, next_decoder_input_ids, lm_logits = mask_and_regen(lm_logits, next_decoder_input_ids)
                        elif next_token in PROPS or next_token == END_TOKEN:
                            if not is_valid(token_list, next_token):
                                next_token, next_decoder_input_ids, lm_logits = mask_and_regen(lm_logits, next_decoder_input_ids)

                            else:
                                token_list.append(next_token)
                                next_token, next_decoder_input_ids, decoder_input_ids, lm_logits, whitespace = add_and_gen_new(lm_logits, next_decoder_input_ids, decoder_input_ids, next_token, whitespace)
                        # no consecutive whitespaces
                        elif next_token == ' ' or next_token == '':
                            if whitespace == True:
                                next_token, next_decoder_input_ids, lm_logits = mask_and_regen(lm_logits, next_decoder_input_ids)
                            else:
                                next_token, next_decoder_input_ids, decoder_input_ids, lm_logits, whitespace = add_and_gen_new(lm_logits, next_decoder_input_ids, decoder_input_ids, next_token, whitespace)
                        # mask all other tokens
                        else:
                            next_token, next_decoder_input_ids, lm_logits = mask_and_regen(lm_logits, next_decoder_input_ids)
                    else:
                        if next_token in PROPS + UNARY_OPERATORS + BINARY_OPERATORS + [END_TOKEN]:
                            if not is_valid(token_list, next_token):
                                next_token, next_decoder_input_ids, lm_logits = mask_and_regen(lm_logits, next_decoder_input_ids)

                            else:
                                token_list.append(next_token)
                                next_token, next_decoder_input_ids, decoder_input_ids, lm_logits, whitespace = add_and_gen_new(lm_logits, next_decoder_input_ids, decoder_input_ids, next_token, whitespace)
                        # no consecutive whitespaces
                        elif next_token == ' ' or next_token == '':
                            if whitespace == True:
                                next_token, next_decoder_input_ids, lm_logits = mask_and_regen(lm_logits, next_decoder_input_ids)
                            else:
                                next_token, next_decoder_input_ids, decoder_input_ids, lm_logits, whitespace = add_and_gen_new(lm_logits, next_decoder_input_ids, decoder_input_ids, next_token, whitespace)
                # mask all other tokens
                        else:
                            next_token, next_decoder_input_ids, lm_logits = mask_and_regen(lm_logits, next_decoder_input_ids)

                elif next_token in PROPS + UNARY_OPERATORS + BINARY_OPERATORS + [END_TOKEN]:
                    if not is_valid(token_list, next_token):
                        next_token, next_decoder_input_ids, lm_logits = mask_and_regen(lm_logits, next_decoder_input_ids)

                    else:
                        token_list.append(next_token)
                        next_token, next_decoder_input_ids, decoder_input_ids, lm_logits, whitespace = add_and_gen_new(lm_logits, next_decoder_input_ids, decoder_input_ids, next_token, whitespace)
                # no consecutive whitespaces
                elif next_token == ' ' or next_token == '':
                    if whitespace == True:
                        next_token, next_decoder_input_ids, lm_logits = mask_and_regen(lm_logits, next_decoder_input_ids)
                    else:
                        next_token, next_decoder_input_ids, decoder_input_ids, lm_logits, whitespace = add_and_gen_new(lm_logits, next_decoder_input_ids, decoder_input_ids, next_token, whitespace)
                # mask all other tokens
                else:
                    next_token, next_decoder_input_ids, lm_logits = mask_and_regen(lm_logits, next_decoder_input_ids)
            
            return [' '.join(token_list[:-1])]

    def parameters(self):
        return self.model.parameters()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", type=str, default="data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_0.pkl", help="complete file path or prefix of file paths to train test split dataset.")
    parser.add_argument("--model_dpath", type=str, default=None, help="directory to save model checkpoints.")
    parser.add_argument("--model", type=str, default="t5-base", choices=S2S_MODELS, help="name of supervised seq2seq model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint to use for inferance.")
    args = parser.parse_args()

    ckpt_dname = f"checkpoint-{args.checkpoint}" if args.checkpoint else "checkpoint-best"

    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(f'results/s2s_tcd_{args.model}_{Path(args.data_fpath).stem}.log', mode='w'),
                            logging.StreamHandler()
                        ]
    )

    if "pkl" in args.data_fpath:  # complete file path, e.g. data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_0.pkl
        data_fpaths = [args.data_fpath]
    else:  # prefix of file paths, e.g. data/holdout_split_batch12_perm/symbolic_batch12_perm_utt
        data_dpath = os.path.dirname(args.data_fpath)
        fname_prefix = os.path.basename(args.data_fpath)
        data_fpaths = [os.path.join(data_dpath, fpath) for fpath in os.listdir(data_dpath) if fname_prefix in fpath]

    for data_fpath in data_fpaths:
        logging.info(f"Inference dataset: {data_fpath}")
        # Load train, test data
        train_iter, train_meta, valid_iter, valid_meta = load_split_dataset(data_fpath)

        # Load trained model
        if "t5" in args.model or "bart" in args.model:  # pretrained T5/Bart from Hugging Face
            model_dpath = os.path.join(args.model_dpath, args.model, ckpt_dname)
            logging.info(f"Load model and checkpoint: {model_dpath}")
            s2s = Seq2Seq(model_dpath, args.model)
        elif args.model == "pt_transformer":  # pretrained seq2seq transformer implemented in PyTorch
            vocab_transform, text_transform, src_vocab_size, tar_vocab_size = pt_transformer_construct_dataset_meta(train_iter)
            model_params = f"model/s2s_{args.model}_{Path(data_fpath).stem}.pth"
            s2s = Seq2Seq(args.model_dpath, args.model,
                          vocab_transform=vocab_transform, text_transform=text_transform,
                          src_vocab_sz=src_vocab_size, tar_vocab_sz=tar_vocab_size, fpath_load=model_params)
        else:
            raise TypeError(f"ERROR: unrecognized model, {args.model}")
        logging.info(f"Number of trainable parameters in {args.model}: {count_params(s2s)}")
        logging.info(f"Number of training samples: {len(train_iter)}")
        logging.info(f"Number of validation samples: {len(valid_iter)}\n")

        # Evaluate
        result_log_fpath = f"results/s2s_{args.model}-{ckpt_dname}_{Path(data_fpath).stem}_log.csv"
        analysis_fpath = "data/analysis_symbolic_batch12_perm.csv"
        acc_fpath = f"results/s2s_{args.model}-{ckpt_dname}_{Path(data_fpath).stem}_acc.csv"
        evaluate_sym_trans(s2s, data_fpath, result_log_fpath, analysis_fpath, acc_fpath)
