"""
Finetune pre-trained transformer models from Hugging Face.
"""
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils import load_from_file

T5_PREFIX = "translate English to Linear Temporal Logic: "


def construct_dataset(fpath):
    data = load_from_file(fpath)
    input_sequences, output_sequences = [], []
    for utt, ltl in data:
        input_sequences.append(f"{T5_PREFIX}{utt}")
        output_sequences.append(ltl)
    return input_sequences, output_sequences


def finetune_t5(input_sequences, output_sequences, tokenizer, model):
    max_source_length = 512
    max_target_length = 128

    source_encoding = tokenizer(
        input_sequences,
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids, attention_mask = source_encoding.input_ids, source_encoding.attention_mask

    target_encoding = tokenizer(
        output_sequences,
        padding="longest",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt",
    )
    labels = target_encoding.input_ids
    labels[labels == tokenizer.pad_token_id] = -100  # replace padding token id's of the labels by -100 so it's ignored by the loss

    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    print(loss.items())

    breakpoint()

    input_ids = tokenizer(f"{T5_PREFIX}visit a then b")
    outputs = model.generate(input_ids)
    ltl = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(ltl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/symbolic_pairs.csv', help='file path to train and test data for supervised seq2seq')
    parser.add_argument('--model', type=str, default='t5-small', choices=["t5-small", "bart"], help='name of supervised seq2seq model')
    args = parser.parse_args()

    input_sequences, output_sequences = construct_dataset(args.data)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    if "t5" in args.model:
        finetune_t5(input_sequences, output_sequences, tokenizer, model)
