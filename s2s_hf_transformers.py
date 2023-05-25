"""
Finetune pre-trained transformer models from Hugging Face.
"""
import os
import argparse
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import evaluate

from dataset_symbolic import load_split_dataset

HF_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b", "facebook/bart-base"]
T5_PREFIX = "translate English to Linear Temporal Logic: "
MAX_SRC_LEN = 512
MAX_TAR_LEN = 256
EPOCHS = 10
BATCH_SIZE = 20
BASE_DATASET_SIZE = 49655


def finetune_t5(model_name, data_fpath, end_idx, model_dpath=None, valid_size=0.2, test_size=0.1):
    """
    Followed most of the tutorial at
    https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887

    Finetune T5 for translation example
    https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb

    For trainer initiation, followed
    https://huggingface.co/docs/transformers/training

    For finetuning T5 tips
    https://discuss.huggingface.co/t/t5-finetuning-tips/684

    Send model and datasets to GPUs
    https://discuss.huggingface.co/t/sending-a-dataset-or-datasetdict-to-a-gpu/17208/13
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print(f"====== right after load. model on cuda: {next(model.parameters()).device}")

    def preprocess_data(examples):
        inputs = [T5_PREFIX + utt for utt in examples["utt"]]
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_SRC_LEN,
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["ltl"],
                max_length=MAX_TAR_LEN,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True, max_length=MAX_TAR_LEN)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, max_length=MAX_TAR_LEN)
        return metric.compute(predictions=decoded_preds, references=decoded_labels)

    in_seqs_train, out_seqs_train, in_seqs_valid, out_seqs_valid = construct_dataset(data_fpath, end_idx)
    symbolic_dataset = DatasetDict({"train": Dataset.from_dict({"utt": in_seqs_train, "ltl": out_seqs_train}),
                                    "test": Dataset.from_dict({"utt": in_seqs_valid, "ltl": out_seqs_valid})})
    dataset_train_valid = symbolic_dataset["test"].train_test_split(test_size=test_size)
    symbolic_dataset["validation"] = dataset_train_valid["test"]
    print(f"Training set size: {len(in_seqs_train)}, {len(out_seqs_train)}\nValidation set size: {len(symbolic_dataset['validation'])}")
    dataset_tokenized = symbolic_dataset.map(preprocess_data, batched=True)

    train_args = Seq2SeqTrainingArguments(
        output_dir=model_dpath,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=1e-5,
        weight_decay=0.01,
        # fp16=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_strategy="steps",
        logging_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        metric_for_best_model="exact_match",
        load_best_model_at_end=True,
        save_total_limit=3,  # best and last chkpt always saved
        predict_with_generate=True,
        report_to="tensorboard"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    metric = evaluate.load("exact_match")

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(f"====== right after init trainer. model on cuda: {trainer.args.device}")

    trainer.train()

    if model_dpath:
        best_ckpt_dpath = os.path.join(model_dpath, "checkpoint-best")
        os.makedirs(best_ckpt_dpath, exist_ok=True)
        trainer.save_model(best_ckpt_dpath)
        print(f"Saved model best checkpoint at: {best_ckpt_dpath}")
    else:
        trainer.save_model()


def construct_dataset(fpath, end_idx):
    train_iter, _, valid_iter, _ = load_split_dataset(fpath)
    if end_idx:
        train_iter = train_iter[:BASE_DATASET_SIZE + end_idx]  # keep all base dataset, slice composed dataset
    in_seqs_train, out_seqs_train, in_seqs_valid, out_seqs_valid = [], [], [], []
    for utt, ltl in train_iter:
        in_seqs_train.append(utt)
        out_seqs_train.append(ltl)
    for utt, ltl in valid_iter:
        in_seqs_valid.append(utt)
        out_seqs_valid.append(ltl)
    return in_seqs_train, out_seqs_train, in_seqs_valid, out_seqs_valid


def finetune_t5_old(input_sequences, output_sequences, tokenizer, model):
    source_encoding = tokenizer(
        input_sequences,
        padding="longest",
        max_length=MAX_SRC_LEN,
        truncation=True,
        return_tensors="pt",
    )
    input_ids, attention_mask = source_encoding.input_ids, source_encoding.attention_mask

    target_encoding = tokenizer(
        text_target=output_sequences,
        padding="longest",
        max_length=MAX_TAR_LEN,
        truncation=True,
        return_tensors="pt",
    )
    labels = target_encoding.input_ids
    labels[labels == tokenizer.pad_token_id] = -100  # replace padding token id's of the labels by -100 so it's ignored by the loss

    for epoch in range(5):
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        print(f"epoch {epoch}: {loss.item()}")

    input_ids = tokenizer(f"{T5_PREFIX}visit a then b", return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    ltl = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(ltl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", type=str, default="data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_0.pkl", help="train and test sets.")
    parser.add_argument("--end_idx", type=int, default=None, help="slicing index for learning curve.")
    parser.add_argument("--model_dpath", type=str, default=None, help="directory to save model checkpoints.")
    parser.add_argument("--cache_dpath", type=str, default="$HOME/.cache/huggingface", help="huggingface cache.")
    parser.add_argument("--model", type=str, choices=HF_MODELS, help="name of supervised seq2seq model")
    args = parser.parse_args()

    model_dpath = os.path.join(args.model_dpath, args.model)

    print(f"Finetune dataset: {args.data_fpath}[:{BASE_DATASET_SIZE}+{args.end_idx}]")
    print(f"Save model checkpoints at: {model_dpath}")

    if "t5" in args.model:
        finetune_t5(args.model, args.data_fpath, args.end_idx, model_dpath)
    else:
        raise TypeError(f"ERROR: unrecognized model, {args.model}")
    # tensorboard --logdir=model/t5-base/runs

    # input_sequences, output_sequences = construct_dataset(args.data_fpath)
    # finetune_t5(input_sequences, output_sequences, tokenizer, model)
