"""
Finetune pre-trained transformer models from Hugging Face.
"""
import argparse
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainer, Adafactor, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate

from utils import load_from_file
from dataset_symbolic import load_split_dataset

T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "facebook/bart-base"]
T5_PREFIX = "translate English to Linear Temporal Logic: "
MAX_SRC_LEN = 512
MAX_TAR_LEN = 256
BATCH_SIZE = 50


def finetune_t5(model_name, tokenizer, fpath, valid_size=0.2, test_size=0.1):
    """
    Followed most of the tutorial at
    https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887

    Finetune T5 for translation example
    https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb

    For trainer initiation, followed
    https://huggingface.co/docs/transformers/training

    For finetuning T5 tips
    https://discuss.huggingface.co/t/t5-finetuning-tips/684
    """
    def preprocess_data(examples):
        inputs = [T5_PREFIX + utt for utt in examples["utterance"]]
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_SRC_LEN,
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["ltl_formula"],
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

    input_seqs, output_seqs, input_seqs_valid, output_seqs_valid = construct_dataset(fpath)
    symbolic_dataset = DatasetDict({'train': Dataset.from_dict({'utterance': input_seqs, 'ltl_formula': output_seqs}),
                                    'test': Dataset.from_dict({'utterance': input_seqs_valid, 'ltl_formula': output_seqs_valid})})
    dataset_train_valid = symbolic_dataset["test"].train_test_split(test_size=test_size)
    symbolic_dataset["validation"] = dataset_train_valid["test"]
    dataset_tokenized = symbolic_dataset.map(preprocess_data, batched=True)
    train_args = Seq2SeqTrainingArguments(
        output_dir=f"model/{model_name}",
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=1e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        num_train_epochs=10,
        # fp16=True,
        predict_with_generate=True,
        metric_for_best_model="exact_match",
        load_best_model_at_end=True,
        save_total_limit=3,
        overwrite_output_dir=True,
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

    trainer.train()
    trainer.save_model()


def construct_dataset(fpath):
    train_iter, _, valid_iter, _ = load_split_dataset(fpath)
    input_sequences, output_sequences = [], []
    input_sequences_valid, output_sequences_valid = [], []
    for utt, ltl in train_iter:
        input_sequences.append(utt)
        output_sequences.append(ltl)
    for utt, ltl in valid_iter:
        input_sequences_valid.append(utt)
        output_sequences_valid.append(ltl)
    return input_sequences, output_sequences, input_sequences_valid, output_sequences_valid


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
    parser.add_argument('--data', type=str, default='data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2_0.pkl', help='file path to train and test data for supervised seq2seq')
    parser.add_argument('--model', type=str, default='t5-base', choices=["t5-small", "t5-base", "t5-large", "t5-3b", "facebook/bart-base"], help='name of supervised seq2seq model')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # config = AutoConfig.from_pretrained(args.model) # load from config lead to much worse performance
    # config.max_lengtgh = 50
    # model = AutoModelForSeq2SeqLM.from_config(config)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    if args.model in T5_MODELS:
        finetune_t5(args.model, tokenizer, args.data)
    else:
        raise TypeError(f"ERROR: unrecognized model, {args.model}")
    # tensorboard --logdir=model/runs

    # input_sequences, output_sequences = construct_dataset(args.data)

    # if args.model in T5_MODELS:
    #     finetune_t5(input_sequences, output_sequences, tokenizer, model)
