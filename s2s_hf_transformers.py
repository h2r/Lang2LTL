"""
Finetune pre-trained transformer models from Hugging Face.
"""
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainer, Adafactor, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate

from utils import load_from_file

T5_MODELS = ["t5-small", "t5-base"]
T5_PREFIX = "translate English to Linear Temporal Logic: "
MAX_SRC_LEN = 512
MAX_TAR_LEN = 128
BATCH_SIZE = 8


def finetune_t5(model_name, tokenizer, fpath, valid_size=0.2, test_size=0.1):
    """
    Followed most of the tutorial at
    https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887

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
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    symbolic_dataset = load_dataset("csv", data_files=fpath)
    dataset_train_test = symbolic_dataset["train"].train_test_split(test_size=test_size)
    dataset_train_valid = dataset_train_test["train"].train_test_split(test_size=valid_size)
    symbolic_dataset["train"] = dataset_train_valid["train"]
    symbolic_dataset["validation"] = dataset_train_valid["test"]
    symbolic_dataset["test"] = dataset_train_test["test"]

    dataset_tokenized = symbolic_dataset.map(preprocess_data, batched=True)

    train_args = Seq2SeqTrainingArguments(
        output_dir=f"model/{model_name}",
        evaluation_strategy="steps",
        eval_steps=20,
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="steps",
        save_steps=40,
        learning_rate=1e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        num_train_epochs=1,
        fp16=True,
        # predict_with_generate=True,
        # metric_for_best_model="rouge1",
        load_best_model_at_end=True,
        save_total_limit=3,
        report_to="tensorboard"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

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


def construct_dataset(fpath, valid_size=0.2, test_size=0.1):
    data = load_from_file(fpath)
    input_sequences, output_sequences = [], []
    for utt, ltl in data:
        input_sequences.append(f"{T5_PREFIX}{utt}")
        output_sequences.append(ltl)
    return input_sequences, output_sequences


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
    parser.add_argument('--data', type=str, default='data/symbolic_pairs_no_perm.csv', help='file path to train and test data for supervised seq2seq')
    parser.add_argument('--model', type=str, default='t5-small', choices=["t5-small", "bart"], help='name of supervised seq2seq model')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    if args.model in T5_MODELS:
        finetune_t5(args.model, tokenizer, args.data)

    # input_sequences, output_sequences = construct_dataset(args.data)

    # if args.model in T5_MODELS:
    #     finetune_t5(input_sequences, output_sequences, tokenizer, model)
