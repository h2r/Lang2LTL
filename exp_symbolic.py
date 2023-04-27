"""
Evaluate different model for symbolic translation.
"""
import os
import logging
from pathlib import Path
import argparse
from pprint import pprint
import random
from collections import defaultdict

from gpt import GPT3, GPT4
from eval import aggregate_results, evaluate_lang_from_file
from utils import load_from_file, save_to_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_fpath", type=str, default="data/holdout_split_batch12_perm/symbolic_batch12_perm_ltl_type_3_42_fold4.pkl", help="path to pkl file storing train set")
    parser.add_argument("--test_dataset_fpath", type=str, default="data/holdout_split_batch12_perm/symbolic_batch12_perm_ltl_type_3_42_fold4.pkl", help="path to pkl file storing test set")
    parser.add_argument("--analysis_fpath", type=str, default="data/analysis_symbolic_batch12_perm.csv", help="path to dataset analysis")
    parser.add_argument("--model", type=str, default="gpt-4", choices=["gpt3_finetuned_symbolic_batch12_perm_utt_0.2_111", "gpt-4", "text-davinci-003"], help="name of model to be evaluated")
    parser.add_argument("--nexamples", type=int, default=3, help="number of examples per instance in prompt for GPT")
    parser.add_argument("--rand_eval_samples", type=int, default=100, help="number of random evaluation samples per formula")
    parser.add_argument("--seed_eval_samples", type=int, default=42, help="seed for randomly sampling evaluation samples")
    parser.add_argument("--aggregate", action="store_true", help="whether to aggregate results or compute new results.")
    parser.add_argument("--aggregate_dpath", type=str, default="results/pretrained_gpt4/type_holdout_batch12_perm", help="dpath to results file to aggregate")
    args = parser.parse_args()
    dataset_name = Path(args.train_dataset_fpath).stem

    if args.aggregate:  # aggregate acc-per-formula result files
        result_dpath = args.aggregate_dpath
        result_fpaths = [os.path.join(result_dpath, fname) for fname in os.listdir(result_dpath) if "acc" in fname and "csv" in fname and "aggregated" not in fname]
        filter_types = ["fair_visit"]
        accumulated_acc, accumulated_std = aggregate_results(result_fpaths, filter_types)
        print("Please verify results files")
        pprint(result_fpaths)
    else:
        if "gpt" in args.model or "davinci" in args.model:  # gpt for finetuned gpt3 or off-the-shelf gpt-4, text-davinci-003 for off-the-shelf gpt-3
            gpt_model_number = 4 if args.model == "gpt-4" else 3
            dataset = load_from_file(args.train_dataset_fpath)
            test_dataset = load_from_file(args.test_dataset_fpath)
            valid_iter = test_dataset["valid_iter"]
            dataset["valid_meta"] = test_dataset["valid_meta"]
            if "utt" in args.train_dataset_fpath:  # results directory based on holdout type
                dname = f"utt_holdout_batch12_perm"
            elif "formula" in args.train_dataset_fpath:
                dname = f"formula_holdout_batch12_perm"
            elif "type" in args.train_dataset_fpath:
                dname = f"type_holdout_batch12_perm"
            if "finetuned" in args.model:
                engine = load_from_file("model/gpt3_models.pkl")[args.model]
                valid_iter = [(f"Utterance: {utt}\nLTL:", ltl) for utt, ltl in valid_iter]
                result_dpath = os.path.join("results", "finetuned_gpt3", dname)
                os.makedirs(result_dpath, exist_ok=True)
                result_log_fpath = os.path.join(result_dpath, f"log_{args.model}.csv")  # fintuned model name already contains dataset name
                acc_fpath = os.path.join(result_dpath, f"acc_{args.model}.csv")
            else:
                engine = args.model
                prompt_fname = f"prompt_nexamples{args.nexamples}_{dataset_name}.txt"  # prompt corresponds to train split dataset
                prompt_fpath = os.path.join("data", "prompt_symbolic_batch12_perm", prompt_fname)
                prompt = load_from_file(prompt_fpath)
                valid_iter = [(f"{prompt} {utt}\nLTL:", ltl) for utt, ltl in valid_iter]
                result_dpath = os.path.join("results", f"pretrained_gpt{gpt_model_number}", dname)
                os.makedirs(result_dpath, exist_ok=True)
                result_log_fpath = os.path.join(result_dpath, f"log_{args.model}_{Path(prompt_fname).stem}_{args.rand_eval_samples}-eval-samples.csv")
                acc_fpath = os.path.join(result_dpath, f"acc_{args.model}_{Path(prompt_fname).stem}_{args.rand_eval_samples}-eval-samples.csv")
            dataset["valid_iter"] = valid_iter

            # Samples a subset of test set by sampling rand_eval_samples per formula
            if args.rand_eval_samples:
                valid_iter, valid_meta = dataset["valid_iter"], dataset["valid_meta"]
                valid_iter_sampled, valid_meta_sampled = [], []
                meta2data = defaultdict(list)
                for idx, ((utt, ltl), (pattern_type, props)) in enumerate(zip(valid_iter, valid_meta)):
                    meta2data[(pattern_type, len(props))].append((utt, ltl, props))
                sorted(meta2data.items(), key=lambda kv: kv[0])

                for (pattern_type, nprop), data in meta2data.items():
                    random.seed(args.seed_eval_samples)
                    examples = random.sample(data, args.rand_eval_samples)
                    valid_iter_sampled.extend([(utt, ltl) for utt, ltl, _ in examples])
                    valid_meta_sampled.extend([(pattern_type, props) for _, _, props in examples])

                dataset["valid_iter"], dataset["valid_meta"] = valid_iter_sampled, valid_meta_sampled
                # dataset["valid_iter"], dataset["valid_meta"] = zip(*random.sample(list(zip(valid_iter, valid_meta)), args.rand_samples))


                print(f"test set size: {len(dataset['valid_iter'])}, {len(dataset['valid_meta'])}\n{meta2data.keys()}")
                breakpoint()


            split_dname = os.path.join("data", f"eval_gpt{gpt_model_number}")
            os.makedirs(split_dname, exist_ok=True)
            split_dataset_fpath = os.path.join(split_dname, f"{dataset_name}_{args.rand_eval_samples}-eval-samples.pkl")
            save_to_file(dataset, split_dataset_fpath)
            model = GPT4(engine, temp=0, max_tokens=128) if gpt_model_number == 4 else GPT3(engine, temp=0, max_tokens=128)
        else:
            raise ValueError(f"ERROR: model not recognized: {args.model}")

        logging.basicConfig(level=logging.DEBUG,
                            format='%(message)s',
                            handlers=[
                                logging.FileHandler(f'{os.path.splitext(result_log_fpath)[0]}.log', mode='w'),
                                logging.StreamHandler()
                            ]
        )
        logging.info(f"test set size: {len(dataset['valid_iter'])}, {len(dataset['valid_meta'])}\n{meta2data.keys()}\n")

        evaluate_lang_from_file(model, split_dataset_fpath, args.analysis_fpath, result_log_fpath, acc_fpath, batch_size=1)
