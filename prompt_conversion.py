"""
For NL2LTL baseline. https://github.com/IBM/nl2ltl
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

from utils import save_to_file
from dataset_symbolic import load_split_dataset

SEEDS = [0, 1, 2, 42, 111]
HEADER = '''
Translate natural language sentences into patterns.

ALLOWED_PATTERNS: visit_1, visit_2, visit_3, visit_4, visit_5, sequenced_visit_2, sequenced_visit_3, sequenced_visit_4, sequenced_visit_5, ordered_visit_2, ordered_visit_3, ordered_visit_4, ordered_visit_5, strictly_ordered_visit_2, strictly_ordered_visit_3, strictly_ordered_visit_4, strictly_ordered_visit_5, patrolling_1, patrolling_2, patrolling_3, patrolling_4, patrolling_5, past_avoidance_2, global_avoidance_1, global_avoidance_2, global_avoidance_3, global_avoidance_4, global_avoidance_5, future_avoidance_2, upper_restricted_avoidance_1, upper_restricted_avoidance_2, upper_restricted_avoidance_3, upper_restricted_avoidance_4, upper_restricted_avoidance_5, lower_restricted_avoidance_1, lower_restricted_avoidance_2, lower_restricted_avoidance_3, lower_restricted_avoidance_4, lower_restricted_avoidance_5, exact_restricted_avoidance_1, exact_restricted_avoidance_2, exact_restricted_avoidance_3, exact_restricted_avoidance_4, exact_restricted_avoidance_5, delayed_reaction_2, prompt_reaction_2, bound_delay_2, wait_2
ALLOWED_SYMBOLS: a, b, c, d, h

'''


def generate_prompts_from_split_dataset(split_fpath, prompt_dpath, nexamples, seed):
    """
    :param split_fpath: path to pickle file containing train, test split for a holdout type
    :param nexamples: number of examples for 1 formula
    :return:
    """
    train_iter, train_meta, _, _ = load_split_dataset(split_fpath)

    meta2data = defaultdict(list)
    for idx, ((utt, ltl), (pattern_type, props)) in enumerate(zip(train_iter, train_meta)):
        meta2data[(pattern_type, len(props))].append(((utt, ltl), props))
    sorted(meta2data.items(), key=lambda kv: kv[0])

    prompt = HEADER
    for (pattern_type, nprop), data in meta2data.items():
        random.seed(seed)
        examples = random.sample(data, nexamples)
        for (utt, ltl), props in examples:
            prompt += f"NL: {utt}\nPATTERN: {pattern_type}_{len(props)}\nSYMBOLS: {', '.join(props)}\n\n"
            # print(f"{pattern_type} | {nprop}\n{utt}\n{ltl}\n")
        # breakpoint()
    prompt += "NL:"

    split_dataset_name = Path(split_fpath).stem
    prompt_fpath = f"{prompt_dpath}/nl2ltl_prompt_nexamples{nexamples}_{split_dataset_name}.txt"
    save_to_file(prompt, prompt_fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_fpath', type=str, default='data/prompt_nl2ltl')
    parser.add_argument('--split_fpath', type=str, default='data/holdout_split_batch12_perm/symbolic_batch12_perm_utt_0.2')
    args = parser.parse_args()

    for seed in SEEDS:
        split_fpath = f'{args.split_fpath}_{seed}.pkl'
        generate_prompts_from_split_dataset(split_fpath, args.prompt_fpath, 1, seed)
