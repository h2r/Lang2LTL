# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:44:47 2023

@author: AJShah
"""
import os
import pandas as pd
from fnmatch import fnmatch


#PARSE_FUNCS = {'utt_holdout': parse_results_utt, 'formula_holdout': parse_results_formula, 'type_holdout': parse_results_type}
SYMBOLIC_MODEL_TYPES = ['finetuned_gpt3', 'pretrained_gpt3','s2s_pt_transformer']
SYMBOLIC_TEST_TYPES = ['utt_holdout','formula_holdout','type_holdout']
MODEL_NAMES = {'finetuned_gpt3': 'Finetuned GPT3', 'pretrained_gpt3': 'Prompt GPT3', 's2s_pt_transformer': 'Seq2Seq'}
TEST_NAMES = {'utt_holdout': 'Utterances','formula_holdout':'Formula','type_holdout':'Type'}


def create_symbolic_accuracies_table(test_types = SYMBOLIC_TEST_TYPES, model_types = SYMBOLIC_MODEL_TYPES):
    entries = []
    for test_type in test_types:
        for model_type in model_types:
            entries.extend(parse_symbolic_accuracies(test_type, model_type))
    #Conver to dataframe
    df = {}
    for (i, entry) in enumerate(entries):
        df[i] = entry
    df = pd.DataFrame.from_dict(df, orient = 'index')
    return df
        

def parse_symbolic_accuracies(test_type, model_type):
    filenames = resolve_filenames(test_type, model_type)
    row_dicts = []
    for (i,file) in enumerate(filenames):
        #parse_func = get_parse_funcs(test_type)
        acc = parse_acc(pd.read_csv(file))
        entry = {}
        entry['Model'] = MODEL_NAMES[model_type]
        entry['Test Type'] = TEST_NAMES[test_type]
        entry['Model ID'] = i
        entry['Accuracy'] = acc
        row_dicts.append(entry)
    return row_dicts


def resolve_filenames(test_type, model_type):
    #returns a list of csv filepaths that need to be parsed by the appropriate 
    filepath = 'results'
    if model_type in SYMBOLIC_MODEL_TYPES:
        filepath = os.path.join(filepath, model_type)
    else:
        raise Exception('Unknown model type')
    if test_type in SYMBOLIC_TEST_TYPES:
        filepath = os.path.join(filepath, test_type+'_batch12_perm')
    else:
        raise Exception('Unknown  test type')
    # print(os.listdir(filepath))
    filenames = [file for file in os.listdir(filepath) if fnmatch(file,'*.csv') and 'aggregated' not in file and 'acc' in file]
    # print(filenames)
    filepaths = [os.path.join(filepath, file) for file in filenames]
    return filepaths

def read_csv_data(filepath):
    return pandas.read_csv(filepath)


def parse_acc(csv_data):
    csv_data = csv_data.loc[csv_data['Accuracy']!='no valid data']
    num = pd.np.sum(csv_data['Number of Utterances']*csv_data['Accuracy'].astype(float))
    den = pd.np.sum(csv_data['Number of Utterances'])
    return num/den


if __name__ == '__main__':
    df = create_symbolic_accuracies_table(test_types = SYMBOLIC_TEST_TYPES, model_types = ['finetuned_gpt3','s2s_pt_transformer'])
    