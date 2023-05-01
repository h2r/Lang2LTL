# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:44:47 2023

@author: AJShah
"""
import os
import pandas as pd
from fnmatch import fnmatch
import json
import math
import numpy as np


#PARSE_FUNCS = {'utt_holdout': parse_results_utt, 'formula_holdout': parse_results_formula, 'type_holdout': parse_results_type}
SYMBOLIC_MODEL_TYPES = ['finetuned_gpt3', 'pretrained_gpt3','s2s_pt_transformer']
SYMBOLIC_TEST_TYPES = ['utt_holdout','formula_holdout','type_holdout']
MODEL_NAMES = {'finetuned_gpt3': 'Finetuned GPT3', 'pretrained_gpt3': 'Prompt GPT3', 's2s_pt_transformer': 'Seq2Seq'}
TEST_NAMES = {'utt_holdout': 'Utterance','formula_holdout':'Formula','type_holdout':'Type'}

OSM_MODEL_NAMES = ['Lang2LTL', 'CopyNet']

def get_full_system_piechart(test_type):
    filepath = os.path.join('.','results','lang2ltl','osm')
    cities = get_osm_cities()
    symbolic_errors = 0
    RER_errors = 0
    
    for city in cities:
        city_accs = []
        filepath1 = os.path.join(filepath, city, test_type+'_batch12')
        if os.path.exists(filepath1):
            files = [file for file in os.listdir(filepath1) if fnmatch(file, 'acc*.csv')]
            for file in files:
                df = pd.read_csv(os.path.join(filepath1,file))
                if 'False' in df['Accuracy'].value_counts().index:
                    symbolic_errors = symbolic_errors + df['Accuracy'].value_counts()['False']
                if 'RER or Grounding Error' in df['Accuracy'].value_counts().index:
                    RER_errors = RER_errors + df['Accuracy'].value_counts()['RER or Grounding Error']
    return symbolic_errors, RER_errors
            

def parse_per_city_accs(test_type):
    filepath = os.path.join('.','results','lang2ltl','osm')
    cities = get_osm_cities()
    accs = []
    df = {}
    entry_id = 0
    
    
    for city in cities:
        city_accs = []
        filepath1 = os.path.join(filepath, city, test_type+'_batch12')
        if os.path.exists(filepath1):
            files = [file for file in os.listdir(filepath1) if fnmatch(file, 'acc*.json')]
            for file in files:
                with open(os.path.join(filepath1, file), 'r') as f:
                    data = json.load(f)
                acc = data['Accumulated Accuracy']
                city_accs.append(acc)
        if len(city_accs):
            accs.append(city_accs)
        for (model_id, acc) in enumerate(city_accs):
            entry = {}
            entry['Model ID'] = model_id
            entry['Accuracy'] = acc
            entry['City'] = city
            df[entry_id] = entry
            entry_id = entry_id + 1
        
    return pd.DataFrame.from_dict(df, orient = 'index')
    
    
    
    
        

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


def create_osm_accuracies_table(test_types = SYMBOLIC_TEST_TYPES, model_types = OSM_MODEL_NAMES):
    entries = []
    for test_type in test_types:
        for model_type in model_types:
            entries.extend(parse_osm_acc(test_type, model_type))
    df = {}
    for (i,entry) in enumerate(entries):
        df[i] = entry
    df = pd.DataFrame.from_dict(df, orient = 'index')
    return df
            

            
def parse_osm_acc(test_type, model_type):
    #returns a list of dicts
    if model_type == 'Lang2LTL':
        accs = get_lang2ltl_accuracies(test_type)
    else:
        accs = get_copynet_accuracies(test_type)
    entries = []
    for (i,acc) in enumerate(accs):
        entry = {}
        entry['Test Type'] = TEST_NAMES[test_type]
        entry['Model'] = model_type
        entry['Model ID'] = i
        entry['Accuracy'] = acc
        entries.append(entry)
    return entries
            
def get_copynet_accuracies(test_type):
    test_strings = {'utt_holdout':'utt','formula_holdout':'formula','type_holdout':'type'}
    test_string = test_strings[test_type]
    
    filepath = os.path.join('.','results','CopyNet')
    relevant_dirs = [os.path.join(filepath,dirname) for dirname in os.listdir(filepath) if test_string in dirname and os.path.isdir(os.path.join(filepath,dirname))]
    accs = []
    
    for dirname in relevant_dirs:
        #print(dirname)
        filenames = [file for file in os.listdir(dirname) if 'aggregate' in file]
        #print(filenames)
        if filenames:
            filename = os.path.join(dirname, filenames[0])
            csv_data = pd.read_csv(filename)
            acc = pd.np.mean(csv_data['accuracy'])
            accs.append(acc)
    return accs

def get_lang2ltl_accuracies(test_type, output='means'):
    filepath = os.path.join('.','results','lang2ltl','osm')
    cities = get_osm_cities()
    accs = []
    
    for city in cities:
        city_accs = []
        filepath1 = os.path.join(filepath, city, test_type+'_batch12')
        if os.path.exists(filepath1):
            files = [file for file in os.listdir(filepath1) if fnmatch(file, 'acc*.json')]
            for file in files:
                with open(os.path.join(filepath1, file), 'r') as f:
                    data = json.load(f)
                acc = data['Accumulated Accuracy']
                city_accs.append(acc)
        if len(city_accs):
            accs.append(city_accs)
    maxlen = max([len(acc) for acc in accs])
    #replace all missing values with nan
    for (i,acc) in enumerate(accs):
        if len(acc) < maxlen:
            acc = acc + [math.nan]*(maxlen - len(acc))
            accs[i] = acc
    accs = np.array(accs)
    if output == 'means':
        return np.nanmean(accs, axis=0)
    else:
        return accs

def parse_n_prop_accuracies(test_type, model_type):
    filenames = resolve_filenames(test_type, model_type)
    dataframes = []
    
    for (i,file) in enumerate(filenames):
        csv_data = pd.read_csv(file)
        dataframes.append(get_n_prop_accuracies(csv_data, i, test_type))
    
    return pd.concat(dataframes, axis=0, ignore_index=True)

def get_n_prop_accuracies(csv_data, model_id, test_type):
    #Assumes csv data provided as pandas file
    #Get a pandas table per file and then merge it
    row_dicts = []
    #csv_data = csv_data.loc[csv_data['Accuracy'] != 'no valid data']
    formula_types = np.unique(csv_data['Number of Propositions'])
    for formula_type in formula_types:
        #print(formula_type)
        data = csv_data.loc[csv_data['Number of Propositions'] == formula_type]
        data = data.loc[data['Accuracy'] != 'no valid data']
        #print(len(data))
        if len(data) > 0:
            acc = np.sum(data['Number of Utterances'] * data['Accuracy'].astype(float))/np.sum(data['Number of Utterances'])
            entry = {}
            entry['Model ID'] = model_id
            entry['Test Type'] = test_type
            entry['N Propositions'] = formula_type
            entry['Accuracy'] = acc
            row_dicts.append(entry)
            #print(entry)
    df = {}        
    for (i,entry) in enumerate(row_dicts):
        df[i] = entry
    df = pd.DataFrame.from_dict(df, orient='index')
    return df
        
        
def parse_per_type_accuracies(test_type, model_type):
    filenames = resolve_filenames(test_type, model_type)
    dataframes = []
    
    for (i,file) in enumerate(filenames):
        csv_data = pd.read_csv(file)
        dataframes.append(get_per_type_accuracies(csv_data, i, test_type))
    
    return pd.concat(dataframes, axis=0, ignore_index=True)

def get_per_type_accuracies(csv_data, model_id, test_type):
    #Assumes csv data provided as pandas file
    #Get a pandas table per file and then merge it
    row_dicts = []
    #csv_data = csv_data.loc[csv_data['Accuracy'] != 'no valid data']
    formula_types = np.unique(csv_data['LTL Type'])
    for formula_type in formula_types:
        #print(formula_type)
        data = csv_data.loc[csv_data['LTL Type'] == formula_type]
        data = data.loc[data['Accuracy'] != 'no valid data']
        #print(len(data))
        if len(data) > 0:
            acc = np.sum(data['Number of Utterances'] * data['Accuracy'].astype(float))/np.sum(data['Number of Utterances'])
            entry = {}
            entry['Model ID'] = model_id
            entry['Test Type'] = test_type
            entry['Formula Type'] = formula_type
            entry['Accuracy'] = acc
            row_dicts.append(entry)
            #print(entry)
    df = {}        
    for (i,entry) in enumerate(row_dicts):
        df[i] = entry
    df = pd.DataFrame.from_dict(df, orient='index')
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
    #print(model_type)
    if model_type in SYMBOLIC_MODEL_TYPES:
        filepath = os.path.join(filepath, model_type)
    else:
        raise Exception('Unknown model type')
    if test_type in SYMBOLIC_TEST_TYPES:
        filepath = os.path.join(filepath, test_type+'_batch12_perm')
    else:
        raise Exception('Unknown  test type')
    # print(os.listdir(filepath))
    filenames = [file for file in os.listdir(filepath) if fnmatch(file,'*.csv') and 'aggregated' not in file and 'acc' in file and 'num_prop' not in file and 'formula_type' not in file]
    # print(filenames)
    filepaths = [os.path.join(filepath, file) for file in filenames]
    return filepaths

def read_csv_data(filepath):
    return pd.read_csv(filepath)


def parse_acc(csv_data):
    csv_data = csv_data.loc[csv_data['Accuracy']!='no valid data']
    num = pd.np.sum(csv_data['Number of Utterances']*csv_data['Accuracy'].astype(float))
    den = pd.np.sum(csv_data['Number of Utterances'])
    return num/den

def get_osm_cities():
    filepath = os.path.join('.','results','lang2ltl','osm')
    cities = os.listdir(filepath)
    cities = [city for city in cities if 'boston' not in city]
    return cities


if __name__ == '__main__':
    #df = create_symbolic_accuracies_table(test_types = SYMBOLIC_TEST_TYPES, model_types = ['finetuned_gpt3','s2s_pt_transformer'])
    a=1    