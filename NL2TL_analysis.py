#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:46:09 2023

@author: ajshah
"""

import json
import os
import re
import spot
import numpy as np
import tqdm
from multiprocessing.pool import Pool


SPOT_OPS = ['G','F','X','U','i','e','|','&','!','^']
PROPS = ['a','b','c','d','h','j','k']

TRAIN_DATA_PATH = os.path.join('/home/ajshah','Github/NL2TL-main/Data_total64220_03_26','combine_train_seq2tree_idea4.jsonl')
TEST_DATA_PATH = os.path.join('/home/ajshah','Github/NL2TL-main/Data_total64220_03_26','combine_test_seq2tree_idea4.jsonl')


def check_leakage_serial(args):
    unique_formulas_str = args[0]
    str_f = args[1]
    unique_formulas = [spot.formula(u) for u in unique_formulas_str]
    f = spot.formula(str_f)
    return np.any([spot.are_equivalent(u,f) for u in unique_formulas_str])

def check_test_leakage_v2(unique_train_formulas, test_formulas):
    
    common_formulas = []
    new_formulas = []
    
    unique_formulas_str = [str(u) for u in unique_train_formulas]
    test_formulas_str = [str(f) for f in test_formulas]
    
    with Pool(processes = 16) as pool:
        common_flags = pool.imap(check_leakage_serial, zip([unique_formulas_str]*len(test_formulas), test_formulas_str))
    
    
        for (i,flag) in tqdm.tqdm(enumerate(common_flags), total = len(test_formulas)):
            if flag:
                common_formulas.append(test_formulas[i])
            else:
                new_formulas.append(test_formulas[i])
    
        n_common = np.sum(common_flags)
    
    return n_common, common_formulas, new_formulas
    

def get_test_leakage(unique_train_formulas, test_formulas):
    common_formulas = []
    new_formulas = []
    for f in tqdm.tqdm(test_formulas):
        if check_existence_formula_set(unique_train_formulas, f):
            common_formulas.append(f)
        else:
            new_formulas.append(f)
    n_common = len(common_formulas)
    return n_common, common_formulas, new_formulas

def process_formulas(raw_formulas):
    #remove all time references that are inside []
    raw_formulas = [re.sub('\[.*?\]', '', s) for s in raw_formulas]
    #replace keywords with spot operators 'globally','finally','imply','equal','and','or', 'until', negation
    formulas = [s.replace('globally','G').replace('finally','F').replace('imply','->').replace('equal','<->').replace('and','&').replace('or','|').replace('until','U').replace('negation','!') for s in raw_formulas]
    failed_formulas = []
    spot_formulas = []
    for (i,f) in enumerate(formulas):
        try:
            new_form = spot.formula(f)
            spot_formulas.append(new_form)
        except:
            failed_formulas.append((i,f))
                
    formulas = [spot.formula(f) for f in spot_formulas]
    formulas = [replace_props(f) for f in formulas]
    return formulas, failed_formulas

def replace_props(formula):
    #parse formula in prefix format, repeat the formula but with the propositions replaced by a standard order
    
    props_used = 0
    prop_map = {}
    
    prefix_formula_as_list = formula.__format__('l').split(' ')
    
    output_formula_as_list = []
    for (i,p) in enumerate(prefix_formula_as_list):
        if p not in SPOT_OPS:
            #it is a propositions
            if p not in prop_map:
                prop_map[p] = PROPS[props_used]
                props_used = props_used + 1
                output_formula_as_list.append(prop_map[p])
            else:
                output_formula_as_list.append(prop_map[p])
        else:
            output_formula_as_list.append(p)
    output_formula = ' '.join(output_formula_as_list)
    return spot.formula(output_formula)

def get_unique_formulas(processed_formulas):
    unique_formulas = []
    variants = dict()
    
    for f in tqdm.tqdm(processed_formulas):
        if len(unique_formulas) > 0:
            #Check equivalence with each formula and add to list if not eq
            flag = False
            for u in unique_formulas:
                if spot.are_equivalent(u,f):
                    variants[str(u)].append(f)
                    flag = True
                    break
            if not flag:
                unique_formulas.append(f)
                variants[str(f)] = []
        else:
            unique_formulas.append(f)
            variants[str(f)] = []
    return unique_formulas, variants


def get_train_ltl():
    with open(TRAIN_DATA_PATH, 'r') as file:
        json_list = list(file)
    raw_formulas = [json.loads(s)['logic_ltl_true_natural_order'] for s in json_list]
    formulas, failed_formulas = process_formulas(raw_formulas)
    return formulas, failed_formulas

def get_test_ltl():
    with open(TEST_DATA_PATH, 'r') as file:
        json_list = list(file)
    raw_formulas = [json.loads(s)['logic_ltl_true_natural_order'] for s in json_list]
    formulas, failed_formulas = process_formulas(raw_formulas)
    return formulas, failed_formulas

if __name__ == '__main__':
    train_formulas, failed_train_formulas = get_train_ltl()
    test_formulas, failed_test_formulas = get_test_ltl()
    unique_train_formulas, variants = get_unique_formulas(train_formulas)
    
    n_common, common_formulas, novel_formulas = get_test_leakage(unique_train_formulas, test_formulas)
    
    