#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 00:53:54 2022

@author: ziyi
"""

import argparse
import json
import openai
import os
from gpt3 import GPT3

openai.api_key = os.getenv("OPENAI_API_KEY")

def read_file(filepath):
    with open(filepath, 'r') as f:
        names = [ line[:-1] for line in f.readlines()]
    return names

def store_embeds(names, filepath, model, engine):
    dic = {}
    
    if args.model == 'gpt3':
        ground_module = GPT3()
    # elif args.ground == 'bert':
    #     ground_module = BERT()
    else:
        raise ValueError("ERROR: grounding module not recognized")
        
    for name in names:
        dic[name] = ground_module.get_embedding(name, engine)

    with open(filepath, 'w') as f:
        json.dump(dic, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-landmark_path', type=str, default='data/landmarks.txt', help='path to landmark file for embedding')
    parser.add_argument('-embed_path', type=str, default='data/name2embed.json', help='path to save name2embed file')
    parser.add_argument('-engine',type=str,default = 'davinci', choices=['ada','babbage','curie','davinci'])
    parser.add_argument('-model',type=str,default = 'gpt3', choices=['gpt3', 'bert'])
    
    args = parser.parse_args()
    
    names= read_file(args.landmark_path)
    store_embeds(names, args.embed_path, args.model, args.engine)
    print('embed file stored at {} with {} model'.format(args.embed_path, args.model))
