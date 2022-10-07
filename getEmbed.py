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

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding_gpt3(in_text: str, engine="davinci") -> list[float]:
    engine = "text-similarity-{}-001".format(engine)
    in_text = in_text.replace("\n", " ")  # replace newlines, which can negatively affect performance

    embedding = openai.Embedding.create(
        input=[in_text],
        engine=engine  # change for different embedding dimension
    )["data"][0]["embedding"]
    return embedding

def get_embedding_bert(in_text: str):
    raise NotImplementedError("bert model isn't finished yet")

def read_file(filepath):
    with open(filepath, 'r') as f:
        names = [ line[:-1] for line in f.readlines()]
    return names

def store_embeds(names, filepath, model, engine):
    dic = {}
    for name in names:
        if model == 'gpt3':
            embed = get_embedding_gpt3(name, engine)
        elif model == 'bert':
            embed = get_embedding_bert(name)
        dic[name] = embed

    with open(filepath,'w') as f:
        json.dump(dic,f)
        
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
