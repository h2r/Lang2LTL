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

def get_embedding(in_text: str, engine="davinci") -> list[float]:
    engine = "text-similarity-{}-001".format(engine)
    in_text = in_text.replace("\n", " ")  # replace newlines, which can negatively affect performance

    embedding = openai.Embedding.create(
        input=[in_text],
        engine=engine  # change for different embedding dimension
    )["data"][0]["embedding"]
    return embedding

def read_file(filepath):
    with open(filepath, 'r') as f:
        names = [ line[:-1] for line in f.readlines()]
    return names

def store_embeds(names, filepath, engine):
    dic = {}
    for name in names:
        embed = get_embedding(name, engine)
        dic[name] = embed

    with open(filepath,'w') as f:
        json.dump(dic,f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-landmark_path', type=str, default='data/landmarks.txt', help='path to landmark file for embedding')
    parser.add_argument('-embed_path', type=str, default='name2embed.json', help='path to save name2embed file')
    parser.add_argument('-engine',type=str,default = 'davinci', choices=['ada','babbage','curie','davinci'])
    
    args = parser.parse_args()
    
    names= read_file(args.landmark_path)
    store_embeds(names, args.embed_path, args.engine)
    print('embed file stored at {} with {} engine'.format(args.embed_path, args.engine))
