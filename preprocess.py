from copy import copy
import os
import pickle
import re

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def load_data():
    # серверный путь
    pth1 = '/datasets/'
    # локальный путь
    pth2 = 'D:\\Dev\\prosept\\datasets\\'

    try:
        if os.path.exists(pth1):
            df_product = pd.read_csv(
                pth1+'marketing_product.csv',
                sep=';',
                header=0,
                on_bad_lines='skip'
            )
        elif os.path.exists(pth2):
            df_product = pd.read_csv(
                pth2+'marketing_product.csv',
                sep=';',
                index_col=0,
                # header=0,
                on_bad_lines='skip'
            )
    except FileNotFoundError:
            print('Path does not exist. Check path')
    
    return df_product


def clear_text(text):
    text = re.sub(r'"{2}|[/]', ' ', text) # убирает 2 подряд кавычки (") + убирает слэш
    text = re.sub(r'(?<=[а-я])[A-Z]', ' \g<0>', text) # разделяет пробелом слип. англ. и русское слова
    text = re.sub(r'[A-Za-z](?=[а-я])', '\g<0> ', text) # разделяет пробелом слип. англ. и русское слова
    text = re.sub(r'(?<=[А-Я]{2})[а-я]', ' \g<0>', text) # разделяет пробелом слип. русские слова
    text = re.sub(r'\W', ' ', text) # убирает знаки препинания
    text = re.sub(r'\d', '', text) # убирает цифры
    text = text.lower() # в нижний регистр
    return ' '.join(text.split())


def get_encoder():
    encoder = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
    return encoder


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
    return tokenizer


def prepare_query(query):
    # normilizing query's header
    query_header = clear_text(query)
    # get tokenizer
    tokenizer = get_tokenizer()
    # get encoder
    encoder = get_encoder()
    # compute tokens
    encoded_input = tokenizer(query_header, padding=True, truncation=True, max_length=64, return_tensors='pt')
    # calc query's embedding
    with torch.no_grad():
        encoded_output = encoder(**encoded_input)
    query_embed = encoded_output.pooler_output
    query_embed = torch.nn.functional.normalize(query_embed)
    query_embed = query_embed.reshape(1, -1)
    return query_embed


def get_model():
    with open('D:\\Dev\\prosept\\model\\nn_model.pkl', 'rb') as file:
        clf = pickle.load(file)
    return clf


def matching(query, n_neighbors=5):
    # get classificator
    clf = get_model()
    # get query's embedding
    query_embedding = prepare_query(query)
    # compute 5 neighbors for query
    neighbors = clf.kneighbors(query_embedding, n_neighbors=n_neighbors, return_distance=False)
    df_product = load_data()
    
    # listing 5 neighbors
    for i in range(len(neighbors)):
        result_matching = []
        for j in neighbors[i]:
            result_matching.append(df_product.iloc[j]['id'])

        return result_matching
