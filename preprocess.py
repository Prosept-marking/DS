from copy import copy
import os
import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data():
    # серверный путь
    pth1 = '/datasets/'
    # локальный путь
    pth2 = 'C:\\Dev\\prosept\\datasets\\'

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


def count_tfidf(data):
    count_tf_idf = TfidfVectorizer()
    count_tf_idf.fit_transform(data)
    return count_tf_idf


def prepare_data():
    df_product = load_data()
    # main base
    data_product = df_product[['id', 'name_1c', 'recommended_price']].copy()
    data_product = data_product.loc[data_product['name_1c'].notna()].reset_index()
    data_product = data_product.drop(['index'], axis=1)
    # normalizing name product in main base
    data_product['clear_name'] = data_product['name_1c'].apply(clear_text)
    return data_product
    

def fit_tfidf():
    # load dataset
    data_product = prepare_data()
    # fit tf_idf
    fitted_tf_idf = count_tfidf(data_product['clear_name'])
    return fitted_tf_idf


def prepare_query(query):
    # normalizing query name product
    clear_name_product = clear_text(query)
    tfidf = fit_tfidf()
    # calculating tfidf
    query_tfidf = tfidf.transform([clear_name_product])
    return query_tfidf


def load_model():
    with open('prosept\model\knn_model.pkl', 'rb') as file:
        clf = pickle.load(file)
    return clf


def matching(query):
    # load clf
    clf = load_model()
    # calc query tfidf
    query_tfidf = prepare_query(query)
    # compute 5 neighbors for query
    neighbors = clf.kneighbors(query_tfidf, n_neighbors=5, return_distance=False)
    data_product = prepare_data()
    result_matching = []
    # listing 5 neighbors
    for i in range(len(neighbors)):
        for j in neighbors[i]:
            result_matching.append(data_product.iloc[j]['id'])

    return result_matching
