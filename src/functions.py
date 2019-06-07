import csv
import numpy as np
import pandas as pd
from metrics import *
import json
import random
from config import news_meta_path
from tqdm import tqdm
from annoy import AnnoyIndex

def cos_sim(a, b):
    """
    Takes 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def load_news_meta(path=news_meta_path):
    metas = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            meta = json.loads(line)
            metas[meta['doc_id']] = (meta['title'],meta['cat_lv1'],meta['cat_lv2'],meta['publish_datetime'])
    return metas

def cos_sim_items(target, items):
    return [cos_sim(target, item) for item in items]

def get_ranking_list_from_vectors(target_vec, candidate_pool, items=20):
    c_ids, c_vecs = zip(*candidate_pool.items())
    c_scores = cos_sim_items(target_vec, c_vecs)
    c_scores, c_ids = zip(*sorted(zip(c_scores, c_ids),reverse=True))
    ranking_list = list(c_ids)[:items]
    ranking_scores = list(c_scores)[:items]
    score_list = np.asarray(ranking_scores)
    min_score =  score_list.min()
    mean_score =  score_list.mean()
    max_score = score_list.max()
    ranking_scores = [ (score_list[i] - min_score)/(max_score-min_score) for i in range(len(score_list))]
    score_dict = dict(zip(ranking_list,ranking_scores))
    return ranking_list, score_dict

def get_ranking_list_from_lists_with_scores(r_lists, score_dict_list):
    r_ids    = list(set([ r_id for r_list  in r_lists for r_id in r_list]))
    r_scores = [0]*len(r_ids)

    if len(r_ids) == 0:
        return []

    for i, r_id in enumerate(r_ids):
        for score_dict in score_dict_list:
            if r_id in score_dict:
                score = score_dict[r_id]
                r_scores[i] += score
    r_scores, r_ids = zip(*sorted(zip(r_scores, r_ids), reverse=True))
    return r_scores, r_ids
    
def get_ranking_list_from_lists(r_lists):
    r_ids    = list(set([ r_id for r_list  in r_lists for r_id in r_list]))
    r_scores = [0]*len(r_ids)

    if len(r_ids) == 0:
        return []

    for i, r_id in enumerate(r_ids):
        for r_list in r_lists:
            if r_id in r_list:
                score = len(r_list) - r_list.index(r_id)
                r_scores[i] += score
    r_scores, r_ids = zip(*sorted(zip(r_scores, r_ids), reverse=True))
    return r_ids

def evaluate_by_sampling(user_id, target_id, candidate_pool, news_vec_pool, user_vec_pool, n=10, m=10):
    n = n - 1
    user_vec = user_vec_pool[user_id]
    keys = list(candidate_pool.keys())
    news_ids = random.sample(keys,n)
    news_ids.append(target_id)
    news_vecs = [news_vec_pool[news_id] for news_id in news_ids]
    news_coss  = cos_sim_items(user_vec,news_vecs)
    news_coss, news_ids = zip(*sorted(zip(news_coss, news_ids),reverse=True))
    news_ids = news_ids[:m]
    hr = get_hit_ratio_(news_ids, target_id)
    mrr = get_MRR_(news_ids, target_id)
    ndcg = get_NDCG_(news_ids, target_id)
    return (hr,mrr,ndcg)

def build_previous_news(df):
    df = df.copy()
    df_len = len(df)
    n_list = df['news_id'].tolist()
    t_list = df['event_time'].tolist()
    u_list = df['user_id'].tolist()
    n_p_list = ['']*df_len

    user_previous_id = {}
    for k in tqdm(range(df_len)):
        user_id = u_list[k]
        news_id = n_list[k]
        if user_id in user_previous_id:
            n_p_list[k] = user_previous_id[user_id]
        user_previous_id[user_id] = news_id
    df = pd.DataFrame(data={'user_id':u_list,'event_time':t_list,'news_id':n_list,'news_id_pre':n_p_list})
    return df

def clean_event_df(df):
    df = df[  (df['news_id'] != df['news_id_pre'])
            & (df['news_id'] != '0')
            & (df['news_id_pre'] != '0')
            & (df['news_id_pre'] != '')]
    return df

def build_news_rec_list(df):
    df = df.groupby(['news_id','news_id_pre'])['user_id'].count().reset_index(name="count")
    df['count'] = df['count'] + np.random.random_sample((len(df),)) # make each `sum` to different value
    df = df[(df['news_id'] != df['news_id_pre']) & (df['news_id'] != '0') & (df['news_id_pre'] != '0')]
    df['rank'] = df.groupby('news_id_pre')['count'].rank(ascending=False)
    df = df[(df['count'] >= 3) & (df['rank'] <= 10)]
    table = df.pivot(index='news_id_pre', columns='rank',values='news_id')
    table = table.dropna()
    return table

def get_ranking_list_from_target_id(target_id,target_vec_pool,candidate_pool,items):
    if target_id in target_vec_pool:
        target_vec = target_vec_pool[target_id]
        return get_ranking_list_from_vectors(target_vec, candidate_pool, items=items)
    else:
        return [],{}

def get_annoy_indexer(candidate_pool,trees=10):
    vec_len = len(list(candidate_pool.values())[0])
    annoy_indexer = AnnoyIndex(vec_len)  # Length of item vector that will be indexed
    for i, v in candidate_pool.items():
        annoy_indexer.add_item(int(i), v)
    annoy_indexer.build(trees) # 10 trees
    return annoy_indexer

def get_ranking_list_from_both(user_id, news_id, user_vec_pool, news_vec_pool, items, candidate_pool=None, annoy_indexer=None, with_score=True):

    _items = items*3
    r_lists = []
    score_dict_lists = []
    if annoy_indexer != None:
        if user_id in user_vec_pool:
            
            r_list, score_list = annoy_indexer.get_nns_by_vector(user_vec_pool[user_id],items,include_distances=True)
            
            r_list = [str(r_id) for r_id in r_list]
            score_list = [2-score_list[i] for i in range(len(score_list))]
            score_list = np.asarray(score_list)
            min_score =  score_list.min()
            mean_score =  score_list.mean()
            max_score = score_list.max()
            score_list = [ (score_list[i] - min_score)/(max_score-min_score) for i in range(len(score_list))]
            score_dict = {r_list[i]:score_list[i] for i in range(len(r_list))}
            r_lists.append(r_list)
            score_dict_lists.append(score_dict)
            
        if news_id in news_vec_pool:
            
            r_list, score_list = annoy_indexer.get_nns_by_vector(news_vec_pool[news_id],items,include_distances=True)
            score_list = [2-score_list[i] for i in range(len(score_list))]
            score_list = np.asarray(score_list)
            min_score =  score_list.min()
            mean_score =  score_list.mean()
            max_score = score_list.max()
            score_list = [ (score_list[i] - min_score)/(max_score-min_score) for i in range(len(score_list))]
            
            r_list = [str(r_id) for r_id in r_list]
            score_dict = {r_list[i]:score_list[i] for i in range(len(r_list))}
            r_lists.append(r_list)
            score_dict_lists.append(score_dict)

    else:
        r_list, score_dict = get_ranking_list_from_target_id(user_id,user_vec_pool,candidate_pool,_items)
        r_lists.append(r_list)
        score_dict_lists.append(score_dict)
        r_list, score_dict = get_ranking_list_from_target_id(news_id,news_vec_pool,candidate_pool,_items)
        r_lists.append(r_list)
        score_dict_lists.append(score_dict)
    
    if with_score:
        r_scores, r_list_both = get_ranking_list_from_lists_with_scores(r_lists,score_dict_lists)
    else:
        r_list_both = get_ranking_list_from_lists(r_lists)
    return r_list_both[:items]
