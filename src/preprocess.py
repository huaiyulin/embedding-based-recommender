import pandas as pd
import pickle
import datetime
import logging
import numpy as np
import random
import os

data_source_dir = '../data'
output_temp_dir = os.path.join(data_source_dir,'mid-product')
if not os.path.exists(output_temp_dir):
        os.makedirs(output_temp_dir)
        
news_train_dates = {'start':11,'end':12}
news_recommend_dates = {'start':11,'end':12}
news_train_paths = [os.path.join(data_source_dir,'event-201810{}.pkl'.format(str(x).zfill(2))) for x in range(news_train_dates['start'], news_train_dates['end'])]
news_recommend_paths = [os.path.join(data_source_dir,'event-201810{}.pkl'.format(str(x).zfill(2))) for x in range(news_recommend_dates['start'], news_recommend_dates['end'])]
news_vec_dic_pah = os.path.join(data_source_dir,'news_vec_dic.pkl')

user_to_news_list_path = os.path.join(output_temp_dir, 'user_to_news_list.pkl')
recommend_pool_path  = os.path.join(output_temp_dir, 'recommend_pool.pkl')
user_to_news_pos_vec_path = os.path.join(output_temp_dir, 'user_to_news_pos_vec.pkl')
user_to_news_neg_vec_path = os.path.join(output_temp_dir, 'user_to_news_neg_vec.pkl')

# datasource config
event_time = 'event_timestamp'
news_id    = 'page.item.id'
user_id    = 'cookie_mapping.et_token'

def logging_setup():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )

def load_datas(news_paths):
    logging.info('loading {} days events...'.format(len(news_paths)))
    dfs = []
    for news_path in news_paths:
        df = pd.read_pickle(news_path)
        dfs.append(df)
    df = pd.concat(dfs,ignore_index=True)
    return df

def clean_data(df):
    logging.info('cleanning data...')
    df = df[[user_id,news_id,event_time]] # 篩選出需要的欄位
    df = df.drop_duplicates(subset=[user_id, news_id]) # 清除重複的 news,user pair
    return df

def dataframe_to_user_news_dic(df):
    logging.info('building user_to_news_list...')
    df_group_by_user_id = df.groupby(user_id)
    user_to_news_list = df_group_by_user_id[news_id].apply(list).to_dict()
    return user_to_news_list

def build_user_to_news_history():
    logging.info('=== BUILDING === user_to_news_list...')
    df = load_datas(news_train_paths)
    df = clean_data(df)
    df.sort_values(by=event_time,ascending=False, inplace=True) # 將點擊事件由新到舊排序
    user_to_news_list = dataframe_to_user_news_dic(df)
    logging.info('saving user_to_news_list...')

    with open(user_to_news_list_path, 'wb') as fp:
        pickle.dump(user_to_news_list,fp)
    logging.info('complete saving user_to_news_list.')
    return user_to_news_list

def build_recommend_pool(top = 5000, at_least = 10):
    logging.info('=== BUILDING === recommend_pool...')
    df = load_datas(news_recommend_paths)
    df = clean_data(df)
    news_count = df.groupby(news_id).size().reset_index().sort_values([0],ascending = False).reset_index()
    recommend_pool = news_count[(news_count.index < top) + (news_count[0] > at_least).values][news_id].tolist()
    logging.info('saving recommend_pool...')

    with open(recommend_pool_path, 'wb') as fp:
        pickle.dump(recommend_pool,fp)
    logging.info('complete saving recommend_pool.')
    return recommend_pool

def convert_news_id_to_vec(news_vec_dic,news_id):
    if news_id in news_vec_dic:
        return news_vec_dic[news_id]
    else:
        return None

def news_ids_to_vecs(news_list,news_vec_dic,items=-1):
    news_vecs = np.asarray([news_vec_dic[x] for x in news_list if x in news_vec_dic])
    if items != -1:
        news_vecs = news_vecs[:items]
    return news_vecs

def get_neg_ids_by_pos_ids(pos_ids, sampling_pool):
    neg_ids = random.sample(sampling_pool, len(pos_ids)*2)
    neg_ids = [x for x in neg_ids if x not in pos_ids]
    return neg_ids

def build_vec_pairs_from_history(user_news_dic,sampling_pool,news_vec_dic):
    logging.info('=== BUILDING === user_news_vec_pairs...')
    user_to_news_pos_vec = {}
    user_to_news_neg_vec = {}
    for user_id, pos_ids in user_news_dic.items():
        neg_ids = get_neg_ids_by_pos_ids(pos_ids,sampling_pool)
        pos_vecs = news_ids_to_vecs(pos_ids, news_vec_dic)
        neg_vecs = news_ids_to_vecs(neg_ids, news_vec_dic, items=len(pos_vecs))
        user_to_news_pos_vec[user_id] = pos_vecs
        user_to_news_neg_vec[user_id] = neg_vecs
    with open(user_to_news_pos_vec_path, 'wb') as fp:
        pickle.dump(user_to_news_pos_vec,fp)
    logging.info('complete saving user_to_news_pos_vec.')
    with open(user_to_news_neg_vec_path, 'wb') as fp:
        pickle.dump(user_to_news_neg_vec,fp)
    logging.info('complete saving user_to_news_neg_vec.')

    return user_to_news_pos_vec, user_to_news_neg_vec


if __name__ == '__main__':
    logging_setup()
    user_news_dic = build_user_to_news_history()
    sampling_pool = build_recommend_pool()
    news_vec_dic  = pd.read_pickle(news_vec_dic_pah)
    user_to_news_pos_vec, user_to_news_neg_vec = build_vec_pairs_from_history(user_news_dic,sampling_pool,news_vec_dic)