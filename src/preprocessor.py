import pandas as pd
import pickle
import logging
import numpy as np
import random
import os
import logging
import time


data_source_dir = '../data'

# datasource config
event_time = 'event_timestamp'
news_id    = 'page.item.id'
user_id    = 'cookie_mapping.et_token'

class Preprocessor:
    """

    Args: Temporary

    """
    def __init__(self, name=None):
        self.model_name = name
        self.logging = logging.getLogger(name=__name__)
        self.events_for_training = None
        self.events_for_candidates = None
        self.candidates_pool = None
        self.user_to_news_history = None
        self.news_dict = None
        self.config = {}
        if name:
            self.config['output_dir'] = os.path.join(data_source_dir, 'mid-product-'+name)
        else:
            localtime = time.asctime(time.localtime(time.time()))
            self.config['output_dir'] = os.path.join(data_source_dir, 'mid-product-'+localtime)
        if not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])
        self.config['user_to_news_list_path']    = os.path.join(self.config['output_dir'], 'user_to_news_list.pkl')
        self.config['candidates_pool_path']      = os.path.join(self.config['output_dir'], 'candidates_pool.pkl')
        self.config['user_to_news_pos_vec_path'] = os.path.join(self.config['output_dir'], 'user_to_news_pos_vec.pkl')
        self.config['user_to_news_neg_vec_path'] = os.path.join(self.config['output_dir'], 'user_to_news_neg_vec.pkl')

    def load_news_dict(self, news_dic_pah):
        self.news_dict = pd.read_pickle(news_dic_pah)

    def load_datas_for_user_model(self, news_paths):
        logging.info('loading {} days events for model training...'.format(len(news_paths)))
        dfs = []
        for news_path in news_paths:
            df = pd.read_pickle(news_path)
            dfs.append(df)
        self.events_for_training = pd.concat(dfs,ignore_index=True)

    def load_datas_for_candidates_pool(self, news_paths):
        logging.info('loading {} days events for candidates_pool...'.format(len(news_paths)))
        dfs = []
        for news_path in news_paths:
            df = pd.read_pickle(news_path)
            dfs.append(df)
        self.events_for_candidates = pd.concat(dfs,ignore_index=True)

    def clean_data(self, df):
        logging.info('cleanning data...')
        df = df[[user_id,news_id,event_time]] # 篩選出需要的欄位
        df = df.drop_duplicates(subset=[user_id, news_id]) # 清除重複的 news,user pair
        return df

    def build_user_to_news_history(self):
        logging.info('=== BUILDING === user_to_news_list...')
        df = self.clean_data(self.events_for_training)
        df.sort_values(by=event_time,ascending=False, inplace=True) # 將點擊事件由新到舊排序
        logging.info('building user_to_news_list...')
        df_group_by_user_id = df.groupby(user_id)
        self.user_to_news_history = df_group_by_user_id[news_id].apply(list).to_dict()
        logging.info('saving user_to_news_list...')

        with open(self.config['user_to_news_list_path'], 'wb') as fp:
            pickle.dump(self.user_to_news_history,fp)
        logging.info('complete saving user_to_news_list.')
        return self.user_to_news_history

    def build_candidates_pool(self, top = 5000, at_least = 10):
        logging.info('=== BUILDING === candidates_pool...')
        df = self.clean_data(self.events_for_candidates)
        news_count = df.groupby(news_id).size().reset_index().sort_values([0],ascending = False).reset_index()
        self.candidates_pool = news_count[(news_count.index < top) + (news_count[0] > at_least).values][news_id].tolist()
        logging.info('saving candidates_pool...')

        with open(self.config['candidates_pool_path'], 'wb') as fp:
            pickle.dump(self.candidates_pool,fp)
        logging.info('complete saving candidates_pool.')
        return self.candidates_pool

    def news_ids_to_vecs(self, news_list,items=-1):
        news_vecs = np.asarray([self.news_dict[x] for x in news_list if x in self.news_dict])
        if items != -1:
            news_vecs = news_vecs[:items]
        return news_vecs

    def get_neg_ids_by_pos_ids(self, pos_ids):
        neg_ids = random.sample(self.candidates_pool, len(pos_ids)*2)
        neg_ids = [x for x in neg_ids if x not in pos_ids]
        return neg_ids

    def build_vec_pairs_from_history(self):
        logging.info('=== BUILDING === user_news_vec_pairs...')
        user_to_news_pos_vec = {}
        user_to_news_neg_vec = {}
        for user_id, pos_ids in self.user_to_news_history.items():
            neg_ids = self.get_neg_ids_by_pos_ids(pos_ids)
            pos_vecs = self.news_ids_to_vecs(pos_ids)
            neg_vecs = self.news_ids_to_vecs(neg_ids, items=len(pos_vecs))
            user_to_news_pos_vec[user_id] = pos_vecs
            user_to_news_neg_vec[user_id] = neg_vecs
        with open(self.config['user_to_news_pos_vec_path'], 'wb') as fp:
            pickle.dump(user_to_news_pos_vec,fp)
        logging.info('complete saving user_to_news_pos_vec.')
        with open(self.config['user_to_news_neg_vec_path'], 'wb') as fp:
            pickle.dump(user_to_news_neg_vec,fp)
        logging.info('complete saving user_to_news_neg_vec.')

        return user_to_news_pos_vec, user_to_news_neg_vec

