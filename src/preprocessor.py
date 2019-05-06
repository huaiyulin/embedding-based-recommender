import pandas as pd
import pickle
import numpy as np
import random
import os
import logging
import time

# datasource config
event_time = 'event_timestamp'
news_id    = 'page.item.id'
user_id    = 'eds_id'

class Preprocessor:
    """

    Args: Temporary

    """
    def __init__(self, dir='../data', name=None):
        if not name:
            name = time.asctime(time.localtime(time.time()))
        self.name = name
        self.config = {}
        self.config['output_dir'] = os.path.join(dir, name)
        self.config['user_to_news_list_path']    = os.path.join(self.config['output_dir'], 'user_to_news_list.pkl')
        self.config['candidates_pool_path']      = os.path.join(self.config['output_dir'], 'candidates_pool.pkl')
        self.config['user_to_news_pos_vec_path'] = os.path.join(self.config['output_dir'], 'user_to_news_pos_vec.pkl')
        self.config['user_to_news_neg_vec_path'] = os.path.join(self.config['output_dir'], 'user_to_news_neg_vec.pkl')

        if not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])

        self.logging = logging.getLogger(name=__name__)
        self.events_for_training = None
        self.events_for_candidates = None
        self.candidates_pool = None
        self.sampling_pool = None
        self.user_to_news_history = None
        self.news_vec_pool = None
        self.user_to_news_pos_vec  = None
        self.user_to_news_neg_vec  = None
        
    def load_news_vec_pool(self, news_vec_pool_path):
        self.news_vec_pool = pd.read_pickle(news_vec_pool_path)

    def load_datas_for_user_model(self, news_paths):
        self.logging.info('loading {} days events for model training...'.format(len(news_paths)))
        dfs = []
        for news_path in news_paths:
            df = pd.read_pickle(news_path)
            dfs.append(df)
        self.events_for_training = pd.concat(dfs,ignore_index=True)

    def load_datas_for_candidates_pool(self, news_paths):
        self.logging.info('loading {} days events for candidates_pool...'.format(len(news_paths)))
        dfs = []
        for news_path in news_paths:
            df = pd.read_pickle(news_path)
            dfs.append(df)
        self.events_for_candidates = pd.concat(dfs,ignore_index=True)

    def _clean_data(self, df):
        self.logging.info('- cleanning data...')
        df = df[[user_id,news_id,event_time]] # 篩選出需要的欄位
        self.logging.info('- drop duplicating user/item pairs...')
        df = df.drop_duplicates(subset=[user_id, news_id]) # 清除重複的 news,user pair
        return df

    def build_user_to_news_history(self):
        self.logging.info('building user_to_news_list...')
        df = self._clean_data(self.events_for_training)
        self.logging.info('- sorting events...')
        df = df.sort_values(by=event_time,ascending=False) # 將點擊事件由新到舊排序
        self.logging.info('- grouping news_id by user_id...')
        df_group_by_user_id = df.groupby(user_id)
        self.logging.info('- convert news history into dictionary...')
        self.user_to_news_history = df_group_by_user_id[news_id].apply(list).to_dict()
        self.logging.info('- saving user_to_news_list...')

        with open(self.config['user_to_news_list_path'], 'wb') as fp:
            pickle.dump(self.user_to_news_history,fp)
        self.logging.info('- complete saving user_to_news_list.')


    def build_user_to_news_history_custom(self):
        self.logging.info('building user_to_news_list...')
        df = self._clean_data(self.events_for_training)
        self.logging.info('- sorting events...')
        u_list,n_list = self.custom_sort(df) # 將點擊事件由新到舊排序
        self.logging.info('- grouping news_id by user_id...')
        self.logging.info('- convert news history into dictionary...')
        self.user_to_news_history = self.custom_group_by(u_list,n_list)
        self.logging.info('- saving user_to_news_list...')

        with open(self.config['user_to_news_list_path'], 'wb') as fp:
            pickle.dump(self.user_to_news_history,fp)
        self.logging.info('- complete saving user_to_news_list.')


    def build_candidates_pool(self, top = 5000, at_least = 10):
        self.logging.info('building candidates_pool...')
        df = self._clean_data(self.events_for_candidates)
        news_count = df.groupby(news_id).size().reset_index().sort_values([0],ascending = False).reset_index()
        self.candidates_pool = news_count[(news_count.index < top) + (news_count[0] > at_least).values][news_id].tolist()
        self.logging.info('- saving candidates_pool...')
        self.candidates_pool = {c_id:self.news_vec_pool[c_id] for c_id in self.candidates_pool if c_id in self.news_vec_pool}
        with open(self.config['candidates_pool_path'], 'wb') as fp:
            pickle.dump(self.candidates_pool,fp)
        self.logging.info('- complete saving candidates_pool.')

    def build_sampling_pool(self, top = 5000, at_least = 10):
        self.logging.info('building sampling_pool...')
        df = self._clean_data(self.events_for_training)
        news_count = df.groupby(news_id).size().reset_index().sort_values([0],ascending = False).reset_index()
        self.sampling_pool = news_count[(news_count.index < top) + (news_count[0] > at_least).values][news_id].tolist()
        self.logging.info('- complete building sampling_pool.')

    def news_ids_to_vecs(self, news_list,items=-1):
        news_vecs = np.asarray([self.news_vec_pool[x] for x in news_list if x in self.news_vec_pool])
        if items != -1:
            news_vecs = news_vecs[:items]
        return news_vecs

    def _get_neg_ids_by_pos_ids(self, pos_ids):
        neg_ids = random.sample(self.sampling_pool, len(pos_ids)*2)
        neg_ids = [x for x in neg_ids if x not in pos_ids]
        return neg_ids

    def build_pos_vec_from_history(self, at_least=10):
        self.logging.info('building user_news_vec_positive...')
        user_to_news_pos_vec = {}
        for user_id, pos_ids in self.user_to_news_history.items():
            if len(pos_ids) < at_least:
                continue
            pos_vecs = self.news_ids_to_vecs(pos_ids)
            user_to_news_pos_vec[user_id] = pos_vecs
        self.logging.info('- complete building user_to_news_pos_vec.')
        self.user_to_news_pos_vec = user_to_news_pos_vec
        with open(self.config['user_to_news_pos_vec_path'], 'wb') as fp:
            pickle.dump(self.user_to_news_pos_vec,fp)
        self.logging.info('- complete saving user_to_news_pos_vec.')
        
    def build_vec_pairs_from_history(self, at_least=10):
        self.logging.info('building user_news_vec_pairs...')
        user_to_news_pos_vec = {}
        user_to_news_neg_vec = {}
        self.logging.info('- negative sampling and vector building start...')
        for user_id, pos_ids in self.user_to_news_history.items():
            if len(pos_ids) < at_least:
                continue
            neg_ids = self._get_neg_ids_by_pos_ids(pos_ids)
            neg_vecs = self.news_ids_to_vecs(neg_ids, items=at_least)
            user_to_news_neg_vec[user_id] = neg_vecs

        self.logging.info('- positive vector building start...')
        for user_id, pos_ids in self.user_to_news_history.items():
            if len(pos_ids) < at_least:
                continue
            pos_vecs = self.news_ids_to_vecs(pos_ids, items=at_least)
            user_to_news_pos_vec[user_id] = pos_vecs
            
        self.user_to_news_pos_vec = user_to_news_pos_vec
        self.user_to_news_neg_vec = user_to_news_neg_vec
        self.logging.info('- {} user_news_vec_pairs builded...'.format(len(self.user_to_news_pos_vec)))
        with open(self.config['user_to_news_pos_vec_path'], 'wb') as fp:
            pickle.dump(self.user_to_news_pos_vec,fp)
        self.logging.info('- complete saving user_to_news_pos_vec.')
        with open(self.config['user_to_news_neg_vec_path'], 'wb') as fp:
            pickle.dump(self.user_to_news_neg_vec,fp)
        self.logging.info('- complete saving user_to_news_neg_vec.')

    def custom_sort(self, df):
        df = df.sort_values(by=event_time,ascending=False) # 將點擊事件由新到舊排序
        n_list = df[news_id].tolist()
        u_list = df[user_id].tolist()
        # t_list = df[event_time].tolist()
        # t_list,u_list,n_list = zip(*sorted(zip(t_list,u_list,n_list), reverse=True))
        return u_list,n_list

    def custom_group_by(self, u_list,n_list):
        u_dic = {}
        for i, r in enumerate(u_list):
            if r not in u_dic:
                u_dic[u_list[i]] = [n_list[i]]
            else:
                u_dic[u_list[i]].append(n_list[i])
        return u_dic