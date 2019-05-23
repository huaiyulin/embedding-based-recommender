import pandas as pd
import pickle
import numpy as np
import random
import os
import logging
import time
# from data.config import Config

class Preprocessor:
    """

    Args: Temporary

    """
    def __init__(self, Config = None):
        
        self.logging = logging.getLogger(name=__name__)
        self.news_id = None
        self.user_id = None
        self.event_time = None
        self.events_for_training = None
        self.events_for_candidates = None
        self.candidate_pool = None
        self.sampling_pool = None
        self.user_to_news_history = None
        self.news_vec_pool = None
        self.user_to_news_pos_id  = None
        self.user_to_news_neg_id  = None
        self.user_to_news_pos_vec  = None
        self.user_to_news_neg_vec  = None
        self.config = {}

        if Config != None:
            self.name = Config.model_name
            self.config['output_dir'] = Config.Directory.model_dir
            self.config['user_to_news_list_path']    = Config.Preprocessor.user_to_news_history_path
            self.config['candidate_pool_path']      = Config.Pool.candidate_pool_path
            self.config['user_to_news_pos_id_path']  = Config.Preprocessor.user_to_news_pos_id_path
            self.config['user_to_news_neg_id_path']  = Config.Preprocessor.user_to_news_neg_id_path
            self.config['user_to_news_pos_vec_path'] = Config.Preprocessor.user_to_news_pos_vec_path
            self.config['user_to_news_neg_vec_path'] = Config.Preprocessor.user_to_news_neg_vec_path
            self.news_id = Config.ColumnName.news_id
            self.user_id = Config.ColumnName.user_id
            self.event_time = Config.ColumnName.event_time
            if not os.path.exists(self.config['output_dir']):
                os.makedirs(self.config['output_dir'])
        else:
            self.logging.warning('You don\'t set the Config, so all paths of this preprocessor are nil!')

        
    def load_news_vec_pool(self, news_vec_pool_path):
        self.news_vec_pool = pd.read_pickle(news_vec_pool_path)

    def load_datas_for_user_model(self, news_paths):
        dfs = []
        for news_path in news_paths:
            try:
                df = pd.read_pickle(news_path)
                dfs.append(df)
                self.logging.info('loading "{}" events for model training...'.format(news_path))
            except:
                continue
        self.events_for_training = pd.concat(dfs,ignore_index=True)

    def load_datas_for_candidate_pool(self, news_paths):
        dfs = []
        for news_path in news_paths:
            try:
                df = pd.read_pickle(news_path)
                dfs.append(df)
                self.logging.info('loading "{}" events for candidate_pool...'.format(news_path))
            except:
                continue
        self.events_for_candidates = pd.concat(dfs,ignore_index=True)

    def _clean_data(self, df):
        self.logging.info('- cleanning data...')
        df = df[[self.user_id,self.news_id,self.event_time]] # 篩選出需要的欄位
        self.logging.info('- drop duplicating user/item pairs...')
        df = df.drop_duplicates(subset=[self.user_id, self.news_id]) # 清除重複的 news,user pair
        return df

    def build_user_to_news_history(self):
        self.logging.info('building user_to_news_list...')
        df = self._clean_data(self.events_for_training)
        self.logging.info('- sorting events...')
        df = df.sort_values(by=self.event_time,ascending=False) # 將點擊事件由新到舊排序
        self.logging.info('- grouping news_id by user_id...')
        df_group_by_user_id = df.groupby(self.user_id)
        self.logging.info('- convert news history into dictionary...')
        # self.user_to_news_history = df_group_by_user_id[news_id].apply(list).to_dict()
        # self.logging.info('- saving user_to_news_list...')
        # with open(self.config['user_to_news_list_path'], 'wb') as fp:
        #     pickle.dump(self.user_to_news_history,fp)
        # self.logging.info('- complete saving user_to_news_list.')


    def build_user_to_news_history_custom(self):
        self.logging.info('building user_to_news_list...')
        df = self._clean_data(self.events_for_training)
        self.logging.info('- sorting events...')
        u_list,n_list = self.custom_sort(df) # 將點擊事件由新到舊排序
        self.logging.info('- grouping news_id by user_id...')
        self.logging.info('- convert news history into dictionary...')
        self.user_to_news_history = self.custom_group_by(u_list,n_list)
        # self.logging.info('- saving user_to_news_list...')
        # with open(self.config['user_to_news_list_path'], 'wb') as fp:
        #     pickle.dump(self.user_to_news_history,fp)
        # self.logging.info('- complete saving user_to_news_list.')


    def build_candidate_pool(self, top = 5000, at_least = 10):
        self.logging.info('building candidate_pool...')
        df = self._clean_data(self.events_for_candidates)
        news_count = df.groupby(self.news_id).size().reset_index().sort_values([0],ascending = False).reset_index()
        self.candidate_pool = news_count[(news_count.index < top) + (news_count[0] > at_least).values][self.news_id].tolist()
        self.logging.info('- saving candidate_pool...')
        self.candidate_pool = {c_id:self.news_vec_pool[c_id] for c_id in self.candidate_pool if c_id in self.news_vec_pool}
        with open(self.config['candidate_pool_path'], 'wb') as fp:
            pickle.dump(self.candidate_pool,fp)
        self.logging.info('- complete saving candidate_pool.')

    def build_sampling_pool(self, top = 5000, at_least = 10):
        self.logging.info('building sampling_pool...')
        df = self._clean_data(self.events_for_training)
        news_count = df.groupby(self.news_id).size().reset_index().sort_values([0],ascending = False).reset_index()
        self.sampling_pool = news_count[(news_count.index < top) + (news_count[0] > at_least).values][self.news_id].tolist()
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
    
    def build_id_pairs_from_history(self, at_least=10):
        self.logging.info('building user_news_to_id_pairs...')
        user_to_news_pos_id = {}
        user_to_news_neg_id = {}
        self.logging.info('- negative sampling start...')
        for user_id, pos_ids in self.user_to_news_history.items():
            if len(pos_ids) < at_least:
                continue
            neg_ids = self._get_neg_ids_by_pos_ids(pos_ids)
            neg_ids = [x for x in neg_ids if x in self.news_vec_pool]
            pos_ids = [x for x in pos_ids if x in self.news_vec_pool]
            user_to_news_pos_id[user_id] = pos_ids
            user_to_news_neg_id[user_id] = neg_ids
        self.user_to_news_pos_id = user_to_news_pos_id
        self.user_to_news_neg_id = user_to_news_neg_id
        self.logging.info('- {} user_news_id_pairs builded...'.format(len(self.user_to_news_neg_id)))
        with open(self.config['user_to_news_pos_id_path'], 'wb') as fp:
            pickle.dump(self.user_to_news_pos_id,fp)
        self.logging.info('- complete saving user_to_news_pos_id.')
        with open(self.config['user_to_news_neg_id_path'], 'wb') as fp:
            pickle.dump(self.user_to_news_neg_id,fp)
        self.logging.info('- complete saving user_to_news_neg_id.')


    def build_vec_pairs_from_history(self, at_least=10):
        self.logging.info('building user_news_vec_pairs...')
        user_to_news_pos_vec = {}
        user_to_news_neg_vec = {}
        self.logging.info('- positive vector building start...')
        for user_id, ids in self.user_to_news_pos_id.items():
            vecs = self.news_ids_to_vecs(ids, items=at_least)
            user_to_news_pos_vec[user_id] = vecs
        
        self.logging.info('- negative vector building start...')
        for user_id, ids in self.user_to_news_neg_id.items():
            vecs = self.news_ids_to_vecs(ids, items=at_least)
            user_to_news_neg_vec[user_id] = vecs
            
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
        df = df.sort_values(by=self.event_time,ascending=False) # 將點擊事件由新到舊排序
        n_list = df[self.news_id].tolist()
        u_list = df[self.user_id].tolist()
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