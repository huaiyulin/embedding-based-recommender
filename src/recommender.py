import pandas as pd
import pickle
import logging
from annoy import AnnoyIndex
import numpy as np
import random
import os
import logging
import time

class Recommender:
    """

    Args: Temporary

    """
    def __init__(self, Config = None):


        self.logging = logging.getLogger(name=__name__)
        self.candidate_pool = None
        self.user_vec_pool = None
        self.news_vec_pool = None
        self.user_to_news_pos_id  = None
        self.user_ranking_list = None
        self.news_ranking_list = None
        self.annoy_indexer = None
        self.config = {}

        if Config != None:
            self.name = Config.model_name
            self.config['output_dir'] = Config.Directory.model_dir
            self.config['candidate_pool_path'] = Config.Pool.candidate_pool_path
            self.config['user_vec_pool_path'] = Config.Pool.user_vec_pool_path
            self.config['news_vec_pool_path'] = Config.Pool.news_vec_pool_path
            self.config['user_to_news_pos_id_path']  = Config.Preprocessor.user_to_news_pos_id_path
            self.config['user_ranking_list_path'] = os.path.join(self.config['output_dir'], 'user_ranking_list.pkl')
            self.config['news_ranking_list_path'] = os.path.join(self.config['output_dir'], 'news_ranking_list.pkl')
            self.config['annoy_indexer_path'] = Config.Recommender.annoy_indexer_path



    def load_news_history(self, pos_path=None):
        if not pos_path:
            pos_path=self.config['user_to_news_pos_id_path']
        self.logging.info('loading reading history...')
        with open(pos_path, 'rb') as fp:
            self.user_to_news_pos_id = pickle.load(fp)
            user_size = len(self.user_to_news_pos_id)
        self.logging.info('- complete loading reading history...')


    def load_vec_pool(self, news_vec_pool_path=None, user_vec_pool_path=None,candidate_pool=None):
        if not news_vec_pool_path:
            news_vec_pool_path = pd.read_pickle(self.config['news_vec_pool_path'])
        if not user_vec_pool_path:
            user_vec_pool_path = pd.read_pickle(self.config['user_vec_pool_path'])
        if not candidate_pool:
            candidate_pool_path = pd.read_pickle(self.config['candidate_pool_path'])
        
        self.candidate_pool = candidate_pool_path
        self.user_vec_pool   = user_vec_pool_path
        self.news_vec_pool   = news_vec_pool_path

    def load_ranking_list(self):
        self.user_ranking_list = pd.read_pickle(self.config['user_ranking_list_path'])
        self.news_ranking_list = pd.read_pickle(self.config['news_ranking_list_path'])

    def build_ranking_list(self, type='both', save=True, items=20):
        c_ids, c_vecs = zip(*self.candidate_pool.items())
        if type == 'both':
            logging.info('=== Building === ranking_list by user_id...')
            i = 0
            self.user_ranking_list = {}
            for u_id, u_vec in self.user_vec_pool.items():
                i += 1
                if i%500 == 0:
                    logging.info(' - progressing {} items..'.format(i))
                c_scores = [self._cos_sim(u_vec,c_vec) for c_vec in c_vecs]
                self.user_ranking_list[u_id] = sorted(zip(c_scores, c_ids),reverse=True)[:items]
            self.news_ranking_list = {}

            logging.info('=== Building === ranking_list by news_id...')
            i = 0
            for n_id, n_vec in self.news_vec_pool.items():
                i += 1
                if i%500 == 0:
                    logging.info(' - progressing {} items..'.format(i))
                c_scores = [self._cos_sim(n_vec,c_vec) for c_vec in c_vecs]
                self.news_ranking_list[n_id] = sorted(zip(c_scores, c_ids),reverse=True)[:items]
            if save:
                with open(self.config['user_ranking_list_path'], 'wb') as fp:
                    pickle.dump(self.user_ranking_list,fp)
                logging.info('complete saving user_ranking_list.')
                with open(self.config['news_ranking_list_path'], 'wb') as fp:
                    pickle.dump(self.news_ranking_list,fp)
                logging.info('complete saving news_ranking_list.')

        elif type == 'user':
            self.user_ranking_list = {}
            for u_id, u_vec in self.user_vec_pool.items():
                c_scores = [self._cos_sim(u_vec,c_vec) for c_vec in c_vecs]
                self.user_ranking_list[u_id] = sorted(zip(c_scores, c_ids),reverse=True)[:items]
            if save:
                with open(self.config['user_ranking_list_path'], 'wb') as fp:
                    pickle.dump(self.user_ranking_list,fp)
                logging.info('complete saving user_ranking_list.')

        elif type == 'news':
            self.news_ranking_list = {}
            for n_id, n_vec in self.news_vec_pool.items():
                c_scores = [self._cos_sim(n_vec,c_vec) for c_vec in c_vecs]
                self.news_ranking_list[n_id] = sorted(zip(c_scores, c_ids),reverse=True)[:items]
            if save:
                with open(self.config['news_ranking_list_path'], 'wb') as fp:
                    pickle.dump(self.news_ranking_list,fp)
                logging.info('complete saving news_ranking_list.')

        else:
            pass


    def _cos_sim(self, a, b):
        """Takes 2 vectors a, b and returns the cosine similarity according 
        to the definition of the dot product
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def get_ranking_list_by_user_id(self, user_id, realtime=True, items=20):
        ranking_list = None
        if user_id not in self.user_vec_pool:
            logging.info('- user_id not in user_pool')
            return []
        if realtime == True:
            c_ids, c_vecs = zip(*self.candidate_pool.items())
            u_vec = self.user_vec_pool[user_id]
            c_scores = [self._cos_sim(u_vec,c_vec) for c_vec in c_vecs]
            c_scores, c_ids = zip(*sorted(zip(c_scores, c_ids),reverse=True))
        else:
            c_scores, c_ids = zip(*self.user_ranking_list[user_id])
        return list(c_ids)[:items]

    def get_ranking_list_by_news_id(self, news_id, realtime=True, items=20):
        ranking_list = None
        if news_id not in self.news_vec_pool:
            logging.info('- news_id not in news_pool')
            return []
        if realtime == True:
            c_ids, c_vecs = zip(*self.candidate_pool.items())
            n_vec = self.news_vec_pool[news_id]
            c_scores = [self._cos_sim(n_vec,c_vec) for c_vec in c_vecs]
            c_scores, c_ids = zip(*sorted(zip(c_scores, c_ids),reverse=True))
        else:
            c_scores, c_ids = zip(*self.news_ranking_list[news_id])
        return list(c_ids)[:items]

    def get_ranking_list_by_both(self,user_id,news_id, realtime=True, items=20):
        ranking_list = None
        items_ = items*3
        r_list_n = self.get_ranking_list_by_news_id(news_id, realtime=realtime, items=items_)
        r_list_u = self.get_ranking_list_by_user_id(user_id, realtime=realtime, items=items_)
        r_ids    = list(set(r_list_u+r_list_n))
        r_scores = [0]*len(r_ids)
        if len(r_ids) == 0:
            return []
        for i, r_id in enumerate(r_ids):
            if r_id in r_list_n:
                r = r_list_n.index(r_id)
                r_scores[i] += (items_-r)
            if r_id in r_list_u:
                r = r_list_u.index(r_id)
                r_scores[i] += (items_-r)
        r_scores, r_ids = zip(*sorted(zip(r_scores, r_ids), reverse=True))
        if self.user_to_news_pos_id != None:
            if user_id in self.user_to_news_pos_id:
                history_list = self.user_to_news_pos_id[user_id]
                r_ids = [ r_id for r_id in r_ids if r_id not in history_list]
        return r_ids[:items]


    def add_news_vec_to_candidate_pool(self, news_id, update_pretrained_list=True):
        logging.info('=== ADDING === news_vec to candidate_pool...')

    def build_annoy_indexer(self, trees=10, path=None):
        if not path:
            path = self.config['annoy_indexer_path']
        vec_len = len(list(self.candidate_pool.values())[0])
        t = AnnoyIndex(vec_len)  # Length of item vector that will be indexed
        for i, v in self.candidate_pool.items():
            t.add_item(int(i), v)
        t.build(trees) # 10 trees
        self.annoy_indexer = t
        t.save(path)

    def load_annoy_indexer(self, path=None):
        if not path:
            path = self.config['annoy_indexer_path']
        vec_len = len(list(self.candidate_pool.values())[0])
        t = AnnoyIndex(vec_len)
        t.load(path)
        self.annoy_indexer = t

    def get_ranking_list_by_annoy(self, news_id=None,user_id=None,items=20):
        t_vec = None
        if (user_id != None) & (user_id in self.user_vec_pool):
            t_vec = self.user_vec_pool[user_id]
        elif (news_id != None) & (news_id in self.news_vec_pool):
            t_vec = self.news_vec_pool[news_id]
        else:
            logging.info('user_id/news_id does not exist.')
            return
        return self.annoy_indexer.get_nns_by_vector(t_vec,items,include_distances=False)