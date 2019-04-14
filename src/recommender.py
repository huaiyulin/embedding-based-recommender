import pandas as pd
import pickle
import logging
import numpy as np
import random
import os
import logging
import time

class Recommender:
    """

    Args: Temporary

    """
    def __init__(self, dir='../data', name=None):
        if not name:
            name = time.asctime(time.localtime(time.time()))
        self.name = name
        self.config = {}
        self.config['output_dir'] = os.path.join(dir, name)
        if not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])

        self.config['candidates_pool_path'] = os.path.join(self.config['output_dir'], 'candidates_pool.pkl')
        self.config['user_vec_pool_path'] = os.path.join(self.config['output_dir'], 'user_vec_pool.pkl')
        self.config['news_vec_pool_path'] = os.path.join(self.config['output_dir'], 'news_vec_pool.pkl')
        self.config['user_ranking_list_path'] = os.path.join(self.config['output_dir'], 'user_ranking_list.pkl')
        self.config['news_ranking_list_path'] = os.path.join(self.config['output_dir'], 'news_ranking_list.pkl')

        self.logging = logging.getLogger(name=__name__)
        self.candidates_pool = None
        self.user_vec_pool = None
        self.news_vec_pool = None
        self.user_ranking_list = None
        self.news_ranking_list = None

    def load_vec_pool(self):
        self.candidates_pool = pd.read_pickle(self.config['candidates_pool_path'])
        self.user_vec_pool   = pd.read_pickle(self.config['user_vec_pool_path'])
        self.news_vec_pool   = pd.read_pickle(self.config['news_vec_pool_path'])

    def load_ranking_list(self):
        self.user_ranking_list = pd.read_pickle(self.config['user_ranking_list_path'])
        self.news_ranking_list = pd.read_pickle(self.config['news_ranking_list_path'])

    def build_ranking_list(self, type='both', save=True, items=20):
        c_ids, c_vecs = zip(*self.candidates_pool.items())
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

    def get_ranking_list_by_user_id(self, user_id, realtime=False, items=20):
        logging.info('=== Getting === ranking_list by user_id...')
        ranking_list = None
        if user_id not in self.user_vec_pool:
            logging.info('- user_id not in user_pool')
            return []
        if realtime == True:
            c_ids, c_vecs = zip(*self.candidates_pool.items())
            u_vec = self.user_vec_pool[user_id]
            c_scores = [self._cos_sim(u_vec,c_vec) for c_vec in c_vecs]
            c_scores, c_ids = zip(*sorted(zip(c_scores, c_ids),reverse=True))
        else:
            c_scores, c_ids = zip(*self.user_ranking_list[user_id])
        return list(c_ids)[:items]

    def get_ranking_list_by_news_id(self, news_id, realtime=False, items=20):
        logging.info('=== Getting === ranking_list by news_id...')
        ranking_list = None
        if news_id not in self.news_vec_pool:
            logging.info('- news_id not in news_pool')
            return []
        if realtime == True:
            c_ids, c_vecs = zip(*self.candidates_pool.items())
            n_vec = self.news_vec_pool[news_id]
            c_scores = [self._cos_sim(n_vec,c_vec) for c_vec in c_vecs]
            c_scores, c_ids = zip(*sorted(zip(c_scores, c_ids),reverse=True))
        else:
            c_scores, c_ids = zip(*self.news_ranking_list[news_id])
        return list(c_ids)[:items]

    def add_news_vec_to_candidates_pool(self, news_id, update_pretrained_list=True):
        logging.info('=== ADDING === news_vec to candidates_pool...')
