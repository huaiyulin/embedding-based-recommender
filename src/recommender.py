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
        self.config['annoy_index_path'] = os.path.join(self.config['output_dir'], 'annoy_index.ann')

        self.logging = logging.getLogger(name=__name__)
        self.candidates_pool = None
        self.user_vec_pool = None
        self.news_vec_pool = None
        self.user_ranking_list = None
        self.news_ranking_list = None
        self.annoy_indexer = None

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

    def get_ranking_list_by_user_id(self, user_id, realtime=True, items=20):
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

    def get_ranking_list_by_news_id(self, news_id, realtime=True, items=20):
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

    def get_ranking_list_by_both(self,user_id,news_id, realtime=True, items=20):
        ranking_list = None
        r_list_n = self.get_ranking_list_by_news_id(news_id, realtime=realtime, items=items*2)
        r_list_u = self.get_ranking_list_by_user_id(user_id, realtime=realtime, items=items*2)
        r_ids    = list(set(r_list_u+r_list_n))
        r_scores = [0]*len(r_ids)
        if len(r_ids) == 0:
            return []
        for i, r_id in enumerate(r_ids):
            if r_id in r_list_n:
                r = r_list_n.index(r_id)
                r_scores[i] += (items*2-r)
            if r_id in r_list_u:
                r = r_list_u.index(r_id)
                r_scores[i] += (items*2-r)
        r_scores, r_ids = zip(*sorted(zip(r_scores, r_ids), reverse=True))
        return r_ids[:items]


    def add_news_vec_to_candidates_pool(self, news_id, update_pretrained_list=True):
        logging.info('=== ADDING === news_vec to candidates_pool...')

    def build_annoy_indexer(self, trees=10, path=None):
        if not path:
            path = self.config['annoy_index_path']
        vec_len = len(list(self.candidates_pool.values())[0])
        t = AnnoyIndex(vec_len)  # Length of item vector that will be indexed
        for i, v in self.candidates_pool.items():
            t.add_item(int(i), v)
        t.build(trees) # 10 trees
        self.annoy_indexer = t
        t.save(path)

    def load_annoy_indexer(self, path=None):
        if not path:
            path = self.config['annoy_index_path']
        vec_len = len(list(self.candidates_pool.values())[0])
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