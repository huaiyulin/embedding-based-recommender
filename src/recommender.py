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
        self.logging = logging.getLogger(name=__name__)
        self.candidates_pool = None
        self.user_vec_pool = None
        self.item_vec_pool = None
        self.user_rec_pool = None
        self.item_rec_pool = None
        self.config = {}
        if not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])
        self.config['user_to_news_list_path']    = os.path.join(self.config['output_dir'], 'user_to_news_list.pkl')
        self.config['candidates_pool_path']      = os.path.join(self.config['output_dir'], 'candidates_pool.pkl')
        self.config['user_to_news_pos_vec_path'] = os.path.join(self.config['output_dir'], 'user_to_news_pos_vec.pkl')
        self.config['user_to_news_neg_vec_path'] = os.path.join(self.config['output_dir'], 'user_to_news_neg_vec.pkl')
    
    def load_model_from_directory(self):
        self.candidates_pool = pd.read_pickle(self.config['candidates_pool_path'])
        self.user_vec_pool   = pd.read_pickle(user_vec_pool_pah)
        self.item_vec_pool   = pd.read_pickle(item_vec_pool_pah)

    def load_pretrained_vec(self, user_vec_pool_path, item_vec_pool_path):
        self.user_vec_pool = pd.read_pickle(user_vec_pool_path)
        self.item_vec_pool = pd.read_pickle(item_vec_pool_path)

    def build_rec_list(self, type='both'):
        if type == 'both':
            pass
        elif type == 'user':
            pass
        elif type == 'news':
            pass
        else:
            pass

    def add_news_vec_to_candidates_pool(self, news_id, update_pretrained_list=True):
        logging.info('=== ADDING === user_news_vec_pairs...')
    
    def get_rec_list_by_user_id(self, user_id, realtime=False):
        logging.info('=== BUILDING === user_news_vec_pairs...')


    def get_rec_list_by_news_id(self, news_id, realtime=False):
        logging.info('=== BUILDING === user_news_vec_pairs...')


    