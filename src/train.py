import logging
import os
from annoy import AnnoyIndex
import random
import pandas as pd
import random
from preprocessor import Preprocessor
from user_model import UserModel
from recommender import Recommender
from config import Config


def logging_setup():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )

if __name__ == '__main__':
    logging_setup()
    logging.info('=============== START RUNNING ===============')

    preprocessor = Preprocessor(Config=Config)
    preprocessor.load_news_vec_pool()
    preprocessor.load_datas_for_user_model()
    preprocessor.build_user_to_news_history_custom()
    preprocessor.build_sampling_pool(top = 5000, at_least = 10)
    preprocessor.build_id_pairs_from_history(at_least=3)
    # preprocessor.build_vec_pairs_from_history(at_least=10)
    preprocessor.load_datas_for_candidate_pool()
    preprocessor.build_candidate_pool(top = 2000, at_least = 100)

    user_model = UserModel(Config=Config)
    user_model.load_news_history(by_id=True)
    user_model.model_training_dict_version(start=0,
                                          items=10,
                                          N=10,
                                          epochs=20, 
                                          batch_size=256, 
                                          model_type='GRU', 
                                          init_rnn_by_avg=False)

    recommender = Recommender(Config=Config)
    recommender.load_vec_pool()
    recommender.build_annoy_indexer()
