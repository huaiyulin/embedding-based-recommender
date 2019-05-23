import logging
import os
from annoy import AnnoyIndex
import random
import pandas as pd
import random
from preprocessor import Preprocessor
from user_model import UserModel
from recommender import Recommender
from data.config import Config
data_source_dir = Config.Directory.data_dir
news_train_paths = Config.TrainingEvent.file_paths
news_candidates_paths = Config.CandidateEvent.file_paths
news_vec_pool_path = Config.Pool.news_vec_pool_path

def logging_setup():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )

if __name__ == '__main__':
    logging_setup()
    logging.info('=============== START RUNNING ===============')

    preprocessor = Preprocessor(Config=Config)
    preprocessor.load_news_vec_pool(news_vec_pool_path)
    preprocessor.load_datas_for_user_model(news_train_paths)
    preprocessor.build_user_to_news_history_custom()
    preprocessor.build_sampling_pool(top = 5000, at_least = 10)
    preprocessor.build_id_pairs_from_history(at_least=10)
    # preprocessor.build_vec_pairs_from_history(at_least=10)
    preprocessor.load_datas_for_candidate_pool(news_candidates_paths)
    preprocessor.build_candidate_pool(top = 5000, at_least = 10)

    user_model = UserModel(Config=Config)
    user_model.load_news_history(by_id=True)
    user_model.model_training_dict_version(start=0,
                                          items=10,
                                          N=10,
                                          epochs=1, 
                                          batch_size=256, 
                                          model_type='GRU', 
                                          init_rnn_by_avg=False)
                
    recommender = Recommender(Config=Config)
    recommender.load_vec_pool()
    recommender.build_annoy_indexer()
