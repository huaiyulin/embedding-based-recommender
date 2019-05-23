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
news_train_paths = [os.path.join(data_source_dir,file_name) for file_name in Config.TrainingEvent.file_names]
news_candidates_paths = [os.path.join(data_source_dir,file_name) for file_name in Config.CandidateEvent.file_names]
news_vec_pool_path = os.path.join(data_source_dir,'news_vec_pool.pkl')

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
    preprocessor.build_vec_pairs_from_history(at_least=10)
    preprocessor.load_datas_for_candidates_pool(news_candidates_paths)
    preprocessor.build_candidates_pool(top = 5000, at_least = 10)

    # user_model = UserModel(dir=data_source_dir,name=output_name)
    # user_model.load_news_history()
    # user_model.model_training(start=0,
    #                           items=10,
    #                           N=10,epochs=20, 
    #                           batch_size=256, 
    #                           model_type='GRU', 
    #                           init_rnn_by_avg=False)
    
    # recommender = Recommender(dir=data_source_dir,name=output_name)
    # recommender.load_vec_pool()
    # recommender.build_annoy_indexer()
