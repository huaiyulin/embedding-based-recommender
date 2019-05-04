import logging
import os
from preprocessor import Preprocessor
from user_model import UserModel
from annoy import AnnoyIndex
from recommender import Recommender
import random
import pandas as pd


data_source_dir = '../data' 
output_name = '0417'
news_train_dates = {'start':12,'end':13}
news_candidates_dates = {'start':12,'end':13}
news_train_paths = [os.path.join(data_source_dir,'event-201810{}.pkl'.format(str(x).zfill(2))) for x in range(news_train_dates['start'], news_train_dates['end']+1)]
news_candidates_paths = [os.path.join(data_source_dir,'event-201810{}.pkl'.format(str(x).zfill(2))) for x in range(news_candidates_dates['start'], news_candidates_dates['end']+1)]
news_vec_pool_pah = os.path.join(data_source_dir,'news_vec_dic.pkl')

def logging_setup():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )
    

if __name__ == '__main__':
	logging_setup()
	logging.info('=============== START RUNNING ===============')

	preprocessor = Preprocessor(dir=data_source_dir,name=output_name)
	preprocessor.load_news_vec_pool(news_vec_pool_pah)
	preprocessor.load_datas_for_user_model(news_train_paths)
	preprocessor.build_user_to_news_history()
	preprocessor.build_sampling_pool(top = 5000, at_least = 10)
	preprocessor.build_vec_pairs_from_history(at_least=15)
	preprocessor.load_datas_for_candidates_pool(news_candidates_paths)
	preprocessor.build_candidates_pool(top = 5000, at_least = 10)

	user_model = UserModel(dir=data_source_dir,name=output_name)
	user_model.load_news_history()
	user_model.model_training(start=5,items=10,N=15,epochs=1, batch_size=1000, model_type='LSTM')
	user_model.save_user_vec_pool()
	user_model.save_user_model()
	user_model.load_user_model()
	user_model.load_news_history()

	recommender = Recommender(dir=data_source_dir,name=output_name)
	recommender.load_vec_pool()
	recommender.get_ranking_list_by_both(user_id,news_id, realtime=True, items=20)
	recommender.build_annoy_indexer()
	# r_list_u = recommender.get_ranking_list_by_annoy(user_id=u_id,items=10)
	# r_list_n = recommender.get_ranking_list_by_annoy(news_id=n_id,items=10)