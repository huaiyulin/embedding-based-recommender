import logging
import os
from user_model import UserModel
from preprocessor import Preprocessor


data_source_dir = '../data' 
output_name = 'default'

news_train_dates = {'start':12,'end':13}
news_candidates_dates = {'start':12,'end':13}
news_train_paths = [os.path.join(data_source_dir,'event-201810{}.pkl'.format(str(x).zfill(2))) for x in range(news_train_dates['start'], news_train_dates['end'])]
news_candidates_paths = [os.path.join(data_source_dir,'event-201810{}.pkl'.format(str(x).zfill(2))) for x in range(news_candidates_dates['start'], news_candidates_dates['end'])]
news_vec_dic_pah = os.path.join(data_source_dir,'news_vec_dic.pkl')

vector_dir = os.path.join(data_source_dir,'mid-product'+output_name)

user_to_news_pos_vec_path = os.path.join(vector_dir, 'user_to_news_pos_vec.pkl')
user_to_news_neg_vec_path = os.path.join(vector_dir, 'user_to_news_neg_vec.pkl')
user_vec_path = os.path.join(data_source_dir,'user_vec.pkl')

def logging_setup():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )
    

if __name__ == '__main__':
logging_setup()
    
preprocessor = Preprocessor(name=output_name)
preprocessor.load_datas_for_user_model(news_train_paths)
preprocessor.load_datas_for_candidates_pool(news_candidates_paths)
preprocessor.load_news_dict(news_vec_dic_pah)
preprocessor.build_user_to_news_history()
preprocessor.build_candidates_pool()
preprocessor.build_vec_pairs_from_history()

user_model = UserModel()
user_model.load_news_history(user_to_news_pos_vec_path, user_to_news_neg_vec_path)
user_model.model_training()
user_model.save_to(user_vec_path)
