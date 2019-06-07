import logging
import os
from annoy import AnnoyIndex
from recommender import Recommender
import pandas as pd
import random
from config import Config
from functions import load_news_meta
def logging_setup():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )

if __name__ == '__main__':
    logging_setup()
    logging.info('=============== Lauch Recommender ===============')

    recommender = Recommender(Config=Config)
    recommender.load_vec_pool()
    recommender.load_news_history()
    recommender.build_annoy_indexer()
    news_meta_dict = None

    while True:
        inputValue = input("user_id,news_id,realtime(y/n),items,by_annoy: ")

        if inputValue == 'stop':
            break

        if inputValue == 'demo':
            user_id, user_vec = random.choice(list(recommender.user_vec_pool.items()))
            news_id, news_vec = random.choice(list(recommender.news_vec_pool.items()))
            r_list_u = recommender.get_ranking_list_by_both(user_id, news_id, True, 10, False)                                   
            print("\n - user_id:{u_id}".format(u_id=user_id))
            print(" - news_id:{n_id}".format(n_id=news_id))
            print(" - rec_list:", r_list_u, "\n")
            continue
        
        if inputValue == 'annoy':
            user_id, user_vec = random.choice(list(recommender.user_vec_pool.items()))
            news_id, news_vec = random.choice(list(recommender.news_vec_pool.items()))
            r_list_u = recommender.get_ranking_list_by_both(user_id, news_id, True, 10, True)
            print("\n - user_id:{u_id}".format(u_id=user_id))
            print(" - news_id:{n_id}".format(n_id=news_id))
            print(" - rec_list:", r_list_u, "\n")
            continue

        if inputValue == 'human_validation':
            if not news_meta_dict:
                logging.info('loading news_meta for human_validation...')
                news_meta_dict = load_news_meta()
                logging.info(' - complete loading news_meta for human_validation...')
            user_id, user_vec = random.choice(list(recommender.user_vec_pool.items()))
            news_id, news_vec = random.choice(list(recommender.news_vec_pool.items()))
            r_list_u = recommender.get_ranking_list_by_both(user_id, news_id, True, 10, False)
            print("\n - user_id:{u_id}".format(u_id=user_id))
            print(" - news_id:{n_id}".format(n_id=news_id))
            print(" - rec_list:", r_list_u, "\n")
            print("--------- current  news ---------")
            print("news_title:", news_meta_dict[news_id][0])
            print("-------- reading history  --------")
            if user_id in recommender.user_to_news_pos_id:
                [print("{}: {}".format(i, news_meta_dict[news_id][0])) for i, news_id in enumerate(recommender.user_to_news_pos_id[user_id]) if news_id in news_meta_dict]
            else:
                print("user_id not exist.")
            print("-------- news recommended --------")
            [print("{}: {}".format(i, news_meta_dict[news_id][0])) for i, news_id in enumerate(r_list_u) if news_id in news_meta_dict]
            continue

        if inputValue == 'human_validation_annoy':
            if not news_meta_dict:
                logging.info('loading news_meta for human_validation...')
                news_meta_dict = load_news_meta()
                logging.info(' - complete loading news_meta for human_validation...')
            user_id, user_vec = random.choice(list(recommender.user_vec_pool.items()))
            news_id, news_vec = random.choice(list(recommender.news_vec_pool.items()))
            r_list_u = recommender.get_ranking_list_by_both(user_id, news_id, True, 10, True, False)
            print("\n - user_id:{u_id}".format(u_id=user_id))
            print(" - news_id:{n_id}".format(n_id=news_id))
            print(" - rec_list:", r_list_u, "\n")
            print("--------- current  news ---------")
            print("news_title:", news_meta_dict[news_id][0])
            print("-------- reading history  --------")
            if user_id in recommender.user_to_news_pos_id:
                [print("{}: {}".format(i, news_meta_dict[news_id][0])) for i, news_id in enumerate(recommender.user_to_news_pos_id[user_id]) if news_id in news_meta_dict]
            else:
                print("user_id not exist.")
            print("-------- news recommended --------")
            [print("{}: {}".format(i, news_meta_dict[news_id][0])) for i, news_id in enumerate(r_list_u) if news_id in news_meta_dict]
            print("--------------------with score--------------------")
            r_list_u = recommender.get_ranking_list_by_both(user_id, news_id, True, 10, True, True)
            print("\n - user_id:{u_id}".format(u_id=user_id))
            print(" - news_id:{n_id}".format(n_id=news_id))
            print(" - rec_list:", r_list_u, "\n")
            print("--------- current  news ---------")
            print("news_title:", news_meta_dict[news_id][0])
            print("-------- reading history  --------")
            if user_id in recommender.user_to_news_pos_id:
                [print("{}: {}".format(i, news_meta_dict[news_id][0])) for i, news_id in enumerate(recommender.user_to_news_pos_id[user_id]) if news_id in news_meta_dict]
            else:
                print("user_id not exist.")
            print("-------- news recommended --------")
            [print("{}: {}".format(i, news_meta_dict[news_id][0])) for i, news_id in enumerate(r_list_u) if news_id in news_meta_dict]
            continue

        if len(inputValue.split(",")) != 5:
            print('Please input exact 5 values and separate them by comma.')
            continue

        ## input processing

        user_id, news_id, realtime, items, by_annoy = inputValue.split(",")

        if realtime.lower() in ['y','yes','true']:
            realtime = True
        else:
            realtime = False

        if by_annoy.lower() in ['y','yes','true']:
            by_annoy = True
        else:
            by_annoy = False

        try:
            items = int(items)
        except:
            items = 10

        ## return rec_list
        r_list_u = recommender.get_ranking_list_by_both(user_id, news_id, realtime, items, by_annoy)
        print(r_list_u)
