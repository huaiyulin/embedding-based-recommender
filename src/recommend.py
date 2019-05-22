import logging
import os
from annoy import AnnoyIndex
from recommender import Recommender
import pandas as pd
import random

data_source_dir = '../data' 
output_name = '0518'

def logging_setup():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )
    

if __name__ == '__main__':
	logging_setup()
	logging.info('=============== Lauch Recommender ===============')

	recommender = Recommender(dir=data_source_dir,name=output_name)
	recommender.load_vec_pool()
	

	while True:
		inputValue = input("user_id,news_id,realtime(y/n),items: ")
		
		if inputValue == 'stop':
			break

		if inputValue == 'demo':
			user_id, user_vec = random.choice(list(recommender.user_vec_pool.items()))
			news_id, news_vec = random.choice(list(recommender.news_vec_pool.items()))
			r_list_u = recommender.get_ranking_list_by_both(user_id,news_id, True, 10)
			print("user_id:{u_id}".format(u_id=user_id))
			print("news_id:{n_id}".format(n_id=news_id))
			print(r_list_u)
			continue
		
		if len(inputValue.split(",")) != 4:
			print('Please input exact 4 values and separate them by comma.')
			continue
		
		## input processing

		user_id, news_id, realtime, items = inputValue.split(",")	
		
		if realtime == 'n':
			realtime = False
		else:
			realtime = True
		try:
			items = int(items)
		except:
			items = 10

		## return rec_list
		r_list_u = recommender.get_ranking_list_by_both(user_id, news_id, realtime, items)
		print(r_list_u)
		