import pandas as pd
import numpy as np
from get_trainTXT_IDList import get_txt_IDList
import pickle
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-i", dest="new_meta_data", type=str, help="new news meta data")
args = parser.parse_args()

def del_first_line(file_name):
    fin = open(file_name,encoding='utf-8')
    a = fin.readlines()
    fout = open(file_name,'w',encoding='utf-8')
    b =''.join(a[1:])
    fout.write(b)
    fin.close()
    fout.close()
    return 0

def get_word_vec(word_vec_file):
    del_first_line(word_vec_file)
    wordvectors = pd.read_csv(word_vec_file, sep=" ", header=None)
    words = wordvectors[0].tolist()
    wordvectors = wordvectors.drop([0], axis=1)
    wordvectors = wordvectors.drop([wordvectors.shape[1]], axis=1)
    wordvectors = wordvectors.as_matrix()
    for i in range(len(words)):
        if type(words[i]) is float:
            words[i] = 'NA'
    return words, wordvectors

def compute_new_news_vec(new_news_seg,words,wordvectors,ID_list,news_vec_pool):
    j = 0
    doc_vec = np.zeros(100)
    for k,doc in enumerate(new_news_seg):
        for i, word in enumerate(words):
            if word in doc:        
                doc_vec = doc_vec + wordvectors[i]
                j += 1
        doc_vec = doc_vec / j
        news_vec_pool[ID_list[k]] = doc_vec
    return news_vec_pool

if __name__ == '__main__':
    
    ID_list, new_news_seg = get_txt_IDList('../data/'+args.new_meta_data)
    words, wordvectors = get_word_vec('../data/wordvectors.txt')
    
    with open('../data/news_vec_pool.pkl', 'rb') as f1:
        news_vec_pool = pickle.load(f1)
    with open('../data/candidate_pool.pkl', 'rb') as f2:
        candidate_pool = pickle.load(f2)
    
    news_vec_pool = compute_new_news_vec(new_news_seg,words,wordvectors,ID_list,news_vec_pool)
    candidate_pool = compute_new_news_vec(new_news_seg,words,wordvectors,ID_list,candidate_pool)

    with open('../data/news_vec_pool.pkl', 'wb') as s1:
        pickle.dump(news_vec_pool, s1)
    with open('../data/candidate_pool.pkl', 'wb') as s2:
        pickle.dump(candidate_pool, s2)
    
    