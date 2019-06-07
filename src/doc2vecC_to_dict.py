import pandas as pd
import pickle

def get_doc2vecC_result(ID_list):
    docvectors_vect = pd.read_csv('../data/docvectors.txt', sep=" ", header=None)
    docvectors_vect = docvectors_vect.drop([docvectors_vect.shape[1]-1], axis=1) # delete -1 col
    docvectors_vect = docvectors_vect.drop([docvectors_vect.shape[0]-1]) # delete -1 row
    docvectors_vec = docvectors_vect.as_matrix()
    news_vec_dC = {}
    
    for i, ID in enumerate(ID_list):
        news_vec_dC[ID] = docvectors_vec[i]
        
    return news_vec_dC

if __name__ == '__main__':
    
    with open('../data/ID_list.pkl', 'rb') as haa:
        ID_list = pickle.load(haa)
    print("get_doc2vecC_result")    
    news_vec_dC = get_doc2vecC_result(ID_list)
    print("dump to news_vec_pool.pkl") 
    with open('../data/news_vec_pool.pkl', 'wb') as hc:
        pickle.dump(news_vec_dC, hc)

    

            