import json
import jieba
import re
import pickle
import io
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-i", dest="meta_data", type=str, help="meta data for taining doc2vecC")
args = parser.parse_args()

def load_news(path):
    news = []
    ID_list = []
    corpus = io.open(path, 'r',encoding='utf-8')
    for line in corpus.readlines():
        post = json.loads(line)
        title = post['title']
        content = post['content']
        news.append((title,content))
        ID = post['doc_id']
        ID_list.append(ID)
    return news,ID_list

def remove_html_tag(text):
    return re.sub(r'</?\w+[^>]*>\"', ' ', text)

def word_token(sent):
    return jieba.cut(sent, cut_all=False)

def get_txt_IDList(NEWS):  
    jieba.set_dictionary('dict.txt.big')
    stop_words = io.open('stops.txt', 'r',encoding='utf-8').read().split('\n')
    news, ID_list = load_news(NEWS)   
    title, content = zip(*news)
    content = list(map(remove_html_tag, content))
#    title = list(map(remove_html_tag, title))
#    news = list(map(remove_html_tag, news))
    news_seg_set = []
    for i, n in enumerate(content):
        seg_list = [x for x in word_token(n) if x not in stop_words]
        item = " ".join(seg_list)
        news_seg_set.append(item)
    return ID_list, news_seg_set

if __name__ == '__main__':
    NEWS = '../data/'+ args.meta_data
    ID_list, alldata_txt = get_txt_IDList(NEWS)
    
    with open('../data/ID_list.pkl', 'wb') as f:
        pickle.dump(ID_list, f)

    print("write txt")    
    with open("../data/alldata.txt", "w",encoding='utf-8') as text_file:
        for news_seg in alldata_txt:
            text_file.write("%s\n" % news_seg)
    
    
    
    
    
    
