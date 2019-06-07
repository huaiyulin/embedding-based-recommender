import os
from datetime import datetime, timedelta
import json
import argparse

with open('training_config.json','r') as reader:
    training_config = json.loads(reader.read())

begin_t = training_config['training_event']['begin']
end_t = training_config['training_event']['end']

begin_c = training_config['candidate_event']['begin']
end_c = training_config['candidate_event']['end']

news_meta_path = os.path.join('../data',training_config['news_meta_name'])


def get_event_date(begin_day,end_day):
    bd = begin_day.split(',') 
    bd = [int(x) for x in bd]
    ed = end_day.split(',') 
    ed = [int(x) for x in ed]
    begin_date = datetime(bd[0],bd[1],bd[2]).date()
    end_date = datetime(ed[0],ed[1],ed[2]).date()
    eventdate = []
    a_day = timedelta(days=1)
    while begin_date <= end_date:
        eventdate.append('event-' + str(begin_date) + '.pkl')
        begin_date += a_day
    return eventdate

class TrainingEvent:
    bd = begin_t.split(',') 
    ed = end_t.split(',') 
    _start = bd[1].zfill(2)+bd[2].zfill(2)
    _end = ed[1].zfill(2)+ed[2].zfill(2)
    file_names = get_event_date(begin_t,end_t)
    file_paths = None

class CandidateEvent:
    file_names = get_event_date(begin_c,end_c)
    file_paths = None

class ColumnName:
    event_time = 'event_time'
    news_id    = 'news_id'
    user_id    = 'user_id'
    columns    = [event_time,news_id,user_id]

class Directory:
    model_name  = "{}-{}".format(TrainingEvent._start,TrainingEvent._end)
    data_dir     = '../data'
    vec_pool_dir = data_dir
    model_dir    = os.path.join(data_dir,model_name)

class Preprocessor:
    _user_to_news_history_name = 'user_to_news_history.pkl'
    _user_to_news_pos_vec_name = 'user_to_news_pos_vec.pkl'
    _user_to_news_pos_vec_name = 'user_to_news_pos_vec.pkl'
    _user_to_news_neg_vec_name = 'user_to_news_neg_vec.pkl'
    _user_to_news_pos_id_name  = 'user_to_news_pos_id.pkl'
    _user_to_news_neg_id_name  = 'user_to_news_neg_id.pkl'
    _user_model_name           = 'user_model_id.h5'

    model_dir = Directory.model_dir
    user_to_news_history_path = os.path.join(model_dir,_user_to_news_history_name)
    user_to_news_pos_vec_path = os.path.join(model_dir,_user_to_news_pos_vec_name)
    user_to_news_neg_vec_path = os.path.join(model_dir,_user_to_news_neg_vec_name)
    user_to_news_pos_id_path = os.path.join(model_dir,_user_to_news_pos_id_name)
    user_to_news_neg_id_path = os.path.join(model_dir,_user_to_news_neg_id_name)
    user_model_path          = os.path.join(model_dir,_user_model_name)
    

class Pool:
    _news_vec_pool_name  = 'news_vec_pool.pkl'
    _user_vec_pool_name  = 'user_vec_pool.pkl'
    _candidate_pool_name = 'candidate_pool.pkl'

    news_vec_pool_path  = os.path.join(Directory.vec_pool_dir,_news_vec_pool_name)
    user_vec_pool_path  = os.path.join(Directory.vec_pool_dir,_user_vec_pool_name)
    candidate_pool_path = os.path.join(Directory.vec_pool_dir,_candidate_pool_name)

class Recommender:
    _annoy_indexer_name    = 'annoy_indexer.ann'
    annoy_indexer_path    = os.path.join(Directory.vec_pool_dir,_annoy_indexer_name)

class Config:
    """

    Args: Temporary

    """
    model_name = Directory.model_name
    Directory = Directory
    CandidateEvent = CandidateEvent
    TrainingEvent  = TrainingEvent
    ColumnName   = ColumnName
    Preprocessor = Preprocessor
    Pool = Pool
    Recommender  = Recommender
    CandidateEvent.file_paths = [os.path.join(Directory.data_dir,file_name) for file_name in CandidateEvent.file_names]
    TrainingEvent.file_paths  = [os.path.join(Directory.data_dir,file_name) for file_name in TrainingEvent.file_names]



def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-t_b", dest="training_begin", type=str, help="event begin day, ex. 2019,4,28")
    parser.add_argument("-t_e", dest="training_end", type=str, help="event end day, ex. 2019,4,29")
    parser.add_argument("-c_b", dest="candidate_begin", type=str, help="candidate begin day, ex. 2019,4,29")
    parser.add_argument("-c_e", dest="candidate_end", type=str, help="candidate end day, ex. 2019,4,30")
    parser.add_argument("-meta_name", dest="news_meta_name", type=str, help="news meta path, ex. news_meta_2019-05-01.json")
    return parser

def read_json():
    with open('training_config.json','r') as reader:
        training_config = json.loads(reader.read())
    return training_config

def save_json(file_to_save):
    with open('training_config.json','w') as output_path:
        json.dump(file_to_save,output_path)


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    training_begin = args.training_begin
    training_end = args.training_end
    candidate_begin = args.candidate_begin
    candidate_end = args.candidate_end
    news_meta_name = args.news_meta_name

    training_config = read_json()

    if training_begin:
        training_config['training_event']['begin'] = training_begin
    if training_end:
        training_config['training_event']['end'] = training_end
    if candidate_begin:
        training_config['candidate_event']['begin'] = candidate_begin
    if candidate_end:
        training_config['candidate_event']['end'] = candidate_end
    if news_meta_name:
        training_config['news_meta_name'] = news_meta_name

    save_json(training_config)

if __name__ == '__main__':
    main()