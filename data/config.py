import os
class ColumnName:
    event_time = 'user_id'
    news_id    = 'event_time'
    user_id    = 'news_id'
    columns    = [event_time,news_id,user_id]

class TrainingEvent:
    _start  = '0427'
    _end    = '0428' # not included
    _prefix = 'event-2019'
    file_names = []
    for x in range(int(_start),int(_end)):
        file_name = _prefix + str(x).zfill(4) + '.pkl'
        file_names.append(file_name)
    file_paths = None

class CandidateEvent:
    _start  = '0428'
    _end    = '0429' # not included
    _prefix = 'event-2019'
    file_names = []
    for x in range(int(_start),int(_end)):
        file_name = _prefix + str(x).zfill(4) + '.pkl'
        file_names.append(file_name)
    file_paths = None

class Directory:
    model_name  = "{}-{}".format(TrainingEvent._start,TrainingEvent._end)
    data_dir     = 'data'
    vec_pool_dir = data_dir
    model_dir    = os.path.join(data_dir,model_name)

class Preprocessor:
    _user_to_news_history_name = 'user_to_news_history.pkl'
    _user_to_news_pos_vec_name = 'user_to_news_pos_vec.pkl'
    _user_to_news_pos_vec_name = 'user_to_news_pos_vec.pkl'
    _user_to_news_neg_vec_name = 'user_to_news_neg_vec.pkl'
    _user_to_news_pos_id_name  = 'user_to_news_pos_id.pkl'
    _user_to_news_neg_id_name  = 'user_to_news_neg_id.pkl'
    _user_model_name           = 'user_model.h5'

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

class Config:
    """

    Args: Temporary

    """
    model_name = Directory.model_name
    Directory = Directory
    CandidateEvent = CandidateEvent
    TrainingEvent  = TrainingEvent
    ColumnName  = ColumnName
    Preprocessor = Preprocessor
    Pool = Pool

    CandidateEvent.file_paths = [os.path.join(Directory.data_dir,file_name) for file_name in CandidateEvent.file_names]
    TrainingEvent.file_paths  = [os.path.join(Directory.data_dir,file_name) for file_name in TrainingEvent.file_names]