import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, GRU
from keras.layers import Embedding
from keras.layers import TimeDistributed, RepeatVector, Input, subtract, Lambda
from keras import backend as K
from keras.callbacks import EarlyStopping
import os
import logging

data_source_dir = '../data'

class UserModel:
    """

    Args: Temporary

    """
    def __init__(self, Config = None):
        self.logging = logging.getLogger(name=__name__)
        self.predict_model = None
        self.user_to_news_history  = None
        self.user_to_news_pos_vec  = None
        self.user_to_news_neg_vec  = None
        self.user_to_news_pos_id  = None
        self.user_to_news_neg_id  = None
        self.user_to_vec = None
        self.news_vec_dict = None
        self.config = {}

        if Config != None:
            self.name = Config.model_name
            self.config['output_dir'] = Config.Directory.model_dir
            self.config['user_to_news_pos_id_path']  = Config.Preprocessor.user_to_news_pos_id_path
            self.config['user_to_news_neg_id_path']  = Config.Preprocessor.user_to_news_neg_id_path
            self.config['user_to_news_pos_vec_path'] = Config.Preprocessor.user_to_news_pos_vec_path
            self.config['user_to_news_neg_vec_path'] = Config.Preprocessor.user_to_news_neg_vec_path
            self.config['news_vec_pool_path'] = Config.Pool.news_vec_pool_path
            self.config['user_vec_pool_path'] = Config.Pool.user_vec_pool_path
            self.config['user_model_path'] = Config.Preprocessor.user_model_path

    def load_news_history(self, pos_path=None, neg_path=None, by_id=False):
        user_size = 0
        
        if not pos_path:
            if by_id:
                pos_path=self.config['user_to_news_pos_id_path']
            else:
                pos_path=self.config['user_to_news_pos_vec_path']
                
        if not neg_path:
            if by_id:
                neg_path=self.config['user_to_news_neg_id_path']
            else:
                neg_path=self.config['user_to_news_neg_vec_path']
            
        
        self.logging.info('loading reading history...')

        with open(pos_path, 'rb') as fp:
            if by_id:
                self.user_to_news_pos_id = pickle.load(fp)
                user_size = len(self.user_to_news_pos_id)
            else:
                self.user_to_news_pos_vec = pickle.load(fp)
                user_size = len(self.user_to_news_pos_vec)

        with open(neg_path, 'rb') as fp:
            if by_id:
                self.user_to_news_neg_id = pickle.load(fp)
            else:
                self.user_to_news_neg_vec = pickle.load(fp)
                
        self.logging.info('- users: {}'.format(user_size))
        self.logging.info('- complete loading reading history...')

    def _news_dict_preprocess(self,dict_path=None):
        if dict_path == None:
            dict_path = self.config['news_vec_pool_path']
        self.logging.info('loading news_vec_pool')
        news_vec_pool = pd.read_pickle(dict_path)
        self.logging.info('building embedding matrix')
        valid_ids = []
        max_id = 0
        for doc_id in news_vec_pool.keys():
            try:
                doc_id = int(doc_id)
                valid_ids.append(doc_id)
                if doc_id > max_id:
                    max_id = doc_id
            except:
                pass
        embedding_matrix = np.zeros((max_id+1, len(news_vec_pool[str(max_id)])))
        for doc_id in valid_ids:
            embedding_vector = news_vec_pool[str(doc_id)]
            embedding_matrix[doc_id] = embedding_vector
        self.news_vec_dict = embedding_matrix
        self.logging.info(' - complete building embedding matrix')


    def _build_news_train(self, start=0, items=10, N=None, by_id=False):
        self.logging.info('building user-vectors...')
        self.logging.info('- news-threshold: {}'.format(N))

        if not N:
            N = start + items
        
        if by_id:
            pos_item_list = self.user_to_news_pos_id
            neg_item_list = self.user_to_news_neg_id
        else:
            pos_item_list = self.user_to_news_pos_vec
            neg_item_list = self.user_to_news_neg_vec
            
        X_pos, X_neg = [], []
        user_list = []
        for i,user_id in enumerate(pos_item_list): 
            pos_list = pos_item_list[user_id]
            neg_list = neg_item_list[user_id]
            if len(pos_list) >= N and len(neg_list) >= N:
                user_list.append(user_id)
                X_pos.append(pos_list[start:start+items][::-1])
                X_neg.append(neg_list[start:start+items][::-1])
        self.logging.info('- qualified users: {}'.format(len(user_list)))
        self.logging.info('- complete building news vectors list')

        return user_list,X_pos,X_neg

    def cos_sim(self, a, b):
        """Takes 2 vectors a, b and returns the cosine similarity according 
        to the definition of the dot product
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def l2_norm(self, x, axis=None):
        square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
        norm = K.sqrt(K.maximum(square_sum, K.epsilon()))
        return norm

    def pairwise_cos_sim(self, tensor):
        """
        t1 [batch x n x d] tensor of n rows with d dimensions
        t2 [batch x m x d] tensor of n rows with d dimensions

        returns:
        t12 [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
        """
        t1, t2 = tensor
        t1_mag = self.l2_norm(t1, axis=-1)
        t2_mag = self.l2_norm(t2, axis=-1)
        num = K.batch_dot(t1, K.permute_dimensions(t2, (0,2,1)))
        den = (t1_mag * K.permute_dimensions(t2_mag, (0,2,1)))
        t12 =  num / den
        return t12

    def _build_many_to_one_dict_model(self, shape, user_length=None, model_type='GRU', init_rnn_by_avg=False, neg_sampling=False, embedding_matrix=None):
        self.logging.info('******** model setting *********')
        self.logging.info('*****')
        self.logging.info('*****   model:{}'.format(model_type))
        self.logging.info('*****   init :{}'.format(init_rnn_by_avg))
        self.logging.info('*****   negs :{}'.format(neg_sampling))
        self.logging.info('*****')
        self.logging.info('********************************')
        
        doc_length = shape[1]
        doc_size   = embedding_matrix.shape[0]
        vector_dim = embedding_matrix.shape[1]
        
        if not user_length:
            user_length=doc_length

        input_pos_ids = Input(shape=(doc_length,),  dtype='int32', name="input_pos_ids")
        input_neg_ids = Input(shape=(doc_length,),  dtype='int32', name="input_neg_ids")
        input_user_ids = Input(shape=(user_length,),  dtype='int32', name="input_user_ids")
        
        e_layer = Embedding(doc_size,vector_dim, weights=[embedding_matrix], input_length=doc_length, trainable=False)
        
        input_pos = e_layer(input_pos_ids)
        input_neg = e_layer(input_neg_ids)
        input_user = e_layer(input_user_ids)
        input_init = Lambda(lambda x:K.mean(x,axis=1,keepdims=False),name='input_init')(input_user)        

        if model_type == 'GRU':
            rnn = GRU(units=vector_dim, input_shape=(user_length,vector_dim), name="rnn")
            if init_rnn_by_avg:
                rnn = rnn(inputs=input_user, initial_state=input_init)
            else:
                rnn = rnn(inputs=input_user)
        else: #LSTM
            rnn = LSTM(units=vector_dim, input_shape=(user_length,vector_dim), name="rnn")
            if init_rnn_by_avg:
                rnn = rnn(inputs=input_user, initial_state=[input_init,input_init])
            else:
                rnn = rnn(inputs=input_user)

        user_vec = Dense(vector_dim, name="user_vec")(rnn)
        user_vec_d3    = Lambda(lambda x: K.expand_dims(x, axis=1), name = "user_vec_3d")(user_vec)
        batch_cos_pos_3d  = Lambda(self.pairwise_cos_sim, name="batch_cos_pos_3d")([input_pos,user_vec_d3])        
        batch_cos_pos_2d  = Lambda(lambda x: K.squeeze(x, axis=-1), name="batch_cos_pos_2d")(batch_cos_pos_3d)

        if neg_sampling:
            batch_cos_neg_3d  = Lambda(self.pairwise_cos_sim, name="batch_cos_neg_3d")([input_neg,user_vec_d3])
            batch_cos_neg_2d  = Lambda(lambda x: K.squeeze(x, axis=-1), name="batch_cos_neg_2d")(batch_cos_neg_3d)
            batch_cos_diff_2d = subtract([batch_cos_pos_2d, batch_cos_neg_2d], name="batch_cos_diff_2d")
            output = Lambda(lambda x:1/(1 + K.exp(-x)), name="output")(batch_cos_diff_2d)
        else:
            output = Lambda(lambda x:1/(1 + K.exp(-x*2)), name="output")(batch_cos_pos_2d)
        model  = Model(inputs=[input_user_ids,input_pos_ids,input_neg_ids], outputs=output)
        model.compile(loss='mse', optimizer="adam")
        self.logging.info(model.summary())
        return model



    def _build_many_to_one_model(self, shape, user_length=None, model_type='GRU', init_rnn_by_avg=False, neg_sampling=False):
        self.logging.info('******** model setting *********')
        self.logging.info('*****')
        self.logging.info('*****   model:{}'.format(model_type))
        self.logging.info('*****   init :{}'.format(init_rnn_by_avg))
        self.logging.info('*****   negs :{}'.format(neg_sampling))
        self.logging.info('*****')
        self.logging.info('********************************')
        if not user_length:
            user_length=shape[1]
        input_pos = Input(shape=(shape[1],shape[2],), name="input_pos")
        input_neg = Input(shape=(shape[1],shape[2],), name="input_neg")
        input_user = Input(shape=(user_length,shape[2],), name="input_user")
        input_init = Lambda(lambda x:K.mean(x,axis=1,keepdims=False),name='input_init')(input_user)        
        if model_type == 'GRU':
            rnn = GRU(units=shape[2], input_shape=(user_length,shape[2]), name="rnn")
            if init_rnn_by_avg:
                rnn = rnn(inputs=input_user, initial_state=input_init)
            else:
                rnn = rnn(inputs=input_user)
        else: #LSTM
            rnn = LSTM(units=shape[2], input_shape=(user_length,shape[2]), name="rnn")
            if init_rnn_by_avg:
                rnn = rnn(inputs=input_user, initial_state=[input_init,input_init])
            else:
                rnn = rnn(inputs=input_user)
        
        user_vec = Dense(shape[2], name="user_vec")(rnn)
        user_vec_d3    = Lambda(lambda x: K.expand_dims(x, axis=1), name = "user_vec_3d")(user_vec)
        batch_cos_pos_3d  = Lambda(self.pairwise_cos_sim, name="batch_cos_pos_3d")([input_pos,user_vec_d3])        
        batch_cos_pos_2d  = Lambda(lambda x: K.squeeze(x, axis=-1), name="batch_cos_pos_2d")(batch_cos_pos_3d)

        if neg_sampling:
            batch_cos_neg_3d  = Lambda(self.pairwise_cos_sim, name="batch_cos_neg_3d")([input_neg,user_vec_d3])
            batch_cos_neg_2d  = Lambda(lambda x: K.squeeze(x, axis=-1), name="batch_cos_neg_2d")(batch_cos_neg_3d)
            batch_cos_diff_2d = subtract([batch_cos_pos_2d, batch_cos_neg_2d], name="batch_cos_diff_2d")
            output = Lambda(lambda x:1/(1 + K.exp(-x)), name="output")(batch_cos_diff_2d)
        else:
            output = Lambda(lambda x:1/(1 + K.exp(-x*2)), name="output")(batch_cos_pos_2d)
        model  = Model(inputs=[input_user,input_pos,input_neg], outputs=output)
        model.compile(loss='mse', optimizer="adam")
        self.logging.info(model.summary())
        return model
    
    def model_training_dict_version(self, start=0, items=10, user_length=10, N=None, model_type='GRU', init_rnn_by_avg=True, neg_sampling=True, epochs=20, batch_size=16, validation_split=0.1, patience=10, verbose=1, suffix_name=''):
        self._news_dict_preprocess()
        user_list, X_pos, X_neg = self._build_news_train(start=start, items=items, N=N, by_id=True)
        if user_length > items:
            print('illegal length')
            user_length = items
        X_pos = np.asarray(X_pos,dtype='int32')
        X_neg = np.asarray(X_neg,dtype='int32')
        X_user = X_pos[:,:user_length]        
        X_train = [X_user,X_pos,X_neg]
        Y_train = np.ones((X_pos.shape[0],X_pos.shape[1]))
        # 3. 開始訓練
        model = self._build_many_to_one_dict_model(X_pos.shape, 
                                                   user_length=user_length, 
                                                   model_type=model_type,
                                                   init_rnn_by_avg=init_rnn_by_avg, 
                                                   neg_sampling=neg_sampling, 
                                                   embedding_matrix=self.news_vec_dict)
        
        callback = EarlyStopping(monitor="loss", patience=patience, verbose=verbose, mode="auto")
        model.fit(X_train,Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[callback])
        # 需要的是模型訓練時的中間產物，user_vec，將 user_vec 層讀出
        layer_name = 'user_vec'
        user_vec_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        self.predict_model = user_vec_model
        # self.save_user_model()
        # self.load_user_model()
        X_user  = X_pos[:,-user_length:]
        X_train = [X_user,X_pos,X_neg]
        user_vec_output = self.predict_model.predict(X_train)
        # 將訓練得到的 user_vec 存起來
        user_dic = {}
        for i in range(len(user_vec_output)):
            user_dic[user_list[i]] = user_vec_output[i]
        self.user_to_vec = user_dic
        self.save_user_vec_pool(suffix_name=suffix_name)
    
    
    def model_training(self, start=0, items=10, user_length=10, N=None, model_type='GRU', init_rnn_by_avg=True, neg_sampling=True, epochs=20, batch_size=16, validation_split=0.1, patience=10, verbose=1, suffix_name=''):
        # 1. 讀入 positive 和 negative 的資料
        user_list, X_pos, X_neg = self._build_news_train(start=start, items=items, N=N)
        if user_length > items:
            print('illegal length')
            user_length = items
        X_pos = np.asarray(X_pos)
        X_neg = np.asarray(X_neg)
        X_user  = X_pos[:,:user_length,:]
        X_train = [X_user,X_pos,X_neg]
        #U_init = X_pos.mean(1)
        #X_train = [X_pos,X_neg,U_init]
        Y_train = np.ones((X_pos.shape[0],X_pos.shape[1]))
        # 3. 開始訓練
        model = self._build_many_to_one_model(X_pos.shape, user_length=user_length, model_type=model_type,init_rnn_by_avg=init_rnn_by_avg, neg_sampling=neg_sampling)
        callback = EarlyStopping(monitor="loss", patience=patience, verbose=verbose, mode="auto")
        model.fit(X_train,Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[callback])
        # 需要的是模型訓練時的中間產物，user_vec，將 user_vec 層讀出
        layer_name = 'user_vec'
        user_vec_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        self.predict_model = user_vec_model
        self.save_user_model()
        self.load_user_model()
        X_user  = X_pos[:,-user_length:,:]
        user_vec_output = self.predict_model.predict([X_user])
        # 將訓練得到的 user_vec 存起來
        user_dic = {}
        for i in range(len(user_vec_output)):
            user_dic[user_list[i]] = user_vec_output[i]
        self.user_to_vec = user_dic
        self.save_user_vec_pool(suffix_name=suffix_name)
    
    def build_user_vec_by_averaging(self,dic,items=10):
        self.logging.info('building user-vectors by averaging history')
        user_vecs = []
        user_list = []
        for i,user_id in enumerate(dic): 
            pos_list = dic[user_id]
            if len(pos_list) >= items:
                user_list.append(user_id)
                user_vecs.append(np.asarray(pos_list[:items]).mean(0).tolist())
        self.logging.info('- qualified users: {}'.format(len(user_list)))
        user_dic = dict(zip(user_list,user_vecs))
        self.logging.info('- complete building user-vectors by averaging')
        return user_dic


    def build_user_vec_by_pretrained_model(self,dic,items=10):
        self.logging.info('building user-vectors from history')
        X_pos = []
        user_list = []
        for i,user_id in enumerate(dic): 
            pos_list = dic[user_id]
            if len(pos_list) >= items:
                user_list.append(user_id)
                X_pos.append(pos_list[:items][::-1])
        self.logging.info('- qualified users: {}'.format(len(user_list)))
        X_pos = np.asarray(X_pos)
        user_vec_output = self.predict_model.predict([X_pos])
        user_dic = {}
        for i in range(len(user_vec_output)):
            user_dic[user_list[i]] = user_vec_output[i]
        self.logging.info('- complete building user-vectors')
        return user_dic

    def save_user_vec_pool(self, path=None, suffix_name=''):
        self.logging.info('saving user-vectors...')
        if not path:
            path = (self.config['user_vec_pool_path']+suffix_name)
        with open(path, 'wb') as fp:
            pickle.dump(self.user_to_vec,fp)
        self.logging.info('- complete saving user-vectors to "{}"'.format(path))

    def save_user_model(self, path=None):
        self.logging.info('saving user-model...')
        if not path:
            path = self.config['user_model_path']
        self.predict_model.save(path)
        self.logging.info('- complete saving user-model to "{}"'.format(path))

    def load_user_model(self, path=None):
        self.logging.info('loading user-model...')
        if not path:
            path = self.config['user_model_path']
        self.predict_model = load_model(path)
        self.logging.info('- complete loading user-model from "{}"'.format(path))

    def build_user_vec(self, news_vecs):
        self.logging.info('building user-vec...')
        news_vecs = np.asarray(news_vecs)
        user_vec_output = self.predict_model.predict([news_vecs])
        self.logging.info('- complete building user-vec...')
        return user_vec_output
