import pandas as pd
import numpy as np
import pickle
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, GRU
from keras.layers import TimeDistributed, RepeatVector, Input, subtract, Lambda
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import logging

data_source_dir = '../data'

class UserModel:
    """

    Args: Temporary

    """
    def __init__(self, dir='../data', name=None):
        if not name:
            name = time.asctime(time.localtime(time.time()))
        self.name = name
        self.config = {}
        self.config['output_dir'] = os.path.join(dir, name)
        self.config['user_to_news_pos_vec_path'] = os.path.join(self.config['output_dir'], 'user_to_news_pos_vec.pkl')
        self.config['user_to_news_neg_vec_path'] = os.path.join(self.config['output_dir'], 'user_to_news_neg_vec.pkl')
        self.config['user_model_path'] = os.path.join(self.config['output_dir'], 'user_model.pkl')
        self.config['user_vec_pool_path'] = os.path.join(self.config['output_dir'], 'user_vec_pool.pkl')

        self.logging = logging.getLogger(name=__name__)
        self.predict_model = None
        self.user_to_news_history  = None
        self.user_to_news_pos_vec  = None
        self.user_to_news_neg_vec  = None
        self.user_to_vec = None

    def load_news_history(self, pos_path=None, neg_path=None):
        if not pos_path:
            pos_path=self.config['user_to_news_pos_vec_path']
        if not neg_path:
            neg_path=self.config['user_to_news_neg_vec_path']
        
        self.logging.info('loading reading history...')
        with open(pos_path, 'rb') as fp:
            self.user_to_news_pos_vec = pickle.load(fp)
        with open(neg_path, 'rb') as fp:
            self.user_to_news_neg_vec = pickle.load(fp)
        self.logging.info('- users: {}'.format(len(self.user_to_news_neg_vec)))
        self.logging.info('- complete loading reading history...')

    def _build_news_train(self, start=0, items=10, N=None):
        self.logging.info('building user-vectors...')
        self.logging.info('- news-threshold: {}'.format(N))

        if not N:
            N = start + items

        X_pos, X_neg = [], []
        user_list = []
        for i,user_id in enumerate(self.user_to_news_pos_vec): 
            pos_list = self.user_to_news_pos_vec[user_id]
            neg_list = self.user_to_news_neg_vec[user_id]
            if len(pos_list) >= N and len(neg_list) >= N:
                user_list.append(user_id)
                X_pos.append(pos_list[start:start+items])
                X_neg.append(neg_list[start:start+items])
        self.logging.info('- qualified users: {}'.format(len(user_list)))
        self.logging.info('- complete building user-vectors')

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

    def _build_many_to_one_model(self, shape, model_type='GRU'):
        input_pos = Input(shape=(shape[1],shape[2],), name="input_pos")
        input_neg = Input(shape=(shape[1],shape[2],), name="input_neg")
        rnn = None

        if model_type == 'GRU':
            rnn = GRU(units=shape[2], input_shape=(shape[1],shape[2]), name="rnn")(input_pos)
        else: #LSTM
            rnn = LSTM(units=shape[2], input_shape=(shape[1],shape[2]), name="rnn")(input_pos)
        
        user_vec = Dense(shape[2], name="user_vec")(rnn)
        user_vec_d3    = Lambda(lambda x: K.expand_dims(x, axis=1), name = "user_vec_3d")(user_vec)
        batch_cos_pos_3d  = Lambda(self.pairwise_cos_sim, name="batch_cos_pos_3d")([input_pos,user_vec_d3])
        batch_cos_neg_3d  = Lambda(self.pairwise_cos_sim, name="batch_cos_neg_3d")([input_neg,user_vec_d3])
        
        batch_cos_pos_2d  = Lambda(lambda x: K.squeeze(x, axis=-1), name="batch_cos_pos_2d")(batch_cos_pos_3d)
        batch_cos_neg_2d  = Lambda(lambda x: K.squeeze(x, axis=-1), name="batch_cos_neg_2d")(batch_cos_neg_3d)
        batch_cos_diff_2d = subtract([batch_cos_pos_2d, batch_cos_neg_2d], name="batch_cos_diff_2d")

        output = Lambda(lambda x:1/(1 + K.exp(-x)), name="output")(batch_cos_diff_2d)
        # output = Dense(shape[1], activation='sigmoid', name="output")(batch_dot_diff)
        model  = Model(inputs=[input_pos,input_neg], outputs=output)
        model.compile(loss='mse', optimizer="adam")
        return model

    def model_training(self, start=0, items=10, N=None, model_type='GRU', epochs=20, batch_size=16, validation_split=0.1, patience=10, verbose=1):
        # 1. 讀入 positive 和 negative 的資料
        user_list, X_pos, X_neg = self._build_news_train(start=start, items=items, N=N)
        X_pos = np.asarray(X_pos)
        X_neg = np.asarray(X_neg)
        X_train = [X_pos,X_neg]
        Y_train = np.ones((X_pos.shape[0],X_pos.shape[1]))
        # 3. 開始訓練
        model = self._build_many_to_one_model(X_pos.shape, model_type=model_type)
        callback = EarlyStopping(monitor="loss", patience=patience, verbose=verbose, mode="auto")
        model.fit(X_train,Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[callback])
        # 需要的是模型訓練時的中間產物，user_vec，將 user_vec 層讀出
        layer_name = 'user_vec'
        user_vec_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        self.predict_model = user_vec_model
        self.save_user_model()
        self.load_user_model()
        user_vec_output = self.predict_model.predict([X_pos])
        # 將訓練得到的 user_vec 存起來
        user_dic = {}
        for i in range(len(user_vec_output)):
            user_dic[user_list[i]] = user_vec_output[i]
        self.user_to_vec = user_dic
        self.save_user_vec_pool()
    
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
                X_pos.append(pos_list[:items])
        self.logging.info('- qualified users: {}'.format(len(user_list)))
        X_pos = np.asarray(X_pos)
        user_vec_output = self.predict_model.predict([X_pos])
        user_dic = {}
        for i in range(len(user_vec_output)):
            user_dic[user_list[i]] = user_vec_output[i]
        self.logging.info('- complete building user-vectors')
        return user_dic

    def save_user_vec_pool(self, path=None):
        self.logging.info('saving user-vectors...')
        if not path:
            path = self.config['user_vec_pool_path']
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
