import pandas as pd
import numpy as np
import pickle
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, GRU
from keras.layers import TimeDistributed, RepeatVector, Input, subtract, Lambda
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import logging


##### Training Process
# data_source_dir = '../data'
# output_temp_dir = os.path.join(data_source_dir,'mid-product')
# user_to_news_pos_vec_path = os.path.join(output_temp_dir, 'user_to_news_pos_vec.pkl')
# user_to_news_neg_vec_path = os.path.join(output_temp_dir, 'user_to_news_neg_vec.pkl')
# user_vec_path = os.path.join(data_source_dir,'user_vec.pkl')

# user_model = UserModel()
# user_model.load_news_history(user_to_news_pos_vec_path, user_to_news_neg_vec_path)
# user_model.model_training()
# user_model.save_to(user_vec_path)

class UserModel:
    """

    Args: Temporary

    """
    def __init__(self):
        self.logging = logging.getLogger(name=__name__)
        self.predict_model = None
        self.user_to_news_history  = None
        self.user_to_news_pos_vec  = None
        self.user_to_news_neg_vec  = None
        self.user_to_vec = None

    def load_news_history(self, pos_path, neg_path):
        with open(pos_path, 'rb') as fp:
            self.user_to_news_pos_vec = pickle.load(fp)
        with open(neg_path, 'rb') as fp:
            self.user_to_news_neg_vec = pickle.load(fp)


    def buildNewsTrain(self):
        X_pos, X_neg = [], []
        user_list = []
        for i,user_id in enumerate(self.user_to_news_pos_vec): 
            pos_list = self.user_to_news_pos_vec[user_id]
            neg_list = self.user_to_news_neg_vec[user_id]
            if len(pos_list) >= 10 and len(neg_list) >= 10:
                user_list.append(user_id)
                X_pos.append(pos_list[0:9])
                X_neg.append(neg_list[0:9])
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

    def buildMany2OneFunctionalModel(self, shape):
        input_pos = Input(shape=(shape[1],shape[2],), name="input_pos")
        input_neg = Input(shape=(shape[1],shape[2],), name="input_neg")
        gru = GRU(output_dim=shape[2], input_length=shape[1], input_dim=shape[2], name="gru")(input_pos)
        user_vec = Dense(shape[2], name="user_vec")(gru)
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

    def model_training(self):
        # 1. 讀入 positive 和 negative 的資料
        user_list, X_pos, X_neg = self.buildNewsTrain()
        X_pos = np.asarray(X_pos)
        X_neg = np.asarray(X_neg)
        X_train = [X_pos,X_neg]
        Y_train = np.ones((X_pos.shape[0],X_pos.shape[1]))
        # 3. 開始訓練
        model = self.buildMany2OneFunctionalModel(X_pos.shape)
        callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
        model.fit(X_train,Y_train, epochs=20, batch_size=16, validation_split=0.1, callbacks=[callback])
        # 需要的是模型訓練時的中間產物，user_vec，將 user_vec 層讀出
        layer_name = 'user_vec'
        user_vec_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        self.predict_model = user_vec_model
        user_vec_output = self.predict_model.predict([X_pos, X_neg])
        # 將訓練得到的 user_vec 存起來
        user_dic = {}
        for i in range(len(user_vec_output)):
            user_dic[user_list[i]] = user_vec_output[i]
        self.user_to_vec = user_dic

    def save_to(self, path):
        with open(path, 'wb') as fp:
            pickle.dump(self.user_to_vec,fp)