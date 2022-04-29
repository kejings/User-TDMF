import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import mean_absolute_error as mae
import tensorflow as tf
from tensorflow.compat.v1.keras.regularizers import l2

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, multiply, concatenate, Lambda,Dot,Reshape,\
    Conv1D,Activation,BatchNormalization,add,MaxPooling1D
from tensorflow.keras.optimizers import Adam
from collections import defaultdict
from time import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from tdaintance import tda_get
import math
import heapq
import pandas as pd
import numpy as np
import random as rn




batch_size = 1024
factor_layers=[8,32]
reg_layers=[0,0.01]
learning_rate=0.003
num_epochs = 10
topK = 10
size = 30

################### data #############################

path='data/'
train = pd.read_csv(path+'train_ml100k.csv', header=0, index_col=None)
test = pd.read_csv(path+'test_ml100k.csv', header=0, index_col=None)
# train input
user = train['cat_user'].values
item = train['cat_item'].values
lable = train['label'].values

user_list, item_list = train['cat_user'].unique() ,train['cat_item'].unique()
# test input
user_ = test['cat_user'].values
item_ = test['cat_item'].values
label_ = test['label'].values

# side information input

patten_user = []
txt = open("ml100k_30.csv", "r")
for line in txt:
    patten_user.append(line.strip('\n').split(",")[0:size])
txt.close()

tda_train_users,tda_test_users = tda_get(patten_user,user, user_)

num_users = len(user_list)
num_items = len(item_list)

print('-------------------- data is ok ------------------------------')
print('users number is:', num_users,'items number is', num_items)
print('train is:',len(train),'test is:',len(test))
print('-------------------- load data done --------------------------')

def model(sd):
    os.environ['PYTHONHASHSEED'] = '0'
    tf.compat.v1.keras.backend.clear_session()
    tf.compat.v1.keras.backend.set_floatx('float64')

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(sd)
    np.random.seed(sd)
    rn.seed(sd)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


    def CNN(input_layer):
        '''
        :param input_layer: input
        :return:
        '''

        input = tf.expand_dims(input_layer, -1)

        x = input

        x = Conv1D(filters=factor_layers[0], kernel_size=3, strides=1, padding='same', activation='relu',kernel_regularizer=l2(reg_layers[0]),
                          kernel_initializer='normal', name='conv_1')(x)
        x = Conv1D(filters=factor_layers[0], kernel_size=3, strides=1, padding='same', activation='relu',
                   kernel_regularizer=l2(reg_layers[0]), kernel_initializer='normal', name='conv_2')(x)

        x = Flatten()(x)
        output_layer = x
        return output_layer


    def get_model(num_users,num_items,factor_layers,reg_layers):
        # input layer
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')
        user_tinput = Input(shape=(size, ), name='user_tinput')
        #print(user_tinput.shape)

        user_Embedding = Embedding(input_dim=num_users, output_dim=int(factor_layers[1]), name='user_embedding',
                                  embeddings_regularizer=l2(reg_layers[0]), input_length=1,
                                   embeddings_initializer='normal')
        item_Embedding = Embedding(input_dim=num_items, output_dim=int(factor_layers[1]), name='item_embedding',
                                   embeddings_regularizer=l2(reg_layers[0]), input_length=1,
                                   embeddings_initializer='normal')
        # get user new latent vector
        cov_user = CNN(user_tinput)
        # latent vectors
        user_latent = Flatten()(user_Embedding(user_input))
        item_latent = Flatten()(item_Embedding(item_input))
        user_new = concatenate([user_latent,cov_user])

        user_new = Dense(factor_layers[1])(user_new)
        user_new = BatchNormalization()(user_new)
        user_new = Activation('relu')(user_new)

        pred1 = multiply([user_new,item_latent])
        pred2 = multiply([user_latent,item_latent])
        lr_layer = concatenate([pred1, pred2])
        prediction = Dense(1, activation='sigmoid', name='prediction')(lr_layer)
        model_ = Model(inputs=[user_input, item_input,user_tinput], outputs=prediction)
        return model_
    model=get_model(num_users, num_items, factor_layers, reg_layers)
    model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')

    def evaluation(score, lable, n):
        '''
        :param score: prediction
        :param n: number of topK
        :return: average-p,average-r
        '''
        sum_precision = 0
        sum_recall = 0
        sum_ndcg = 0
        for user_no in range(num_users):
            user_score = score[user_no]
            user_lable = lable[user_no]
            d = defaultdict(list)
            for k, v in enumerate(user_score):
                d[k].append(v)

            topn_recommendation_index = heapq.nlargest(n, d, key=d.get)
            counter = 0
            dcg = 0
            true_positive = 0
            for rem in topn_recommendation_index:
                if user_lable[rem] == 1:
                    true_positive += 1
                    dcg += math.log(2) / math.log(counter + 2)  # discountlist[counter]
                counter += 1
            sum_recall += true_positive
            sum_precision += true_positive / n
            sum_ndcg += dcg
        average_precision = sum_precision / num_users
        average_recall = sum_recall / num_users
        ndcg = sum_ndcg / num_users
        return average_precision, average_recall, ndcg

    # Training model
    r = []
    T2, T3 = 0,0
    for epoch in range(num_epochs):

        t1 = time()
        # Generate training instances
        user_input, item_input, labels = user, item, lable

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input), np.array(tda_train_users)],  # input
                         np.array(labels), batch_size=batch_size, epochs=1, verbose=0)

        t2 = time()
        predictions = model.predict([np.array(user_), np.array(item_), np.array(tda_test_users)], batch_size=2048, verbose=0)
        roc_auc = roc_auc_score(label_, predictions)
        prec, reca, _ = precision_recall_curve(label_, predictions)
        aupr = auc(reca, prec)
        MAE = mae(label_,predictions)
        T2 += t2 - t1
        t3 = time() - t2
        T3 += t3

        predict = []
        test_lable = []

        for i in range(len(user_list)):
            pre = predictions[i:i + 101]
            lab = label_[i:i + 101]
            test_lable.append(lab)
            predict.append(pre)
        # print('predict',predict)
        average_precision, average_recall, ndcg = evaluation(predict, test_lable, topK)
        loss = hist.history['loss'][0]
        f1 = (2 * average_precision * average_recall) / (average_precision + average_recall)
        print(
            "Iteration %d [%.1f s]: Recall = %.4f, NDCG = %.4f, p = %.4f, F1 = %.4f, auc=%.4f, mae=%.4f, loss = %.4f[%.1f s]. "
            % (epoch, t2 - t1, average_recall, ndcg, average_precision, f1, roc_auc, MAE, loss, time() - t2))
        r.append([average_recall, average_precision, ndcg, f1, roc_auc, MAE, loss])
    recall = []
    for i in range(len(r)):
        recall.append(r[i][0])
    max_index = recall.index(max(recall))
    best_Recall, best_precision, best_NDCG, = r[max_index][0], r[max_index][1], r[max_index][2],
    best_F1, best_auc, best_mae, loss = r[max_index][3], r[max_index][4], r[max_index][5], r[max_index][6]

    print(f" best Recall = {best_Recall:.4f}, best NDCG = {best_NDCG:.4f}, best precision = {best_precision:.4f}., best F1 = {best_F1:.4f}")


    ranklist = [best_Recall, best_precision, best_NDCG, best_F1, best_auc, best_mae, topK, loss, sd, batch_size,
                factor_layers, learning_rate,T2,T3]

    with open('ml100k_result.csv', 'a') as f:
        #  f.write(names)
        for i in ranklist:
            f.write(f'{i},')
            # f.write('{},'.format(i))
        f.write('\n')
        f.close()


for sd in range(7,8):
    print(sd)
    model(sd)

