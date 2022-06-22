from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,RepeatVector,Bidirectional,LSTM
from keras.utils import np_utils
from keras.optimizer_v2 import adam,adamax,nadam,gradient_descent
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import random
from sklearn.preprocessing import StandardScaler
import utils

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
num_classes=34
test_size = 0.2

# load data
def get_data_RF(i, data_feature, data_origin):
    filter_feature_tmp = data_feature['%s'%i] # 分别取不同维度的特征
    filter_feature = filter_feature_tmp[filter_feature_tmp.notna()]
    filename = pd.Series(['filename'])
    filter_feature = filename.append(filter_feature).reset_index(drop=True)
    # 从原数据中获取带有标签的数据，返回相应特征和标签
    data_final = data_origin.loc[:, filter_feature]
    # print(data_final.shape)
    feature = data_final.iloc[:, 1:]
    label = data_final.iloc[:, 0].astype('int64')
    return feature, label

def train_metric(feature, label, save_resPath):
        # split data
        train_feature, test_feature, train_label, test_label = train_test_split(
                feature, label, test_size=test_size, stratify=label, random_state=seed, shuffle=True)

        # scaler
        stdScaler = StandardScaler().fit(train_feature)
        train_stdScaler = stdScaler.transform(train_feature)
        test_stdScaler = stdScaler.transform(test_feature)


        # one-hot label
        train_label = np_utils.to_categorical(train_label, num_classes=num_classes)
        test_label = np_utils.to_categorical(test_label, num_classes=num_classes)

        # model
        model = Sequential()
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(num_classes,activation='softmax'))
        
        # optim
        #optim = gradient_descent.SGD(lr=0.009,decay=1e-5,momentum=0.9,nesterov=True)
        optim = adam.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        # compile             
        model.compile(loss='categorical_crossentropy',
                optimizer= optim,
                metrics=['accuracy'])

        # fit  
        model.fit(train_stdScaler, train_label,
                epochs=25,
                batch_size=64)

        # metric
        # score = model.evaluate(test_stdScaler,test_label,batch_size=64)
        test_label = np.argmax(test_label, axis=1)
        predict_label = np.argmax(model.predict(test_stdScaler), axis=1)
        acc, pre, rec, f1 = utils.metrics_report(test_label, predict_label, average='macro')
        # save result
        with open('%s'%save_resPath, 'w', encoding='utf-8') as fw:
                fw.write(('%s,{},{},{},{}\n'%'mlp').format(acc, pre, rec, f1))

if __name__ == '__main__':
    # get_data()
    data_feature = pd.read_csv('../data/n_feature_diagnosis.csv')
    data_origin = pd.read_csv('../data/data_diagnosis.csv')

    for i in ['all_1307','0.001_268', '0.0015_114', '0.002_46']:
        feature, label = get_data_RF(i, data_feature, data_origin)
        print("\nfeature:", i)
        save_resPath = '../result/diagnosis/res_%sfeature_mlp.txt'%i
        train_metric(feature, label, save_resPath)
