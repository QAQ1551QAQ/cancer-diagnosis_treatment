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
import keras_metrics as km  # pip install keras_metrics
from keras import metrics
from keras.models import load_model

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

def ml_focalloss_predict(feature, label, save_res):
        # scaler
        test_stdScaler = StandardScaler().fit_transform(feature)

        # one-hot label
        test_label = np_utils.to_categorical(label, num_classes=34)
        #print(test_label)

        # optim
        optim = gradient_descent.SGD(lr=0.9,decay=1e-5,momentum=0.9,nesterov=True)

        # model load
        model = load_model('../result/diagnosis/model_save/%s.h5'%save_res,compile = False)

        # compile             
        model.compile(loss=[utils.multi_category_focal_loss2(alpha=0.65, gamma=2)],
                optimizer= optim,
                metrics=['accuracy', metrics.Precision(), metrics.Recall(), km.f1_score()])
        # 测试数据有标签，可以输出模型测试的accuracy，Precision，Recall，f1_score
        score = model.evaluate(test_stdScaler,test_label,batch_size=64)
        print(score)
        # 测试数据无标签，可以输出模型预测的标签
        print('predict:', np.argmax(model.predict(test_stdScaler), axis=1))

if __name__ == '__main__':
    # get_data()
    data_feature = pd.read_csv('../data/n_feature_diagnosis.csv')
    data_origin = pd.read_csv('../data/data_diagnosis.csv')
    '''
        # 使用for循环, 对所有模型进行测试
    for i in ['all_1307','0.001_268', '0.0015_114', '0.002_46']:
        feature, label = get_data_RF(i, data_feature, data_origin)
        print("\nfeature:", i)
        save_res = 'res_%sfeature_mlp_focalLoss'%i
        ml_focalloss_predict(feature, label, save_res)
    '''
        # 对单个模型进行测试
    for i in ['all_1307']:
        feature, label = get_data_RF(i, data_feature, data_origin)
        print("\nfeature:", i)
        save_res = 'res_%sfeature_mlp_focalLoss'%i
        ml_focalloss_predict(feature, label, save_res)

