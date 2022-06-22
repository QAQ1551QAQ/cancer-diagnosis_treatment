from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score
import numpy as np
import pandas as pd
import random, os
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.neural_network import MLPClassifier  # 多层感知机
from sklearn.neighbors import KNeighborsClassifier  # K最近邻
from sklearn.svm import SVC  # 支持向量机
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB  # 高斯朴素贝叶斯
import joblib
import utils
from imblearn.over_sampling import SMOTE 

# set random seed
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
test_size = 0.2

# classifiers
classifiers = [
    ('MLP', MLPClassifier(random_state=seed)) # 多层感知机
]

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
    # over_sampling
    # ros = SMOTE (random_state=seed)
    # feature, label= ros.fit_resample(feature, label)  #数据源
    return feature, label

def mlp_predict(test_feature, test_label, model_Path):

    stdScaler = StandardScaler()
    test_stdScaler = stdScaler.fit_transform(test_feature)
    clf=joblib.load('%s'%model_Path) 
    predict_label = clf.predict(test_stdScaler)
    print("predict_label:\n",predict_label)
    # print("test_label:\n",test_label)
    # print(classification_report(test_label, predict_label))
    print('acc: ',clf.score(test_stdScaler, test_label))


if __name__ == '__main__':
    # get_data()
    data_feature = pd.read_csv('../data/n_feature_treatment.csv')
    data_origin = pd.read_csv('../data/data_treatment.csv')

    '''
        # 使用for循环, 对所有模型进行测试
        for i in ['all_1310','0.001_241','0.0014_120','0.002_39']:
            feature, label = get_data_RF(i, data_feature, data_origin)
            print("\nfeature:", i)
            model_Path = '../result/treatment/model_save/mlp_%s.pkl'%i
            mlp_predict(feature, label, model_Path)
    '''
        # 对单个模型进行测试
    for i in ['all_1310']:
        feature, label = get_data_RF(i, data_feature, data_origin)
        print("\nfeature:", i)
        model_Path = '../result/treatment/model_save/mlp_%s.pkl'%i
        mlp_predict(feature, label, model_Path)

