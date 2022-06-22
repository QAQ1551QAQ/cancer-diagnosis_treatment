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

# set random seed
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
test_size = 0.2

# classifiers
classifiers = [
    ('NaiveBayes', GaussianNB()),  # 朴素贝叶斯
    ('NearestNeighbors', KNeighborsClassifier()),  # K最近邻
    ('RandomForest', RandomForestClassifier(random_state=seed)),  # 随机森林
    ('SVM', SVC(random_state=seed)),  # 支持向量机
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
    return feature, label

def train_metric(feature, label, save_resPath):
    # split data
    train_feature, test_feature, train_label, test_label = train_test_split(
            feature, label, test_size=test_size, stratify=label, random_state=seed, shuffle=True)

    # scaler
    stdScaler = StandardScaler().fit(train_feature)
    train_stdScaler = stdScaler.transform(train_feature)
    test_stdScaler = stdScaler.transform(test_feature)

    # model train, metric
    with open('%s'%save_resPath, 'w', encoding='utf-8') as fw:
        for model_name, clf in classifiers:
                clf.fit(train_stdScaler, train_label)  # 训练
                # score = clf.score(test_stdScaler, test_label)  # 模型评分 acc
                # joblib.dump(clf,'%s.pkl'%name)  # save model
                predict_label = clf.predict(test_stdScaler)
                # print(classification_report(test_label, predict_label))
                # print(name, score)
                acc, pre, rec, f1 = utils.metrics_report(test_label, predict_label, average='macro')
                fw.write(('%s,{},{},{},{}\n'%model_name).format(acc, pre, rec, f1))


if __name__ == '__main__':
    # get_data()
    data_feature = pd.read_csv('../data/n_feature_diagnosis.csv')
    data_origin = pd.read_csv('../data/data_diagnosis.csv')

    for i in ['all_1307','0.001_268', '0.0015_114', '0.002_46']:
        feature, label = get_data_RF(i, data_feature, data_origin)
        print("\nfeature:", i)
        save_resPath = '../result/diagnosis/res_%sfeature_ml.txt'%i
        train_metric(feature, label, save_resPath)
