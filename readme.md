#### 环境要求
	Python 3.8.12
	keras 2.8.0
	
#### 文件
.
├── code                         
│   ├── dl_diagnosis_predict.py -- MLP-focalLoss模型，癌症诊断预测代码\n
│   ├── dl_focalLoss.py         -- 使用focalLoss 的MLP癌症诊断模型训练代码
│   ├── dl.py                   -- 使用crossentropy 的MLP癌症诊断模型训练代码
│   ├── dl_treatment_predict.py -- MLP模型，癌症治疗预测代码
│   ├── ml.py                   -- 癌症诊断模型训练代码，包含朴素贝叶斯、K最近邻、随机森林、支持向量机
│   ├── ml_treatment.py         -- 癌症治疗模型训练代码，包含朴素贝叶斯、K最近邻、随机森林、支持向量机、多层感知机
│   ├── __pycache__
│   └── utils.py
├── data
│   ├── data_diagnosis.csv      -- 癌症诊断数据集
│   ├── data_treatment.csv      -- 癌症治疗数据集
│   ├── n_feature_diagnosis.csv -- 癌症诊断模型训练特征维度
│   └── n_feature_treatment.csv -- 癌症治疗模型训练特征维度
├── readme.md
├── result
│   ├── diagnosis               -- 癌症诊断模型训练结果
│   └── treatment               -- 癌症治疗模型训练结果
└── result_display
    ├── code                    -- 结果展示代码
    ├── data                    -- 结果数据
    └── picture                 -- 结果图

10 directories, 12 files
