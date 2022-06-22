import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dispaly_plot(res_plt, tick_label, name):
    
    width = 0.42
    x = np.arange(0,8,2) # 控制x轴的个数

    fig = plt.figure(figsize = (14,8))
    plt.bar(x, res_plt.loc[:,'acc'],width, label='Accuracy',color="c",alpha=0.5)
    for i, j in zip(x, res_plt.loc[:,'acc']):
        plt.text(i+0.001,j+0.001, '%.3f'%j, ha='center')

    plt.bar(x+ width, res_plt.loc[:,'pre'], width, label='Precision',color="b", alpha=0.5)
    for i, j in zip(x+ width, res_plt.loc[:,'pre']):
        plt.text(i+0.001,j+0.001, '%.3f'%j, ha='center')

    plt.bar(x+ 2*width, res_plt.loc[:,'rec'], width,label='Recall', color="y",alpha=0.5)
    for i, j in zip(x+ 2*width, res_plt.loc[:,'rec']):
        plt.text(i+0.001,j+0.001, '%.3f'%j, ha='center')

    plt.bar(x+ 3*width, res_plt.loc[:,'f1'], width,label='F1-score',color="chocolate",alpha=0.7)
    for i, j in zip(x+ 3*width, res_plt.loc[:,'f1']):
        plt.text(i+0.001,j+0.001, '%.3f'%j, ha='center')

    plt.ylim(0.7,1.0)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper left', fontsize=15)
    plt.xticks(x+1.5*width,tick_label)
    plt.savefig('../picture/%s_plot.pdf'%name)
    # plt.show()


if __name__ == "__main__":
    # names = ['diagnosis_1307','diagnosis_268','diagnosis_114', 'diagnosis_46'] #plt.ylim(0.65,1.0) #x轴5个
    names = ['treatment_1310', 'treatment_241', 'treatment_120', 'treatment_39'] #plt.ylim(0.7,1.0) #x轴4个
    for name in names:
        path = '../data/%s.txt'%name
        data = pd.read_csv(path, encoding='utf-8')
        # print(data.head())
        res_plt = data[["acc", "pre", "rec", "f1"]]
        tick_label = data.index
        dispaly_plot(res_plt, tick_label, name)
    pass