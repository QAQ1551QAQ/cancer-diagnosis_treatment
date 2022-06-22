from cProfile import label
import matplotlib.pyplot as plt           #用 plot()函数绘制折线图
import numpy as np

x = ['46','114','268','1307']
# accuracy
pl = plt.figure(figsize=(8,8), dpi=100)
ax1 = pl.add_subplot(2,2,1)
mlp_acc=[0.876 ,0.905 ,0.921  ,0.933 ]
mlp_focalLoss_acc = [0.877, 0.903, 0.916, 0.941]
ax1.plot(x, mlp_acc,color='r',marker='o',linestyle='-.',alpha=0.5, label='MLP')
ax1.plot(x, mlp_focalLoss_acc,color='b',marker='o',linestyle='-.',alpha=0.5, label='MLP-focalLoss')
plt.title('Accuracy contrast')
plt.xlabel('number of dimensions')
plt.ylabel('accuracy')
plt.legend()
plt.tight_layout(1.3)

# precision
ax2 = pl.add_subplot(2,2,2)
mlp_pre=[0.800, 0.886, 0.892, 0.886]
mlp_focalLoss_pre = [0.804, 0.876, 0.875, 0.911]
ax2.plot(x, mlp_pre,color='r',marker='o',linestyle='-.',alpha=0.5, label='MLP')
ax2.plot(x, mlp_focalLoss_pre,color='b',marker='o',linestyle='-.',alpha=0.5, label='MLP-focalLoss')
plt.title('Precision contrast')
plt.xlabel('number of dimensions')
plt.ylabel('precision')
plt.legend()
plt.tight_layout(1.3)

# Recall
ax3 = pl.add_subplot(2,2,3)
mlp_rec=[0.802, 0.860, 0.877, 0.880]
mlp_focalLoss_rec = [0.806, 0.859, 0.858, 0.907]
ax3.plot(x, mlp_rec,color='r',marker='o',linestyle='-.',alpha=0.5, label='MLP')
ax3.plot(x, mlp_focalLoss_rec,color='b',marker='o',linestyle='-.',alpha=0.5, label='MLP-focalLoss')
plt.title('Recall contrast')
plt.xlabel('number of dimensions')
plt.ylabel('recall')
plt.legend()
plt.tight_layout(1.3)

# F1-score
ax4 = pl.add_subplot(2,2,4)
mlp_f1=[0.797, 0.865, 0.881, 0.881]
mlp_focalLoss_f1 = [0.801, 0.862, 0.863, 0.907]
ax4.plot(x, mlp_f1,color='r',marker='o',linestyle='-.',alpha=0.5, label='MLP')
ax4.plot(x, mlp_focalLoss_f1,color='b',marker='o',linestyle='-.',alpha=0.5, label='MLP-focalLoss')
plt.title('F1-score contrast')
plt.xlabel('number of dimensions')
plt.ylabel('f1-score')
plt.legend()
plt.tight_layout(1.3)
 
plt.savefig('../picture/MLP-MLP_focalLoss_contrast.pdf')
plt.show()