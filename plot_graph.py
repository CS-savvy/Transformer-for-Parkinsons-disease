from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import matplotlib

font = {'family' : 'normal',
        'size'   : 6}

matplotlib.rc('font', **font)

xls = pd.ExcelFile('plots/metric/Experiment-profile.xlsx')
df = pd.read_excel(xls, 'No of Features', header=[0, 1])

df_trans = df['Transformer']
df_mlp = df['MLP']
df_p = df['Pivot']
df_xg = df['XGboost']

# df = pd.read_csv("plots/metric/Random-Init.csv", header=[0, 1])

fig = plt.figure(figsize=(6, 3))

# fig.suptitle("Comparision")

ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

# col = 'No of Features'
# col = df_p['Stack']
col = list(range(1, 11))

# ax1.plot(df[col], df['K-Fold Avg ROC-AUC'], label='K-Fold Avg ROC-AUC')
# ax1.plot(df[col], df['Test ROC-AUC'], label='Test ROC-AUC')
# ax1.plot(df[col], df['Test F1'], label='Test F1')
# ax1.plot(df[col], df['Test Precision'], label='Test Precision')
# ax1.plot(df[col], df['Test Recall'], label='Test Recall')
# ax1.plot(df[col], df['Test Accuracy'], label='Test Accuracy')
# ax1.legend(loc='lower right')

ax1.title.set_text('K-Fold Avg ROC-AUC')
ax1.plot(col, df_trans['K-Fold Avg ROC-AUC'], label='Transformer')
ax1.plot(col, df_mlp['K-Fold Avg ROC-AUC'], label='MLP')
ax1.plot(col, df_xg['K-Fold Avg ROC-AUC'], label='XgBoost')
ax1.legend()

ax2.title.set_text('Test ROC-AUC')
ax2.plot(col, df_trans['Test ROC-AUC'], label='Transformer')
ax2.plot(col, df_mlp['Test ROC-AUC'], label='MLP')
ax2.plot(col, df_xg['Test ROC-AUC'], label='XgBoost')
ax2.legend()

ax3.title.set_text('Test F1')
ax3.plot(col, df_trans['Test F1'], label='Transformer')
ax3.plot(col, df_mlp['Test F1'], label='MLP')
ax3.plot(col, df_xg['Test F1'], label='XgBoost')
ax3.legend()

ax4.title.set_text('Test Precision')
ax4.plot(col, df_trans['Test Precision'], label='Transformer')
ax4.plot(col, df_mlp['Test Precision'], label='MLP')
ax4.plot(col, df_xg['Test Precision'], label='XgBoost')
ax4.legend()

ax5.title.set_text('Test Recall')
ax5.plot(col, df_trans['Test Recall'], label='Transformer')
ax5.plot(col, df_mlp['Test Recall'], label='MLP')
ax5.plot(col, df_xg['Test Recall'], label='XgBoost')
ax5.legend()

ax6.title.set_text('Test Accuracy')
ax6.plot(col, df_trans['Test Accuracy'], label='Transformer')
ax6.plot(col, df_mlp['Test Accuracy'], label='MLP')
ax6.plot(col, df_xg['Test Accuracy'], label='XgBoost')
ax6.legend()

# plt.show()

plt.savefig("plots/metric/plots_v2/stack.pdf", dpi=(1000), bbox_inches='tight', format='pdf')