import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# sns.set_context('talk')

# sns.set()
# sns.set_style('whitegrid')
sns.set_style('ticks')

acc_array_i_p_0_1 = np.array([77.333, 77.667, 78.0])
acc_array_i_p_0_5 = np.array([78.333, 78.167, 77.833])
acc_array_i_p_1_0 = np.array([78.0, 77.667, 77.333])
acc_array_i_p_2_0 = np.array([77.167, 77.167, 77.667])

acc_array_p_i_0_1 = np.array([92.667, 93.0, 92.833])
acc_array_p_i_0_5 = np.array([92.833, 93.333, 93.167])
acc_array_p_i_1_0 = np.array([93.333, 93.333, 93.667])
acc_array_p_i_2_0 = np.array([93.167, 92.833, 93.0])

acc_df_i_p_0_1 = pd.DataFrame({'acc': acc_array_i_p_0_1, 'gamma': 0.1})
acc_df_i_p_0_5 = pd.DataFrame({'acc': acc_array_i_p_0_5, 'gamma': 0.5})
acc_df_i_p_1_0 = pd.DataFrame({'acc': acc_array_i_p_1_0, 'gamma': 1.0})
acc_df_i_p_2_0 = pd.DataFrame({'acc': acc_array_i_p_2_0, 'gamma': 2.0})

acc_df_p_i_0_1 = pd.DataFrame({'acc': acc_array_p_i_0_1, 'gamma': 0.1})
acc_df_p_i_0_5 = pd.DataFrame({'acc': acc_array_p_i_0_5, 'gamma': 0.5})
acc_df_p_i_1_0 = pd.DataFrame({'acc': acc_array_p_i_1_0, 'gamma': 1.0})
acc_df_p_i_2_0 = pd.DataFrame({'acc': acc_array_p_i_2_0, 'gamma': 2.0})

acc_i_p_data = pd.concat([
    acc_df_i_p_0_1,
    acc_df_i_p_0_5,
    acc_df_i_p_1_0,
    acc_df_i_p_2_0
])
acc_p_i_data = pd.concat([
    acc_df_p_i_0_1,
    acc_df_p_i_0_5,
    acc_df_p_i_1_0,
    acc_df_p_i_2_0
])

f = plt.figure(figsize=(3, 2))
ax = sns.lineplot(
    data=acc_i_p_data,
    x='gamma',
    y='acc',
    ci='sd'
    # err_style='bars'
)

# ax.set_xlabel(r'$\gamma$')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel('Accuracy (%)')
ax.set(ylim=(70, 80))
# ax.set(xticks=([0, 1, 2, 3], [0.1, 0.5, 1.0, 2.0]))
ax.set_xticks([0.1, 0.5, 1.0, 2.0])
# ax.grid(axis='x', visible=False)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles[:], labels=labels[:])
# sns.despine(left=True)
sns.despine()

# plt.show()

f.savefig("sensitivity_i_p.pdf", bbox_inches='tight')

f = plt.figure(figsize=(3, 2))
ax = sns.lineplot(
    data=acc_p_i_data,
    x='gamma',
    y='acc',
    ci='sd'
    # err_style='bars'
)

# ax.set_xlabel(r'$\gamma$')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel('Accuracy (%)')
ax.set(ylim=(86, 96))
# ax.set(xticks=([0, 1, 2, 3], [0.1, 0.5, 1.0, 2.0]))
ax.set_xticks([0.1, 0.5, 1.0, 2.0])
# ax.grid(axis='x', visible=False)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles[:], labels=labels[:])
# sns.despine(left=True)
sns.despine()

plt.show()

f.savefig("sensitivity_p_i.pdf", bbox_inches='tight')
