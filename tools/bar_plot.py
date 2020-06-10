import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# sns.set_context('paper')

# sns.set()
sns.set_style('ticks')

error_array_jcl_src = np.array([0.034, 0.035, 0.034])
error_array_jcl_src /= 100.
error_array_jcl_tgt = np.array([13.461, 13.344, 13.405])
error_array_jcl_tgt /= 100.
error_array_jcl_avg = (error_array_jcl_src + error_array_jcl_tgt) / 2.0

error_array_dann_src = np.array([0.466, 0.488, 0.482])
error_array_dann_src /= 100.
error_array_dann_tgt = np.array([18.257, 18.201, 18.282])
error_array_dann_tgt /= 100.
error_array_dann_avg = (error_array_dann_src + error_array_dann_tgt) / 2.0

error_array_resnet_src = np.array([5.653, 5.733, 5.681])
error_array_resnet_src /= 100.
error_array_resnet_tgt = np.array([12.734, 12.907, 12.904])
error_array_resnet_tgt /= 100.
error_array_resnet_avg = (error_array_resnet_src + error_array_resnet_tgt) / 2.0

error_df_jcl_src = pd.DataFrame({'error': error_array_jcl_src, 'method': 'JCL', 'domain': 'Source'})
error_df_jcl_tgt = pd.DataFrame({'error': error_array_jcl_tgt, 'method': 'JCL', 'domain': 'Target'})
error_df_jcl_avg = pd.DataFrame({'error': error_array_jcl_avg, 'method': 'JCL', 'domain': 'Average'})

error_df_dann_src = pd.DataFrame({'error': error_array_dann_src, 'method': 'DANN', 'domain': 'Source'})
error_df_dann_tgt = pd.DataFrame({'error': error_array_dann_tgt, 'method': 'DANN', 'domain': 'Target'})
error_df_dann_avg = pd.DataFrame({'error': error_array_dann_avg, 'method': 'DANN', 'domain': 'Average'})

error_df_resnet_src = pd.DataFrame({'error': error_array_resnet_src, 'method': 'ResNet-101', 'domain': 'Source'})
error_df_resnet_tgt = pd.DataFrame({'error': error_array_resnet_tgt, 'method': 'ResNet-101', 'domain': 'Target'})
error_df_resnet_avg = pd.DataFrame({'error': error_array_resnet_avg, 'method': 'ResNet-101', 'domain': 'Average'})

# error_data = pd.concat([
#     error_df_resnet_src,
#     error_df_resnet_tgt,
#     error_df_resnet_avg,
#     error_df_dann_src,
#     error_df_dann_tgt,
#     error_df_dann_avg,
#     error_df_jcl_src,
#     error_df_jcl_tgt,
#     error_df_jcl_avg
# ])

error_data = pd.concat([
    error_df_dann_src,
    error_df_dann_tgt,
    error_df_dann_avg,
    error_df_jcl_src,
    error_df_jcl_tgt,
    error_df_jcl_avg
])

f = plt.figure(figsize=(3, 2.45))
ax = sns.barplot(data=error_data,
                 x='domain',
                 y='error',
                 hue='method',
                 ci=None)

ax.set_xlabel(None)
ax.set_ylabel('Error rate')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[:], labels=labels[:])
sns.despine()

# y3 = rs.choice(y1, len(y1), replace=False)
# sns.barplot(x=x, y=y3, palette="deep", ax=ax3)
# ax3.axhline(0, color="k", clip_on=False)
# ax3.set_ylabel("Qualitative")

# f = plt.figure()
# plt.plot(range(10), range(10), "o")
plt.show()

f.savefig("ideal_hypothesis.pdf", bbox_inches='tight')
