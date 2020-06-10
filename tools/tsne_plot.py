import argparse
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()

parser.add_argument('--feature_dir',
                    type=str,
                    default=None,
                    help='Feature directory')
parser.add_argument('--label_dir',
                    type=str,
                    default=None,
                    help='Label directory')
parser.add_argument('--df_dir',
                    type=str,
                    default=None,
                    help='Label directory')
parser.add_argument('--plot',
                    type=bool,
                    default=False,
                    help='Plot flag')
parser.add_argument('--full',
                    type=bool,
                    default=False,
                    help='Use all of data flag')


def main():
    args = parser.parse_args()

    if not args.plot:
        features_file = open(args.feature_dir, 'rb')
        labels_file = open(args.label_dir, 'rb')
        features = pickle.load(features_file)
        labels = pickle.load(labels_file)

        print(features.shape)
        print(labels.shape)

        feat_cols = ['feature' + str(i) for i in range(features.shape[1])]
        df = pd.DataFrame(features, columns=feat_cols)
        df['y'] = labels
        df['label'] = df['y'].apply(lambda i: str(i))

        print('Size of the dataframe: {}'.format(df.shape))

        if args.full:
            df_subset = df.loc[:, :].copy()

        else:
            np.random.seed(42)
            rndperm = np.random.permutation(df.shape[0])

            N = 10000
            df_subset = df.loc[rndperm[:N], :].copy()

        data = df_subset[feat_cols].values

        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(data)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        df_subset['tsne-2d-one'] = tsne_results[:, 0]
        df_subset['tsne-2d-two'] = tsne_results[:, 1]

        df_subset.to_pickle(args.df_dir)

    else:
        plot(args.df_dir)


def plot(load_dir):
    df = pd.read_pickle(load_dir)
    f = plt.figure(figsize=(12, 14.8))
    plt.axis('off')

    sns.set_context('paper')
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        size=1,
        palette=sns.color_palette("hls", 12),
        data=df,
        legend=False,
        alpha=0.5,
        linewidth=0
    )
    sns.despine(left=True, bottom=True)

    plt.show()

    f.savefig("jcl_tsne.pdf", bbox_inches='tight')

if __name__ == '__main__':
    main()
