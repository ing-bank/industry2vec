# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def load_embedding_vec(embedding_name):
    '''
    load numpy array embedding vector from file
    '''
    return np.load(open(embedding_name, 'rb'))


def get_tsne_2d(vec):
    '''
    Get T-SNE 2 components dimension reduction
    '''
    X_embedded = TSNE(n_components=2, verbose=1).fit_transform(vec)
    return X_embedded


def plot_embedding(X_embedded, df_desc, idx2industry, color_digit):
    '''
    Plot embedding vector in 2D scatter plot
    Group the color by first `color_digit` digit
    Input:
    - X_embedded: embedding vectors
    - df_desc:dataframe with industry description
    - idx2industry: dictionary mapping index to industry
    - color_digit: digit to group the color
    '''

    code_top_digit = set([code[:color_digit] for code in df_desc.Code.unique()])
    code_top_digit = sorted(list(code_top_digit))

    rgb_values = sns.color_palette("Set2", len(code_top_digit))
    color_map = dict(zip(code_top_digit, rgb_values))

    industry_l = [str(idx2industry[idx]) for idx, _ in enumerate(X_embedded)]
    color_l = [color_map.get(ind[:color_digit], rgb_values[-1]) for ind in industry_l]

    x = X_embedded[:, 0]
    y = X_embedded[:, 1]
    plt.scatter(x, y, c=color_l)


def get_most_similar(naics, top_n, vec, df_desc, industry2idx, idx2industry):
    '''
    Get most similar top n NAICS code

    '''
    v = vec[industry2idx[naics], :]

    nbrs = NearestNeighbors(n_neighbors=top_n,
                            algorithm='ball_tree').fit(vec)

    distances, indices = nbrs.kneighbors(v.reshape(1, -1))

    naics_l = [str(idx2industry[idx]) for idx in indices.tolist()[0]]
    desc_l = [df_desc[df_desc.Code == industry]['Title'].values[0]
              if sum(df_desc.Code == industry) > 0 else 'Not found'
              for industry in naics_l]

    df = pd.DataFrame({'naics': naics_l,
                       'distance': distances.tolist()[0],
                       'description': desc_l})
    return df[['naics', 'distance', 'description']]
