#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import re
from ast import literal_eval
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from params import MAX_SEQUENCE_LENGTH, MAX_NUM_WORDS, WORD_EMBEDDING_DIM


def save_weights(filename, vec, index_l, idx2industry):
    """
    Save vector array to file
    :param index2word: list of string
    :param vec_dim: dim of embedding vector
    :return:
    """
    f = open(filename, 'w')

    for i, idx in enumerate(index_l):
        industry_code = idx2industry.get(idx, None)
        if industry_code is not None:
            f.write(str(idx2industry[idx]))
            f.write(" ")
            f.write(" ".join(map(str, list(vec[i, :]))))
            f.write("\n")
    f.close()


def load_data(context_data_file, weight_data_file):
    """
    Load the csv file and do some basic preprocessing
    """
    df_context = pd.read_csv(context_data_file, sep=';')
    df_context.industry_list = df_context.industry_list.apply(lambda l: literal_eval(l))
    df_context = df_context[~pd.isnull(df_context.naics)]
    df_context.naics = df_context.naics.astype(int)

    df_weight = pd.read_csv(weight_data_file, sep=';')

    return df_context, df_weight


def explode(df, sublist, naics):
    """
    Explode naics code from a list to multiple rows
    """
    rows = []
    df.apply(lambda row: [rows.append([row[naics], nn])
                          for nn in row[sublist]], axis=1)
    return pd.DataFrame(rows, columns=[naics, 'context'])


def prepare_data(df_context, df_weight, digit=4):
    """
    Prepare data:
    - Cutoff nr of digit of naics as input parameter "digit"
    - Explode list of naics code to multiple rows
    - get normalized weight probability of each naics code
    - get industry code to/from index dictionary
    """

    # cut off naics code to given digit for target and context naics code
    list_name = 'industry_list'
    new_list_name = '{}_{}_digit'.format(list_name, digit)
    df_context[new_list_name] = df_context[list_name].apply(lambda l: [int(str(e)[:digit]) for e in l])
    df_context['nr_subnode'] = df_context[new_list_name].apply(len)

    naics_name = 'naics'
    new_naics_name = '{}_{}_digit'.format(naics_name, digit)
    df_context[new_naics_name] = df_context[naics_name].apply(lambda code: int(str(code)[:digit]))

    # explode the naics code list to multiple rows
    df_pos = explode(df_context, sublist=new_list_name, naics=new_naics_name)
    df_pos['label'] = 1

    # cut off naics code to given digit for naics
    df_weight[new_naics_name] = df_weight[naics_name].apply(lambda code: int(str(code)[:digit]))

    # set the normalized weighting
    df_weight = df_weight.groupby(new_naics_name)[['cnt']].sum().reset_index()
    df_weight['norm_p'] = df_weight.cnt / sum(df_weight.cnt)

    # get industry to/from index dictionary
    idx2industry = {k: v for k, v in enumerate(df_weight[new_naics_name].tolist())}
    industry2idx = {v: k for k, v in enumerate(df_weight[new_naics_name].tolist())}

    return df_context[[new_naics_name, new_list_name, 'nr_subnode']], df_weight, df_pos, idx2industry, industry2idx


def append_neg_set(df_context, df_weight, df_pos, naics_col, neg_factor=10):
    """
    Negative sampling
    """

    context_list_dict = df_pos.groupby(naics_col)['context'].unique().to_dict()
    # context_cnt_dict = df_pos[naics_col].value_counts().to_dict()
    cnt_neg_sampel = len(df_pos) // len(df_pos[naics_col].unique())

    df_neg_l = []
    for naics in df_pos[naics_col].unique():
        df_neg = pd.DataFrame()
        df_neg[naics_col] = np.array([naics]*cnt_neg_sampel)

        sample_l = np.array(list(set(df_weight[naics_col]) - set(context_list_dict[naics])))
        sample_p_l = df_weight.set_index(naics_col).loc[sample_l]['norm_p']
        sample_p_l = sample_p_l / sum(sample_p_l)
        df_neg['context'] = np.random.choice(sample_l, cnt_neg_sampel, p=sample_p_l)

        df_neg_l.append(df_neg)

    df_neg_total = pd.concat(df_neg_l)
    df_neg_total['label'] = 0

    return pd.concat([df_pos, df_neg_total])


def expand_bad_naics_code(row, out_l):
    code_l = row['Code'].split('-')
    if len(code_l) > 2:
        return []

    for code in range(int(code_l[0]), int(code_l[1])+1):
        out_l.append([str(code), row['Title']])

    return


def get_2digit_idx_dict(desc_file):
    df_desc = pd.read_csv(desc_file)

    naics_bad = df_desc[['Code', 'Title']][df_desc.Code.str.contains('-')]
    out_l = []
    naics_bad.apply(lambda row: expand_bad_naics_code(row, out_l), axis=1)

    naics_2digit = df_desc[['Code', 'Title']][df_desc.Code.str.len() == 2]
    naics_diff_digit = pd.DataFrame(out_l, columns=['Code', 'Title'])

    naics_2digit = pd.concat([naics_2digit, naics_diff_digit])

    idx_to_naics_2digit = {k: v for k, v in enumerate(set(naics_2digit['Title'].tolist()))}
    naics_2digit_to_idx = {v: k for k, v in idx_to_naics_2digit.items()}

    naics_2digit['2digit_idx'] = naics_2digit['Title'].apply(lambda t: naics_2digit_to_idx[t])

    naics_2digit_to_idx = naics_2digit.set_index('Code')['2digit_idx'].to_dict()
    idx_to_naics_2digit = naics_2digit.set_index('2digit_idx')['Code'].to_dict()

    return naics_2digit_to_idx, idx_to_naics_2digit


def get_input(df_context, df_weight, df_pos, dict_sequence, naics_col, industry2idx, naics_2digit_to_idx):
    """
    Get input data for embedding layer
    """
    len_naics = len(industry2idx)
    df_train = append_neg_set(df_context, df_weight, df_pos, naics_col)

    df_train['target_desc'] = df_train[naics_col].apply(lambda naics: dict_sequence.get(str(naics),
                                                                                        [0]*MAX_SEQUENCE_LENGTH))
    df_train['context_desc'] = df_train['context'].apply(
                                lambda context: dict_sequence.get(str(context), [0]*MAX_SEQUENCE_LENGTH))

    df_train[naics_col+'_2digit_idx'] = df_train[naics_col].apply(
                                    lambda naics: naics_2digit_to_idx.get(str(naics)[:2],
                                                                          len(naics_2digit_to_idx)))
    df_train['context_2digit_idx'] = df_train['context'].apply(
                                    lambda naics: naics_2digit_to_idx.get(str(naics)[:2],
                                                                          len(naics_2digit_to_idx)))

    # add a column indicate if the first two digits of target and context naics code are the same
    df_train['same_2_digit'] = df_train.apply(lambda row: 1 if row[naics_col+'_2digit_idx'] == row['context_2digit_idx'] else 0, axis=1)
    df_train = df_train.drop(naics_col+'_2digit_idx', axis=1)
    df_train = df_train.drop(context_2digit_idx, axis=1)

    df_train[naics_col] = df_train[naics_col].apply(lambda x: industry2idx.get(x, len_naics))
    df_train['context'] = df_train['context'].apply(lambda x: industry2idx.get(x, len_naics))

    return df_train


def correct_desc_df(df_desc):
    """
    Some NAICS code is like "32-34".
    This function expend one row to multiple rows with "32", "33", "34" separately
    """

    naics_good = df_desc[['Code', 'Title']][~df_desc.Code.str.contains('-')]
    naics_bad = df_desc[['Code', 'Title']][df_desc.Code.str.contains('-')]

    out_l = []
    naics_bad.apply(lambda row: expand_bad_naics_code(row, out_l), axis=1)

    df = pd.concat((naics_good, pd.DataFrame(out_l, columns=['Code', 'Title'])), axis=0)
    return df


def get_desc_dict(desc_file, digit=4):
    df_desc = pd.read_csv(desc_file)
    df_desc = correct_desc_df(df_desc)

    digit_l = list(range(2, digit+1))
    df_desc = df_desc[df_desc.Code.str.len().isin(digit_l)][['Code', 'Title']]
    dict_digit_desc = df_desc.set_index('Code')['Title'].to_dict()

    return df_desc, dict_digit_desc, digit_l


def get_desc(df, naics_col, digit_l, dict_digit_desc):
    def get_desc_func(naics):
        desc_str = ""
        for digit in digit_l:
            desc_str += dict_digit_desc.get(naics[:digit], "") + " "
        return desc_str

    return df[naics_col].astype(str).apply(get_desc_func)


def get_glove_embedding_dict(glove_file):
    # # Use pretrain GLOVE
    embeddings_index = {}
    f = open(glove_file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def text_to_wordlist(text):
    # Clean the text
    text = re.sub(r"[\^,!.\/'+-=$]", " ", text)

    # Convert words to lower case and split them
    text = text.lower().split()
    text = " ".join(text)

    return text


def get_embedding_vector(df, col_desc, col_naics, embeddings_index):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    texts = df[col_desc].apply(text_to_wordlist)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    dict_sequence = dict(zip(df[col_naics], data.tolist()))

    print('Shape of data tensor:', data.shape)

    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, WORD_EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, dict_sequence, num_words


def get_train_test(df, batch_size, test_size=0.1):
    '''
    Split dataset as train and test set
    Return:
        X_train, X_test: features
        y_train, y_test: labels if two industry codes are similar
        y_train_same_2_digit, y_test_same_2_digit: labels if first two digits are the same
    '''

    y = df['label']
    X = df.drop('label', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=test_size)
    y_train_same_2_digit = X_train['same_2_digit']
    y_test_same_2_digit = X_test['same_2_digit']

    X_train = X_train.drop('same_2_digit', axis=1),
    X_test = X_test.drop('same_2_digit', axis=1),

    return X_train, X_test, y_train, y_test, y_train_same_2_digit, y_test_same_2_digit


def rebuild_array(s):
    return np.array([list(l) for l in s])
