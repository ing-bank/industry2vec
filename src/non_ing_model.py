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

'''
Model building and fitting function by only using industry code description text
'''

# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.layers import *
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences


NUM_DIGITS = 6
WORD_EMBEDDING_DIM = 300
EMBEDDING_FILE = "../data/glove.840B.300d.txt"
NAICS_DESC_FILE = "../data/naics_code_description.csv"
MAX_SEQUENCE_LENGTH = 10
NAICS_EMBEDDING_DIM = 8
BATCH_SIZE = 1024
NUM_EPOCHS = 7

OUT_OF_VOC_MAP = {"schiffli": "textile machine",
                  "miniwarehouses": "mini warehouse",
                  "teleproduction": "tele production",
                  "nonupholstered": "non upholstered",
                  "noncitrus": "non citrus",
                  "nonchocolate": "non chocolate"}


def preprocess_title(title):
    title = title.lower().replace("?", "").replace("'", "").replace("’", "").strip()
    for key, value in OUT_OF_VOC_MAP.items():
        title = title.replace(key, value)
    return title


def text2seq(tokenizer, text_list, seq_len):
    return pad_sequences(tokenizer.texts_to_sequences(text_list), maxlen=seq_len)


def euclidean_distance(tensors):
    return tf.reduce_sum(tf.square(tf.subtract(tensors[0], tensors[1])), axis=1, keep_dims=True)


def get_model(nb_words, embedding_matrix):
    embedding_layer = Embedding(nb_words,
                                WORD_EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False,
                                name="word_embedding")
    dropout_layer = SpatialDropout1D(0.2, name="dropout")
    masking_layer = Masking(name="mask_sequence")
    gru_layer = GRU(NAICS_EMBEDDING_DIM*2, name="sentence_vector")
    dense_layer = Dense(NAICS_EMBEDDING_DIM, name="naics_vector")

    code1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32", name="naics_code1")
    code2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32", name="naics_code2")

    code1_emb = dense_layer(gru_layer(masking_layer(dropout_layer(embedding_layer(code1_input)))))
    code2_emb = dense_layer(gru_layer(masking_layer(dropout_layer(embedding_layer(code2_input)))))

    out = Lambda(euclidean_distance, name="euclidean_distance")([code1_emb, code2_emb])

    embedding_model = Model(inputs=[code1_input], outputs=[code1_emb])
    regression_model = Model(inputs=[code1_input, code2_input], outputs=out)
    regression_model.compile(loss="mse", optimizer="nadam")
    return embedding_model, regression_model


def get_combinations(df):
    df["key"] = 0
    df = pd.merge(df, df, on="key").drop("key", axis=1)
    df = df[df["Code_x"] <= df["Code_y"]]
    return df


def get_distance(code1, code2):
    distance = NUM_DIGITS
    for i in range(NUM_DIGITS):
        if code1[i] == code2[i]:
            distance -= 1
        else:
            return distance
    return distance


def get_glove_embedding_dict(glove_file, voc):
    embeddings_index = {}
    f = open(glove_file)
    for line in f:
        values = line.split()
        word = values[0]
        if word in voc and len(values) == WORD_EMBEDDING_DIM + 1:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def generate_embeddings(experiment_mode):
    df = pd.read_csv(NAICS_DESC_FILE, usecols=["Code", "Title"])
    df["Title"] = df["Title"].apply(preprocess_title)
    df["code_len"] = df["Code"].apply(len)
    df = df[df["code_len"] == NUM_DIGITS].drop("code_len", axis=1)
    unique_naics = df.copy()

    df = get_combinations(df)
    df["distance"] = df.apply(lambda x: get_distance(x["Code_x"], x["Code_y"]), axis=1)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df["Title_x"])
    vocabulary = list(tokenizer.word_index.keys())

    embeddings_index = get_glove_embedding_dict(EMBEDDING_FILE, vocabulary)

    nb_words = len(vocabulary) + 1
    embedding_matrix = np.zeros((nb_words, WORD_EMBEDDING_DIM))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    df = df.sample(frac=1)
    X, y = df, df["distance"]

    embedding_model, regression_model = get_model(nb_words, embedding_matrix)
    regression_model.summary()

    code1 = text2seq(tokenizer, X["Title_x"], MAX_SEQUENCE_LENGTH)
    code2 = text2seq(tokenizer, X["Title_y"], MAX_SEQUENCE_LENGTH)

    regression_model.fit([code1, code2], y, validation_split=0.2*experiment_mode,
                         epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=1)
    X["pred_dist"] = regression_model.predict([code1, code2], batch_size=BATCH_SIZE, verbose=1)

    unique_naics["Embedding"] = embedding_model.predict(text2seq(tokenizer, unique_naics["Title"], MAX_SEQUENCE_LENGTH),
                                                        verbose=1).tolist()

    return unique_naics, X
