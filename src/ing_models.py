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
This file provide model building and fitting function by using ING data
'''

import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import dot, Input, Activation,\
                         Embedding, Flatten, BatchNormalization,\
                         LSTM, concatenate
from keras.callbacks import EarlyStopping

import utils
from params import MAX_SEQUENCE_LENGTH, WORD_EMBEDDING_DIM




SELECT_DIGIT = 4


def industry_embedding_model_build(vocab_size,
                                   num_words,
                                   vec_dim_4digit,
                                   vec_dim_desc,
                                   embedding_matrix):

    input_pvt = Input(batch_shape=(None, 1), dtype='int32')
    input_ctx = Input(batch_shape=(None, 1), dtype='int32')
    embedded_pvt = Embedding(input_dim=vocab_size,
                             output_dim=vec_dim_4digit,
                             input_length=1)(input_pvt)

    embedded_ctx = Embedding(input_dim=vocab_size,
                             output_dim=vec_dim_4digit,
                             input_length=1)(input_ctx)

    input_target_desc = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    input_context_desc = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                WORD_EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    embedded_target_desc = embedding_layer(input_target_desc)
    embedded_context_desc = embedding_layer(input_context_desc)

    lstm_layer = LSTM(vec_dim_desc, input_shape=(1, WORD_EMBEDDING_DIM))
    targe_desc_vec = lstm_layer(embedded_target_desc)
    context_desc_vec = lstm_layer(embedded_context_desc)

    target_vec = concatenate([Flatten()(embedded_pvt), targe_desc_vec])
    context_vec = concatenate([Flatten()(embedded_ctx), context_desc_vec])

    target_vec = BatchNormalization()(target_vec)
    context_vec = BatchNormalization()(context_vec)

    merged = dot([target_vec, context_vec], axes=1)

    predictions = Activation('sigmoid')(merged)
    digit2_match = Activation('sigmoid')(merged)

    # build and train the model
    model = Model(inputs=[input_pvt,
                          input_ctx,
                          input_target_desc,
                          input_context_desc,
                          ],
                  output=[predictions, digit2_match])
    model.compile(optimizer='adam', loss='binary_crossentropy', loss_weights=[0.8, 0.2])

    embedding_model = Model(inputs=[input_pvt, input_target_desc], outputs=target_vec)

    print(model.summary())

    return model, embedding_model


def industry_embedding_model_fit(model,
                                 embedding_model,
                                 nb_epoch,
                                 batch_size,
                                 df,
                                 dict_sequence,
                                 industry2idx,
                                 output_file):

    X_train, X_test, y_train, y_test, y_train_2digit, y_test_2digit = utils.get_train_test(df, batch_size, 0.1)

    target_grid_col = 'naics_4_digit'
    context_grid_col = 'context'
    target_col = 'target_desc'
    context_col = 'context_desc'

    model.fit(x=[X_train[target_grid_col],
                 X_train[context_grid_col],
                 utils.rebuild_array(X_train[target_col]),
                 utils.rebuild_array(X_train[context_col])], y=[y_train, y_train_2digit],
              epochs=nb_epoch,
              shuffle=True, batch_size=batch_size,
              verbose=1,
              validation_data=([X_test[target_grid_col],
                               X_test[context_grid_col],
                               utils.rebuild_array(X_test[target_col]),
                               utils.rebuild_array(X_test[context_col])],
                               [y_test, y_test_2digit]),
              callbacks=[EarlyStopping(monitor='val_loss',
                                       min_delta=0.0001,
                                       patience=3,
                                       verbose=1,
                                       mode='auto')])

    sequence_df = pd.DataFrame({k: v for k, v in dict_sequence.items() if len(k) == SELECT_DIGIT}).T
    naics_idx_l = np.array([industry2idx.get(int(naics), len(industry2idx)+1) for naics in sequence_df.index])
    in_data = [naics_idx_l, sequence_df.values]
    embedding_vec = embedding_model.predict(x=in_data, verbose=1, batch_size=1)

    utils.save_weights(output_file, embedding_vec, naics_idx_l, idx2industry)

    return embedding_vec
