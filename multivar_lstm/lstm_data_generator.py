import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
import keras


def strtuple2arr(x):
    return np.array(x.replace('(', '').replace(')', '').split(','), dtype=np.float32)


def get_lstm_sub_data(filepath, modelpath):
    df = pd.read_csv(filepath, index_col='Unnamed: 0')
    size = len(df)//1000 * 1000 
    df = df.iloc[:size]
    w2v_model = Word2Vec.load(modelpath)

    result_X = []
    Y = 0

    size = len(df)
    for col in df.columns:
        if 'MOVIE_ID' in col:
            #print(col)
            arr = w2v_model.wv[df[col].astype(int).astype(str).tolist()]
            result_X.append(arr)

        if 'scaled_duration' in col:
            #print(col)
            arr = df[col].values.reshape(size, 1)
            result_X.append(arr)

        if 'diff_day' in col:
            #print(col)
            arr = df[col].values.reshape(size, 1)
            result_X.append(arr)

        if 'week_coord' in col:
            #print(col)
            arr = np.array(df[col].apply(strtuple2arr).tolist())
            result_X.append(arr)

        if 'day_coord' in col:
            #print(col)
            arr = np.array(df[col].apply(strtuple2arr).tolist())
            result_X.append(arr)

        if 'month_coord' in col:
            #print(col)
            arr = np.array(df[col].apply(strtuple2arr).tolist())
            result_X.append(arr)

        if '(t-10)' in col:
            #print(col)
            Y = df[col].values.astype(int).astype(str)

    X = np.concatenate(result_X, axis=1)
    X = X.reshape(X.shape[0], 10, 118)

    # encode class values as integers
    encoder = OneHotEncoder(sparse=False)
    # get movie all list
    movie_arr = pd.read_csv('data/KISA_TBC_MOVIES.tsv',sep='\t' ,encoding='utf-8')['MOVIE_ID'].apply(lambda x: str(x).replace(',', '')).values

    encoder.fit(movie_arr.reshape(-1,1))
    encoded_Y = encoder.transform(Y.reshape(Y.shape[0], 1))
    #print(encoded_Y.shape)
    # convert integers to dummy variables (i.e. one hot encoded)

    return X, encoded_Y


