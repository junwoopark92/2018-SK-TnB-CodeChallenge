import sys
import os
import glob
import time
from gensim.models import Word2Vec
import pandas as pd


def make_word2vec_model(filepath):
    user_hist = pd.read_csv(filepath)
    result = []
    for i, index in enumerate(user_hist['USER_ID'].value_counts().index):
        st = time.time()
        sub = user_hist[user_hist['USER_ID'] == index].sort_values(['WATCH_DAY', 'WATCH_SEQ'])
        inseq = 10
        temp_list = []
        temp_col_list = []
        for k in range(inseq):
            temp_col_list.append('MOVIE_ID_t' + str(k))
            temp_list.append(sub['MOVIE_ID'].shift(-1 * k))

        df = pd.concat(temp_list, axis=1)
        df.columns = temp_col_list
        df.dropna(inplace=True)
        df = df.astype(int).astype(str)
        result.extend(df.values.tolist())
        et = time.time()

        if i % 1000 == 0:
            print(i, index, (et - st) * 1000)

    return result


def main():
    dirpath = sys.argv[1]
    file_list = glob.glob(dirpath + 'sample*')

    input_list = []
    for file in file_list:
        print(file)
        input_list.extend(make_word2vec_model(file))

    if not os.path.isdir(dirpath + 'model'):
        os.mkdir(dirpath + 'model')

    model = Word2Vec(input_list, window=4, min_count=0, sg=1)
    model.wv[['5375', '13']]

    model.save(dirpath + 'model/word2vec_model')
    print('done')


if __name__== '__main__':
    main()