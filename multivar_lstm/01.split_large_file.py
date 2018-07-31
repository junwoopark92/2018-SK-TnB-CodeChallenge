import os
import sys
import time
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def main():
    train_filepath = sys.argv[1]
    test_filepath = sys.argv[2]
    train_filename = train_filepath.split('/')[-1]
    test_filename = test_filepath.split('/')[-1]
    n_split = int(sys.argv[3])

    train_df = pd.read_csv(train_filepath)
    train_df.info()

    test_df = pd.read_csv(test_filepath)
    test_df.info()

    # diff day
    train_df.sort_values(['USER_ID', 'WATCH_DAY', 'WATCH_SEQ'], inplace=True)
    test_df.sort_values(['USER_ID', 'WATCH_DAY', 'WATCH_SEQ'], inplace=True)
    print('sorted')

    date_format = "%Y%m%d"
    train_df['datetime_watch_day'] = train_df['WATCH_DAY'].apply(lambda x: datetime.strptime(str(x), date_format))
    test_df['datetime_watch_day'] = test_df['WATCH_DAY'].apply(lambda x: datetime.strptime(str(x), date_format))

    train_df['diff_day'] = train_df['datetime_watch_day'] - train_df['datetime_watch_day'].shift(1)
    train_df['diff_day'] = train_df['diff_day'].apply(lambda x: x.days)

    test_df['diff_day'] = test_df['datetime_watch_day'] - test_df['datetime_watch_day'].shift(1)
    test_df['diff_day'] = test_df['diff_day'].apply(lambda x: x.days)

    train_result = []
    for i, index in enumerate(train_df['USER_ID'].value_counts().index):
        st = time.time()
        sub = train_df.loc[train_df.USER_ID == index, 'diff_day']
        sub.iloc[0] = 0
        train_result.append(sub)
        et = time.time()
        if i % 1000 == 0:
            print(i, index, (et - st) * 1000)

    test_result = []
    for i, index in enumerate(test_df['USER_ID'].value_counts().index):
        st = time.time()
        sub = test_df.loc[test_df.USER_ID == index, 'diff_day']
        sub.iloc[0] = 0
        test_result.append(sub)
        et = time.time()
        if i % 1000 == 0:
            print(i, index, (et - st) * 1000)

    train_diff_df = pd.concat(train_result, axis=0)
    test_diff_df = pd.concat(test_result, axis=0)

    train_df['diff_day'] = train_diff_df
    test_df['diff_day'] = test_diff_df

    # scaling
    duration_arr = np.concatenate([train_df['DURATION'].values, test_df['DURATION'].values], axis=0).reshape(-1, 1)
    duration_scaler = MinMaxScaler(feature_range=(-1, 1))
    duration_scaler.fit(duration_arr)

    train_df['scaled_duration'] = duration_scaler.transform(train_df['DURATION'].values.reshape(-1, 1))
    test_df['scaled_durtion'] = duration_scaler.transform(test_df['DURATION'].values.reshape(-1, 1))

    diff_arr = np.concatenate([train_df['diff_day'].values, test_df['diff_day'].values], axis=0).reshape(-1, 1)
    diff_scaler = MinMaxScaler(feature_range=(-1, 1))
    diff_scaler.fit(diff_arr)

    train_df['diff_day'] = diff_scaler.transform(train_df['diff_day'].values.reshape(-1, 1))
    test_df['diff_day'] = diff_scaler.transform(test_df['diff_day'].values.reshape(-1, 1))

    test_df.to_csv('scaled_'+test_filename)

    user_list = train_df['USER_ID'].value_counts().index.tolist()
    random.shuffle(user_list)

    num_file = n_split
    num_user = len(user_list)//n_split + 1

    if not os.path.isdir('./split_data'):
        os.mkdir('./split_data')

    filename = './split_data/'+train_filename+'.part{}'

    for index in range(num_file):

        st = index * num_user
        et = (index+1) * num_user
        target_user = user_list[st:et]

        file_df = train_df[train_df['USER_ID'].isin(target_user)]
        file_df.to_csv(filename.format(index))
        print(index, st, et, 'done')


if __name__ == "__main__":
    main()
