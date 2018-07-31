import sys
import os
import glob
import time
import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.preprocessing import minmax_scale

num_values = 7
degrees_week = np.linspace(0, 360, num_values + 1)[:-1]
print(degrees_week)
sin_week = np.sin(np.deg2rad(degrees_week))
print(sin_week)
cos_week = np.cos(np.deg2rad(degrees_week))
print(cos_week)

num_values = 12
degrees_month = np.linspace(0, 360, num_values + 1)[:-1]
print(degrees_month)
sin_month = np.sin(np.deg2rad(degrees_month))
print(sin_month)
cos_month = np.cos(np.deg2rad(degrees_month))
print(cos_month)

num_values = 31
degrees_day = np.linspace(0, 360, num_values + 1)[:-1]
print(degrees_day)
sin_day = np.sin(np.deg2rad(degrees_day))
print(sin_day)
cos_day = np.cos(np.deg2rad(degrees_day))
print(cos_day)


def make_coord(date,num_class):
    # 1 ~ 12, 1 ~ 7, 1 ~ 31
    degrees = np.linspace(0,360,num_values + 1)[:-1]
    sin = np.sin(np.deg2rad(degrees))
    cos = np.cos(np.deg2rad(degrees))
    return sin, cos


def date2coord(date, sin, cos):
    return cos[date-1], sin[date-1]


def date2week(x):
    y = int(str(x)[:4])
    m = int(str(x)[4:6])
    d = int(str(x)[6:])
    day = datetime(y, m, d)
    return day.weekday()


def make_lstm_dataset(filepath, dirpath, istrain):

    print(filepath)
    df = pd.read_csv(filepath)

    # date to coord
    df['watch_month'] = df['WATCH_DAY'].apply(lambda x: int(str(x)[4:6]))
    df['watch_day'] = df['WATCH_DAY'].apply(lambda x: int(str(x)[6:]))

    df['week'] = df['WATCH_DAY'].apply(date2week)
    df['week'] = df['week'] + 1

    df['day_coord'] = df['watch_day'].apply(date2coord, args=(sin_day, cos_day,))
    df['month_coord'] = df['watch_month'].apply(date2coord, args=(sin_month, cos_month))
    df['week_coord'] = df['week'].apply(date2coord, args=(sin_week, cos_week,))

    result = []
    for i, index in enumerate(df['USER_ID'].value_counts().index):
        st = time.time()
        # 한유저의 마지막 diff_day 는 다른유저와의 차이임으로 제거
        sub = df[df['USER_ID'] == index].sort_values(['WATCH_DAY', 'WATCH_SEQ'])
        # make train, label
        inseq = 10
        names = []
        temp_list = []
        features = ['MOVIE_ID', 'scaled_duration', 'diff_day', 'week_coord', 'day_coord', 'month_coord']

        for k in range(inseq):
            names += [(feature_name + '(t-%d)' % k) for feature_name in features]
            temp_list.append(sub[features].shift(-1 * k))
            if (k == inseq - 1)  and istrain == 'True':
                names += [('MOVIE_ID' + '(t-%d)' % inseq)]
                temp_list.append(sub['MOVIE_ID'].shift(-1 * inseq))

        temp_df = pd.concat(temp_list, axis=1)
        temp_df.columns = names
        temp_df.dropna(inplace=True)
        temp_df['USER_ID'] = index
        result.append(temp_df)
        et = time.time()

        if i % 1000 == 0:
            print(i, index, (et - st) * 1000)

    filename = filepath.split('/')[-1]
    if len(result) == 1:
        merged_df  = result[0]
    else:
        merged_df = pd.concat(result, axis=0)
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.to_csv(dirpath+'lstm_dataset/lstm_'+filename)


def main():
    dirpath = sys.argv[1]
    filename_pattern = sys.argv[2] 
    istrain =  sys.argv[3]
    file_list = glob.glob(dirpath+filename_pattern)
    print(len(file_list))
    for file in file_list:
        make_lstm_dataset(file, dirpath, istrain)

    if not os.path.isdir(dirpath+'lstm_dataset'):
        os.mkdir(dirpath+'lstm_dataset')

    print('done')


if __name__ == '__main__':
    main()
