import os
import sys
import glob
import pandas as pd


def main():
    dirpath = sys.argv[1]
    file_list = glob.glob(dirpath+'*.csv*')

    for file in file_list:
        print(file)
        make_label(file, dirpath)

    if not os.path.isdir(dirpath+'label'):
        os.mkdir(dirpath+'label')

    print('done')


def make_label(filepath, dirpath):
    df = pd.read_csv(filepath)
    result = []

    for i, index in enumerate(df['USER_ID'].value_counts().index):
        if i % 1000 == 0:
            print(i, index)

        sub = df[df['USER_ID'] == index].sort_values(['WATCH_DAY', 'WATCH_SEQ'])
        after_ten = sub.iloc[10:]['MOVIE_ID'].tolist()
        dummy_set = set()
        movie_list = [(movie_id, dummy_set.add(movie_id))[0] for movie_id in after_ten if movie_id not in dummy_set]
        result.append((index, movie_list))

    result_df = pd.DataFrame(result, columns=['USER_ID', 'movie_list'])
    result_df.head()
    filename = filepath.split('/')[-1]
    result_df.to_csv(dirpath+"label_"+filename, sep='\t', index=False)


if __name__ == '__main__':
    main()

