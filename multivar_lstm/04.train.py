import glob
import sys
import numpy as np
sys.path.append('..')
from multivar_lstm.models import baseline_model
from multivar_lstm.lstm_data_generator import get_lstm_sub_data

import time


def main():
    dirpath = sys.argv[1]
    modelpath = 'movie_w2v_mincount_model.model'
    file_list = glob.glob(dirpath+'/*')
    model = baseline_model(shape=(1000, 10, 118), batch_size=1000)
    epochs = int(sys.argv[2])
    batch_size = 1000

    rate = int(len(file_list)*0.6)

    train_file_list = file_list[:rate]
    test_file_list = file_list[rate:-1]
    val_file = file_list[-1]

    for epoch in range(epochs):
        st = time.time()
        for train_file in train_file_list:
            print('{} / {} {}'.format(epoch, epochs, train_file))
            train_X, train_y = get_lstm_sub_data(train_file, modelpath)
            model.fit(train_X, train_y, batch_size=batch_size, epochs=1)
            val_X, val_y = get_lstm_sub_data(val_file, modelpath)
            print()
            print(model.evaluate(val_X[:10000], val_y[:10000], batch_size=batch_size))
        et = time.time()
        print(epoch,'is done time needed ',et - st ,'sec')

    test_results = []
    for test_file in test_file_list:
        test_X, test_y = get_lstm_sub_data(test_file, modelpath)
        test_results.append(model.evaluate(test_X, test_y, batch_size=batch_size)[1])

    print(np.mean(test_results))
    model_json = model.to_json()
    with open("model/model.json","w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("saved model")

if __name__ == '__main__':
    main()
