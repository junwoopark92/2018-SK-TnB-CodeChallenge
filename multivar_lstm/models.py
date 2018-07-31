from keras.layers import Input, LSTM, Dense
from keras.models import Model


# define baseline model
def baseline_model(shape, batch_size):
    main_input = Input(shape=(shape[1], shape[2]),
                       batch_shape=(batch_size, shape[1], shape[2]), name='main_input')

    x = LSTM(128, return_sequences=True, stateful=True)(main_input)
    x = LSTM(128, return_sequences=True, stateful=True)(x)
    x = LSTM(128, return_sequences=True, stateful=True)(x)
    x = LSTM(128, stateful=True)(x)

    x = Dense(256, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    main_output = Dense(8259, activation='softmax')(x)

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def regression_model(shape, batch_size):
    main_input = Input(shape=(shape[1], shape[2]),
                       batch_shape=(batch_size, shape[1], shape[2]), name='main_input')

    x = LSTM(128, return_sequences=True, stateful=True)(main_input)
    x = LSTM(128, return_sequences=True, stateful=True)(x)
    x = LSTM(128, return_sequences=True, stateful=True)(x)
    x = LSTM(128, stateful=True)(x)

    x = Dense(256, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    main_output = Dense(100, activation='tanh')(x)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    return model