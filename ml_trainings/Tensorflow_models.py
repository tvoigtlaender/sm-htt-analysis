from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.regularizers import l2


def example(num_inputs, num_outputs):
    """
    Example Keras model
    """
    model = Sequential()
    model.add(Dense(10, init="glorot_normal", activation="relu", input_dim=num_inputs))
    model.add(Dense(num_outputs, init="glorot_uniform", activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=[
            "categorical_accuracy",
        ],
    )
    return model


def smhtt_simple(num_inputs, num_outputs):
    model = Sequential()
    model.add(Dense(100, init="glorot_normal", activation="tanh", input_dim=num_inputs))
    model.add(Dense(num_outputs, init="glorot_normal", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Nadam(), metrics=[])
    return model


def smhtt_mt(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(
            300,
            init="glorot_normal",
            activation="tanh",
            W_regularizer=l2(1e-4),
            input_dim=num_inputs,
        )
    )
    model.add(
        Dense(300, init="glorot_normal", activation="tanh", W_regularizer=l2(1e-4))
    )
    model.add(
        Dense(300, init="glorot_normal", activation="tanh", W_regularizer=l2(1e-4))
    )
    model.add(Dense(num_outputs, init="glorot_normal", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Nadam(), metrics=[])
    return model


def smhtt_et(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(
            1000,
            init="glorot_normal",
            activation="tanh",
            W_regularizer=l2(1e-4),
            input_dim=num_inputs,
        )
    )
    model.add(Dense(num_outputs, init="glorot_normal", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Nadam(), metrics=[])
    return model


def smhtt_tt(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(
            200,
            init="glorot_normal",
            activation="tanh",
            W_regularizer=l2(1e-4),
            input_dim=num_inputs,
        )
    )
    model.add(
        Dense(200, init="glorot_normal", activation="tanh", W_regularizer=l2(1e-4))
    )
    model.add(
        Dense(200, init="glorot_normal", activation="tanh", W_regularizer=l2(1e-4))
    )
    model.add(Dense(num_outputs, init="glorot_normal", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Nadam(), metrics=[])
    return model


def smhtt_legacy(num_inputs, num_outputs):
    model = Sequential()
    model.add(
        Dense(
            300,
            init="glorot_normal",
            activation="relu",
            W_regularizer=l2(1e-4),
            input_dim=num_inputs,
        )
    )
    model.add(
        Dense(300, init="glorot_normal", activation="relu", W_regularizer=l2(1e-4))
    )
    model.add(
        Dense(300, init="glorot_normal", activation="relu", W_regularizer=l2(1e-4))
    )
    model.add(Dense(num_outputs, init="glorot_normal", activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Adam(), metrics=[])
    return model


def smhtt_dropout(num_inputs, num_outputs):
    model = Sequential()

    for i, nodes in enumerate([200] * 2):
        if i == 0:
            model.add(Dense(nodes, input_dim=num_inputs))
        else:
            model.add(Dense(nodes))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

    model.add(Dense(num_outputs))
    model.add(Activation("softmax"))

    model.compile(loss="mean_squared_error", optimizer=Nadam())
    return model


# Model used by Janek Bechtel in his thesis
def smhtt_dropout_tanh(num_inputs, num_outputs, node_num=200, layer_num=2):
    model = Sequential()

    for i, nodes in enumerate([node_num] * layer_num):
        if i == 0:
            model.add(Dense(nodes, kernel_regularizer=l2(1e-5), input_dim=num_inputs))
        else:
            model.add(Dense(nodes, kernel_regularizer=l2(1e-5)))
        model.add(Activation("tanh"))
        model.add(Dropout(0.3))

    model.add(Dense(num_outputs, kernel_regularizer=l2(1e-5)))
    model.add(Activation("softmax", dtype="float32"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-4),
        weighted_metrics=["mean_squared_error"],
    )
    return model


# Model used by Tim Voigtländer in his thesis
def smhtt_dropout_tanh_GPU(num_inputs, num_outputs, node_num=512, layer_num=3):
    model = Sequential()

    for i, nodes in enumerate([node_num] * layer_num):
        if i == 0:
            model.add(Dense(nodes, kernel_regularizer=l2(1e-5), input_dim=num_inputs))
        else:
            model.add(Dense(nodes, kernel_regularizer=l2(1e-5)))
        model.add(Activation("tanh"))
        model.add(Dropout(0.3))

    model.add(Dense(num_outputs, kernel_regularizer=l2(1e-5)))
    model.add(Activation("softmax", dtype="float32"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-4),
        weighted_metrics=["mean_squared_error"],
    )
    return model
