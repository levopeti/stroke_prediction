from torch.nn import Module, Linear, ReLU, Sequential
from torch.optim import Optimizer, Adam

from typing import Tuple

def define_mlp_model(input_shape: int,
                     output_shape: int,
                     layer_sizes: list,
                     learning_rate: float,
                     stroke_loss_factor: float,
                     wd: float,
                     **kwargs) -> Tuple[Module, Optimizer, Module]:
    assert len(layer_sizes) > 0, layer_sizes

    layers = [Linear(input_shape, layer_sizes[0])]
    for i in range(len(layer_sizes) - 1):
        layers.append(ReLU())
        layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))

    layers.append(Linear(layer_sizes[-1], output_shape))
    model = Sequential(*layers)
    model = model.float()

    if output_shape == 1:
        # regression
        custom_loss = stroke_loss_reg(stroke_loss_factor)
        stroke_accuracy = stroke_accuracy_reg
    else:
        # classification
        loss = CrossEntropyLoss()
        stroke_accuracy = stroke_accuracy_clas

    optimizer = Adam(model.parameters(),
                     lr=learning_rate,
                     weight_decay=wd,
                     amsgrad=True)

    return model, optimizer, loss


def define_model(input_shape, output_shape, layer_sizes, learning_rate, stroke_loss_factor, **kwargs) -> Model:
    assert len(layer_sizes) > 0, layer_sizes

    ip = Input(shape=(input_shape,), name="input")
    x = Dense(units=layer_sizes[0], name="hidden_layer", activation=None)(ip)  # "relu"
    x = ReLU()(x)

    for i, layer_size in enumerate(layer_sizes[1:]):
        x = Dense(units=layer_size, name="hidden_layer_{}".format(i + 2), activation=None)(x)  # "relu"
        x = ReLU()(x)

    if output_shape == 1:
        op = Dense(units=output_shape, name="prediction", activation="sigmoid")(x)
        custom_loss = stroke_loss_reg(stroke_loss_factor)
        stroke_accuracy = stroke_accuracy_reg
    else:
        op = Dense(units=output_shape, name="prediction", activation="softmax")(x)
        custom_loss = stroke_loss_clas(stroke_loss_factor)
        stroke_accuracy = stroke_accuracy_clas

    _model = Model(inputs=ip, outputs=op, name="full_model")
    _model.summary()

    optimizer = Adam(learning_rate, amsgrad=True)
    _model.compile(loss="mse",  # "categorical_crossentropy", custom_loss
                   optimizer=optimizer,
                   run_eagerly=True,
                   metrics=["accuracy"])  # , stroke_accuracy
    return _model