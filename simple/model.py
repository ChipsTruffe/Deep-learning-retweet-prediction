from tensorflow.keras import models, layers, optimizers, losses

def build_model(input_dim):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=losses.MeanAbsoluteError(),
        metrics=['mae']
    )
    return model
