from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam

def build_model_with_embedding(text_maxlen, meta_dim):
    # Texte
    text_input = Input(shape=(text_maxlen,), name="text_input")
    x_text = Embedding(input_dim=10000, output_dim=256)(text_input) 
    x_text = GlobalAveragePooling1D()(x_text)

    # Features numériques
    meta_input = Input(shape=(meta_dim,), name="meta_input")
    x_meta = Dense(128, activation='relu')(meta_input) 

    # Fusion
    x = Concatenate()([x_text, x_meta])
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)  # Sortie pour régression

    model = Model(inputs=[text_input, meta_input], outputs=x)
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model
