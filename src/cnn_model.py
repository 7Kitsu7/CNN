# src/cnn_model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def build_model():
    # Entrada de imagen
    image_input = Input(shape=(224, 224, 1), name="image_input")
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Entrada tabular (edad, sexo)
    tabular_input = Input(shape=(2,), name="tabular_input")

    # Concatenar y clasificar
    merged = Concatenate()([x, tabular_input])
    dense = Dense(64, activation='relu')(merged)
    output = Dense(1, activation='sigmoid', name='output')(dense)

    model = Model(inputs=[image_input, tabular_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
