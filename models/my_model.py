from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

#def create_model(input_shape=(20, 4)):
#   model = Sequential()
#    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape))
#    model.add(Dropout(0.2))
#    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
#    model.add(Dropout(0.2))
#    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
#    model.add(Dropout(0.2))
#    model.add(MaxPooling1D(pool_size=2))
#    model.add(Flatten())
#    model.add(Dense(256, activation='relu'))
#    model.add(Dropout(0.2))
#    model.add(Dense(1, activation="sigmoid"))
#    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#    return model)
def create_model(input_shape=(20,4)):
    model = Sequential()
    
    # Convolutional layers
    model.add(layers.Conv1D(16, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(16, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.Dropout(0.2))
    
    # Max pooling layer
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # Flatten layer
    model.add(layers.Flatten())
    
    # Fully connected layers
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer
    
    return model

def create_deep_model(input_shape=(1000, 4)):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def create_simple_model(input_shape=(20, 4)):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

