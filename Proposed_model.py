import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, Input, Bidirectional, LSTM, Reshape
from tensorflow.keras.models import Model

def conv_block(x, filters, kernel_size=3, stride=1, use_batchnorm=True):
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same', use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def residual_block(x, filters, stride=1):
    shortcut = x
    x = conv_block(x, filters, stride=stride)
    x = conv_block(x, filters, use_batchnorm=False)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def build_resnet50_layer(input_tensor):
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual Blocks
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128, stride=1)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256, stride=1)
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512, stride=1)

    x = GlobalAveragePooling2D()(x)
    return x

def build_bilstm_model(input_shape=(224, 224, 3), lstm_units=128):
    inputs = Input(shape=input_shape)
    
    # Build decoder block
    resnet_features = build_resnet50_layer(inputs)
    
    # Reshape features to fit LSTM requirements
    # Expand dims to simulate sequence length
    features_reshaped = Reshape((1, resnet_features.shape[1]))(resnet_features)
    
    # Apply BiLSTM
    x = Bidirectional(LSTM(lstm_units, return_sequences=False))(features_reshaped)
    
    # Add a Dense layer for classification
    x = Dense(100, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=inputs, outputs=x)
    
    return model

def  Mul_EBi_ResNLSTM():
    # Define the input shape and number of classes
    input_shape = (224, 224, 3)
    
    # Build and compile the model
    model = build_bilstm_model(input_shape=input_shape, lstm_units=128)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Print the model summary
    model.summary()

    return model
