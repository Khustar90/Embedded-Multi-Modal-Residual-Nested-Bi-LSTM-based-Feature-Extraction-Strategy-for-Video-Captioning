import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, MaxPooling2D, Flatten, Dense, Input, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model

" Define the GRFM (ResNet50-like) model "

def build_resnet50(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    
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

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128, stride=1)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256, stride=1)
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512, stride=1)
    
    x = GlobalAveragePooling2D()(x)  # Convert to a vector
    model = Model(inputs=inputs, outputs=x)
    return model

" Define the CSFM (3D CNN-like) model "

def build_3d_cnn(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = GlobalAveragePooling2D()(x)  # Convert to a vector
    model = Model(inputs=inputs, outputs=x)
    return model

" Define the TFM (R-CNN-like) model "

def build_rcnn(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    model = Model(inputs=inputs, outputs=x)
    return model

# Visual feature - GRFM + CSFM + TFM

def Visual_feature_GRFM_CSFM_TFM():
    
    # Build each model
    GRFM = build_resnet50(input_shape=(224, 224, 3))
    CSFM = build_3d_cnn(input_shape=(224, 224, 3))
    TFM = build_rcnn(input_shape=(224, 224, 3))
    
    # Concatenate the outputs
    concatenated = Concatenate()([GRFM.output, CSFM.output, TFM.output])
    
    # Add classification head
    x = Dense(512, activation='relu')(concatenated)
    x = Dense(10, activation='softmax')(x)  # Example: 10 classes for classification
    
    # Create the final model
    model = Model(inputs=[GRFM.input, CSFM.input, TFM.input], outputs=x)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Print the model summary
    model.summary()
    
    return model

