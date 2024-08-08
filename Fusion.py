import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Multiply, Concatenate, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def conv_block(x, filters, kernel_size=3, stride=1, use_batchnorm=True):
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same', use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def attention_block(x):
    # Apply 1x1 convolution to create attention weights
    attention = Conv2D(x.shape[-1], (1, 1), padding='same', activation='sigmoid')(x)
    return Multiply()([x, attention])

def EFFA_model(x1, x2):
    # Ensure x1 and x2 have the same shape
    if x1.shape[-1] != x2.shape[-1]:
        x1 = Conv2D(x2.shape[-1], (1, 1), padding='same')(x1)
    
    # Feature Fusion
    fused_features = Concatenate()([x1, x2])
    
    # Apply attention to fused features
    attention_features = attention_block(fused_features)
    
    # Optionally apply a convolutional layer to refine features
    refined_features = Conv2D(x1.shape[-1], (1, 1), padding='same')(attention_features)
    
    return refined_features


def effective_feature_fusion_attention():
    
    # Define the input layers
    input_shape = (224, 224, 3)
    inputs1 = Input(shape=input_shape)
    inputs2 = Input(shape=input_shape)
    
    # Create feature maps from both inputs
    x1 = conv_block(inputs1, 64)
    x2 = conv_block(inputs2, 64)
    
    # Apply the EFFA block
    effa_output = EFFA_model(x1, x2)
    
    # Define a simple model for demonstration
    model = Model(inputs=[inputs1, inputs2], outputs=effa_output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    # Print the model summary
    model.summary()
    
    return model
