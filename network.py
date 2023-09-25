"""
    function for generating the ANN
"""

# TensorFlow is an open source machine learning library
import tensorflow as tf
# Keras is TensorFlow's high-level API for deep learning
from tensorflow import keras

# numpy
import numpy as np

# QKeras
import qkeras as qk

def truth_function(x):
    return np.sin(x)

def generate_data(NSAMPLES):
    """
        generate training / validation data for the network below
    """

    # Generate some random samples
    x_values = np.random.uniform(low=0, high=(2 * np.pi), size=NSAMPLES)

    # Create a noisy sinewave with these values
    y_values = truth_function(x_values)\
                    + (0.1 * np.random.randn(x_values.shape[0]))

    return x_values, y_values

def create_model(name="int_v0.1", quantized=True):
    """
        Test Network to use integer inputs and outputs
    """

    assert type(quantized) == bool

    if quantized:
        name = name + "_quant"

    if not quantized:
        inputs = keras.Input(shape=(1,), name="kinput")

        layer_cnt=0
        x = keras.layers.Dense(16,
                                name="layer_%d"%(layer_cnt))(inputs)
        layer_cnt+=1
        x = keras.layers.Activation("relu")(x)

        x = keras.layers.Dense(16,
                                name="layer_%d"%(layer_cnt))(x)
        layer_cnt+=1
        x = keras.layers.Activation("relu")(x)

        # final layer
        outputs = keras.layers.Dense(1, name="output")(x)
        
    else:
        inputs = keras.Input(shape=(1,), name="input")

        layer_cnt=0

        quant_bit_param = (13,6,1)

        # quantized_bits(bits=8, integer=0, symmetric=0, keep_negative=1)
        x = qk.QDense(16,
                    kernel_quantizer= qk.quantized_bits( *quant_bit_param ),
                    bias_quantizer  = qk.quantized_bits( *quant_bit_param ),
                    name="layer_%d"%(layer_cnt))(inputs)
        layer_cnt+=1
        x = qk.QActivation("quantized_relu(5)")(x)

        x = qk.QDense(8,
                    kernel_quantizer= qk.quantized_bits( *quant_bit_param ),
                    bias_quantizer  = qk.quantized_bits( *quant_bit_param ),
                    name="layer_%d"%(layer_cnt))(x)
        layer_cnt+=1
        x = qk.QActivation("quantized_relu(5)")(x)

        x = qk.QDense(8,
                    kernel_quantizer= qk.quantized_bits( *quant_bit_param ),
                    bias_quantizer  = qk.quantized_bits( *quant_bit_param ),
                    name="layer_%d"%(layer_cnt))(x)
        layer_cnt+=1
        x = qk.QActivation("quantized_relu(5)")(x)

        x = qk.QDense(8,
                    kernel_quantizer= qk.quantized_bits( *quant_bit_param ),
                    bias_quantizer  = qk.quantized_bits( *quant_bit_param ),
                    name="layer_%d"%(layer_cnt))(x)
        layer_cnt+=1
        x = qk.QActivation("quantized_relu(5)")(x)

        # final layer
        outputs = qk.QDense(1,
                    kernel_quantizer= qk.quantized_bits( *quant_bit_param ),
                    bias_quantizer  = qk.quantized_bits( *quant_bit_param ),
                    name="output")(x)


    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.summary()

    # Compile the model using the standard 'adam' optimizer and
    # the mean squared error or 'mse' loss function for regression.
    # the mean absolute error or 'mae' is also used as a metric
    model.compile(optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse'])

    if quantized:
        qk.print_qstats(model)

    return  model

if __name__ == "__main__":
    model = create_model( quantized=True )
