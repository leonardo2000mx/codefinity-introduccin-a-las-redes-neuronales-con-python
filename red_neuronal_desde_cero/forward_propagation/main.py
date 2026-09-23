import numpy as np
import os
os.system('wget https://codefinity-content-media.s3.eu-west-1.amazonaws.com/f9fc718f-c98b-470d-ba78-d84ef16ba45f/section_2/layers.py 2>/dev/null')
from layers import hidden_1, hidden_2, output_layer

# Fix the seed of the "random" library, so it will be easier to test our code 
np.random.seed(10)

class Perceptron:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        x = np.array(inputs).reshape(1, -1)
        for layer in self.layers:
            x = layer.forward(x)
        return x

layers = [hidden_1, hidden_2, output_layer]
perceptron = Perceptron(layers)

# Testing the perceptron with two inputs: 1 and 0
inputs = [1, 0]
print(f'Inputs: {inputs}')
print(f'Output: {perceptron.forward(inputs)[0, 0]:.2f}')