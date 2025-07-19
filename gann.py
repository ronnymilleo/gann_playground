#
# Created by Ronny Milleo based on:
# github.com/RomanMichaelPaolucci/Genetic_Neural_Network
#

import random
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import globals

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, enable=True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# New Type of Neural Network
class GeneticANN(Sequential):
    # Define architecture inside the class for better modularity
    i_size = 4  # Simplified input: player_x, player_y, delta_x, delta_y
    h1_size = 8  # Adjusted complexity
    h2_size = 8  # Adjusted complexity
    o_size = 4

    # Constructor
    def __init__(self):
        # Initialize Sequential Model Super Class
        super().__init__()
        self.fitness = 0
        self.fitness_array = []
        self.max_fitness = 0
        self.mean_fitness = 0
        
        # Use better weight initialization for more stable training
        # Layers are created with Xavier/Glorot initialization
        layer1 = Dense(self.h1_size,  # Corrected layer size
                      input_shape=(self.i_size,), 
                      activation='tanh', 
                      kernel_initializer='glorot_uniform',
                      bias_initializer='zeros')
        layer2 = Dense(self.h2_size, 
                      activation='tanh',
                      kernel_initializer='glorot_uniform', 
                      bias_initializer='zeros')
        layer3 = Dense(self.o_size, 
                      activation='softmax',
                      kernel_initializer='glorot_uniform',
                      bias_initializer='zeros')
        
        # Layers are added to the model
        self.add(layer1)
        self.add(layer2)
        self.add(layer3)

    def fitness_update(self, distance, steps):
        # Reward path efficiency and proximity to target
        # Normalize distance and steps
        max_distance = math.sqrt(20**2 + 20**2)
        normalized_distance = distance / max_distance
        
        # Encourage fewer steps, penalize more steps
        step_efficiency = 1.0 - (steps / 30.0)  # 30 is max_steps
        
        # Combine metrics: 70% distance, 30% efficiency
        self.fitness = (0.7 * (1 - normalized_distance)) + (0.3 * step_efficiency)
        
        # Add a bonus for reaching the target
        if distance == 0:
            self.fitness += 0.5  # Bonus for success
            
        self.fitness_array.append(self.fitness)
        self.mean_fitness = np.mean(self.fitness_array)


# More efficient mutation
def mutation(child: 'GeneticANN'):
    # Mutate weights by adding random noise
    if random.random() < globals.mutation_rate:
        for layer in child.layers:
            weights = layer.get_weights()
            # Mutate kernel weights
            kernel_weights = weights[0]
            # Add small random values to a portion of weights
            mask = np.random.choice([0, 1], size=kernel_weights.shape, p=[0.9, 0.1])
            noise = np.random.normal(0, globals.mutation_strength_kernel, kernel_weights.shape)
            weights[0] = kernel_weights + (noise * mask)
            
            # Mutate bias weights
            bias_weights = weights[1]
            mask = np.random.choice([0, 1], size=bias_weights.shape, p=[0.9, 0.1])
            noise = np.random.normal(0, globals.mutation_strength_bias, bias_weights.shape)
            weights[1] = bias_weights + (noise * mask)
            
            layer.set_weights(weights)


# More efficient and standard crossover
def dynamic_crossover(parent_1: 'GeneticANN', parent_2: 'GeneticANN') -> 'GeneticANN':
    child = GeneticANN()
    
    # Randomly select a layer to swap weights and biases
    crossover_point = random.randint(0, len(parent_1.layers) - 1)
    
    for i, (p1_layer, p2_layer, child_layer) in enumerate(zip(parent_1.layers, parent_2.layers, child.layers)):
        p1_weights = p1_layer.get_weights()
        p2_weights = p2_layer.get_weights()
        
        # Before crossover point, take from parent 1
        if i < crossover_point:
            child_layer.set_weights(p1_weights)
        # After crossover point, take from parent 2
        else:
            child_layer.set_weights(p2_weights)
            
    return child


def dna_exam(parent_1: GeneticANN, parent_2: GeneticANN, child: GeneticANN):
    p1_layer_1 = parent_1.layers[0].get_weights()[0]
    p1_layer_2 = parent_1.layers[1].get_weights()
    p1_layer_3 = parent_1.layers[2].get_weights()
    p1_layer_4 = parent_1.layers[3].get_weights()
    p2_layer_1 = parent_2.layers[0].get_weights()
    p2_layer_2 = parent_2.layers[1].get_weights()
    p2_layer_3 = parent_2.layers[2].get_weights()
    p2_layer_4 = parent_2.layers[3].get_weights()
    ch_layer_1 = child.layers[0].get_weights()[0]
    ch_layer_2 = child.layers[1].get_weights()
    ch_layer_3 = child.layers[2].get_weights()
    ch_layer_4 = child.layers[3].get_weights()
    return np.array(ch_layer_1) - np.array(p1_layer_1)
