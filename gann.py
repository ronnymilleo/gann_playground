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
    # Inputs
    # Player x, y
    # Target x, y
    # Distance
    # Quadrant
    # Angle
    i_size = globals.i_size
    h1_size = globals.h1_size
    h2_size = globals.h2_size
    o_size = globals.o_size

    # Outputs
    # Move Up, Down, Left, Right

    # Constructor
    def __init__(self):
        # Initialize Sequential Model Super Class
        super().__init__()
        self.fitness = 0
        self.fitness_array = []
        self.max_fitness = 0
        self.mean_fitness = 0
        # If no weights provided randomly generate them
        # Layers are created and randomly generated
        layer1 = Dense(self.i_size, input_shape=(self.i_size,), activation='tanh', bias_initializer='random_normal')
        layer2 = Dense(self.h1_size, activation='tanh', bias_initializer='random_normal')
        layer3 = Dense(self.h2_size, activation='tanh', bias_initializer='random_normal')
        layer4 = Dense(self.o_size, activation='softmax')
        # Layers are added to the model
        self.add(layer1)
        self.add(layer2)
        self.add(layer3)
        self.add(layer4)

    def fitness_update(self, distance):
        # Calculate max possible distance dynamically
        max_distance = math.sqrt(20**2 + 20**2)  # Grid is 20x20
        self.fitness = (1 - distance / max_distance)
        self.fitness_array.append(self.fitness)
        self.mean_fitness = np.mean(self.fitness_array)


# Chance to mutate weights
def mutation(child_weights):
    # Add a chance for random mutation
    # child_weights is a list with [kernel_weights, bias_weights]
    mut = random.uniform(0, 1)
    if mut >= globals.mutation_rate:
        # Mutate kernel weights (2D array)
        kernel_shape = child_weights[0].shape
        for _ in range(2):  # Apply mutations to 2 random kernel weights
            i = random.randint(0, kernel_shape[0] - 1)
            j = random.randint(0, kernel_shape[1] - 1)
            child_weights[0][i, j] *= random.random() * globals.mutation_strength_kernel
        
        # Mutate bias weights (1D array)
        bias_shape = child_weights[1].shape
        for _ in range(2):  # Apply mutations to 2 random bias weights
            i = random.randint(0, bias_shape[0] - 1)
            child_weights[1][i] *= random.random() * globals.mutation_strength_bias
    else:
        # No mutation
        pass


# Crossover traits between two Genetic Neural Networks
def dynamic_crossover(parent_1, parent_2) -> GeneticANN:
    # A new child is born
    child = GeneticANN()
    
    # Layer 1 (input to first hidden)
    p1_weights = parent_1.layers[0].get_weights()
    p2_weights = parent_2.layers[0].get_weights()
    child_weights_l1 = [np.zeros_like(p1_weights[0]), np.zeros_like(p1_weights[1])]
    
    # Crossover weights
    for w_I in range(p1_weights[0].shape[0]):
        for w_J in range(p1_weights[0].shape[1]):
            roll_d20 = random.randint(0, 20)
            if roll_d20 < 10:
                child_weights_l1[0][w_I, w_J] = p1_weights[0][w_I, w_J]
            else:
                child_weights_l1[0][w_I, w_J] = p2_weights[0][w_I, w_J]
    
    # Crossover biases
    for b_I in range(len(p1_weights[1])):
        roll_d20 = random.randint(0, 20)
        if roll_d20 < 10:
            child_weights_l1[1][b_I] = p1_weights[1][b_I]
        else:
            child_weights_l1[1][b_I] = p2_weights[1][b_I]
    
    mutation(child_weights_l1)
    child.layers[0].set_weights(child_weights_l1)

    # Layer 2 (first hidden to second hidden)
    p1_weights = parent_1.layers[1].get_weights()
    p2_weights = parent_2.layers[1].get_weights()
    child_weights_l2 = [np.zeros_like(p1_weights[0]), np.zeros_like(p1_weights[1])]
    
    for w_I in range(p1_weights[0].shape[0]):
        for w_J in range(p1_weights[0].shape[1]):
            roll_d20 = random.randint(0, 20)
            if roll_d20 < 10:
                child_weights_l2[0][w_I, w_J] = p1_weights[0][w_I, w_J]
            else:
                child_weights_l2[0][w_I, w_J] = p2_weights[0][w_I, w_J]
    
    for b_I in range(len(p1_weights[1])):
        roll_d20 = random.randint(0, 20)
        if roll_d20 < 10:
            child_weights_l2[1][b_I] = p1_weights[1][b_I]
        else:
            child_weights_l2[1][b_I] = p2_weights[1][b_I]
    
    mutation(child_weights_l2)
    child.layers[1].set_weights(child_weights_l2)

    # Layer 3 (second hidden to third hidden)
    p1_weights = parent_1.layers[2].get_weights()
    p2_weights = parent_2.layers[2].get_weights()
    child_weights_l3 = [np.zeros_like(p1_weights[0]), np.zeros_like(p1_weights[1])]
    
    for w_I in range(p1_weights[0].shape[0]):
        for w_J in range(p1_weights[0].shape[1]):
            roll_d20 = random.randint(0, 20)
            if roll_d20 < 10:
                child_weights_l3[0][w_I, w_J] = p1_weights[0][w_I, w_J]
            else:
                child_weights_l3[0][w_I, w_J] = p2_weights[0][w_I, w_J]
    
    for b_I in range(len(p1_weights[1])):
        roll_d20 = random.randint(0, 20)
        if roll_d20 < 10:
            child_weights_l3[1][b_I] = p1_weights[1][b_I]
        else:
            child_weights_l3[1][b_I] = p2_weights[1][b_I]
    
    mutation(child_weights_l3)
    child.layers[2].set_weights(child_weights_l3)

    # Layer 4 (third hidden to output)
    p1_weights = parent_1.layers[3].get_weights()
    p2_weights = parent_2.layers[3].get_weights()
    child_weights_l4 = [np.zeros_like(p1_weights[0]), np.zeros_like(p1_weights[1])]
    
    for w_I in range(p1_weights[0].shape[0]):
        for w_J in range(p1_weights[0].shape[1]):
            roll_d20 = random.randint(0, 20)
            if roll_d20 < 10:
                child_weights_l4[0][w_I, w_J] = p1_weights[0][w_I, w_J]
            else:
                child_weights_l4[0][w_I, w_J] = p2_weights[0][w_I, w_J]
    
    for b_I in range(len(p1_weights[1])):
        roll_d20 = random.randint(0, 20)
        if roll_d20 < 10:
            child_weights_l4[1][b_I] = p1_weights[1][b_I]
        else:
            child_weights_l4[1][b_I] = p2_weights[1][b_I]
    
    mutation(child_weights_l4)
    child.layers[3].set_weights(child_weights_l4)

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
