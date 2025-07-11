width = 1440
height = 900
b_size = 32
left_margin = 40
right_margin = b_size * 20
top_margin = 60
down_margin = 20

# Evolution Parameters
generations = 100
population = 50

# Neural Network Architecture
i_size = 8      # Input layer size (8 features)
h1_size = 8     # First hidden layer size
h2_size = 6     # Second hidden layer size  
o_size = 4      # Output layer size (4 movement directions)

# Genetic Algorithm Parameters
mutation_rate = 0.4     # Threshold for mutation (lower = more mutation)
mutation_strength_kernel = 1.5  # Multiplier for kernel weight mutations
mutation_strength_bias = 0.5    # Multiplier for bias weight mutations

# Colors
PLAYER_RED = (230, 51, 51)
PLAYER_GREEN = (51, 230, 51)
PLAYER_BLUE = (51, 51, 230)
WHITE = (255, 255, 255)
