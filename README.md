# Genetic Neural Network Game

## Overview

This project implements a genetic algorithm that evolves neural networks to navigate from a starting point to a target in a 2D grid environment. The system uses TensorFlow/Keras for neural network implementation and Pygame for visualization.

## 🎯 Project Goal

Train AI agents using evolutionary algorithms to learn optimal pathfinding strategies through:
- **Selection**: Best performing agents are chosen for reproduction
- **Crossover**: Neural network weights are combined from parent agents
- **Mutation**: Random modifications introduce genetic diversity
- **Fitness Evaluation**: Agents are scored based on their distance to the target

## 📁 Project Structure

```
geneticNNgame/
├── main.py           # Main game loop and simulation logic
├── gann.py           # Genetic Algorithm Neural Network implementation
├── player.py         # Player/Agent class definition
├── globals.py        # Global configuration parameters
├── font/             # Font assets
│   └── Roboto-Regular.ttf
├── deep-learning.png # Game icon
├── start_flag.png    # Start position marker
├── target.png        # Target position marker
└── README.md         # This documentation
```

## 🧠 Neural Network Architecture

### Input Layer (8 neurons)
The neural network receives 8 normalized inputs:
1. **Player X position** (0-1): Current X coordinate normalized by grid width
2. **Player Y position** (0-1): Current Y coordinate normalized by grid height  
3. **Distance to target** (0-1): Euclidean distance normalized by maximum possible distance (26.87)
4. **Quadrant** (0-0.25): Which quadrant the player is in relative to the grid center
5. **X angle** (0-π/2): Angle component in X direction to target
6. **Y angle** (0-π/2): Angle component in Y direction to target
7. **Target X position** (0-1): Target X coordinate normalized
8. **Target Y position** (0-1): Target Y coordinate normalized

### Hidden Layers
- **Layer 1**: 8 neurons, tanh activation
- **Layer 2**: 8 neurons, tanh activation  
- **Layer 3**: 6 neurons, tanh activation

### Output Layer (4 neurons)
- **Softmax activation** producing probabilities for 4 actions:
  - Index 0: Move Right
  - Index 1: Move Left
  - Index 2: Move Down
  - Index 3: Move Up

## 🧬 Genetic Algorithm Implementation

### Population Structure
- **Population Size**: 50 AI agents
- **Generations**: 100 (configurable)
- **Elite Selection**: Top 5 performers saved each generation

### Selection Process
1. **Fitness Calculation**: `fitness = 1 - (distance_to_target / max_distance)`
2. **Ranking**: Players sorted by fitness (higher = better)
3. **Elite Preservation**: Best 5 models saved as `.keras` files

### Crossover Strategy
- **Method**: Uniform crossover at individual weight level
- **Parents**: Top 5 performers from previous generation
- **Offspring Generation**: 5×5 = 25 combinations, plus random combinations to fill population
- **Weight Selection**: 50% probability from each parent for each weight

### Mutation Parameters
- **Mutation Rate**: 60% chance (when `random.uniform(0,1) >= 0.4`)
- **Kernel Weight Mutation**: 2 random weights × `random() * 1.5`
- **Bias Weight Mutation**: 2 random weights × `random() * 0.5`

## 🎮 Simulation Parameters

### Environment Configuration
```python
# Screen Settings (globals.py)
width = 1440          # Screen width
height = 900          # Screen height  
b_size = 32           # Grid cell size (32×32 pixels)
grid_size = 20×20     # Logical grid dimensions

# Evolution Settings (globals.py)
generations = 100     # Number of evolution cycles
population = 50       # AI agents per generation
max_steps = 30        # Steps per agent per generation (in main.py)

# Neural Network Architecture (globals.py)
i_size = 8           # Input layer neurons (8 features)
h1_size = 8          # First hidden layer neurons
h2_size = 6          # Second hidden layer neurons
o_size = 4           # Output layer neurons (4 directions)

# Genetic Algorithm Settings (globals.py)
mutation_rate = 0.4           # Mutation threshold (0.0-1.0)
mutation_strength_kernel = 1.5 # Weight mutation multiplier
mutation_strength_bias = 0.5   # Bias mutation multiplier
```

### Objective Randomization
- **Frequency**: Every 10 generations
- **Constraint**: Start and target positions must be at least 5 grid cells apart
- **Purpose**: Prevent overfitting to specific start/target combinations

## 🚀 Usage

### Prerequisites
```bash
pip install tensorflow pygame numpy
```

### Running the Simulation
```bash
python main.py
```

### Controls
- **ESC**: Exit simulation
- **Ctrl+C**: Graceful shutdown
- **Window Close**: Exit application

### Visual Elements
- **Green squares**: Successful agents (reached target)
- **Blue squares**: Failed agents (hit boundary or timeout)
- **Default color**: Active agents still navigating
- **Grid lines**: 20×20 navigation grid
- **Flag icon**: Starting position
- **Target icon**: Goal position

## 📊 Performance Monitoring

### Real-time Display
- **Generation Counter**: Current evolution cycle
- **Population Size**: Total number of agents
- **Step Counter**: Current step within generation
- **Fitness Table**: Individual agent performance scores

### Model Persistence
- **Auto-save**: Top 5 models saved each generation
- **Format**: TensorFlow Keras native format (`.keras`)
- **Filenames**: `model_0.keras` through `model_4.keras`
- **Location**: Project root directory

## 🔧 Code Architecture

### Main Components

#### `main.py` - Simulation Engine
- **Game loop management**
- **Pygame rendering system**
- **Agent movement logic**
- **Fitness evaluation**
- **Generation evolution orchestration**

#### `gann.py` - Genetic Algorithm Core
- **`GeneticANN`**: Neural network class inheriting from Keras Sequential
- **`dynamic_crossover()`**: Weight combination from two parent networks
- **`mutation()`**: Random weight perturbation for genetic diversity

#### `player.py` - Agent Definition  
- **Player sprite management**
- **Position and state tracking**
- **Neural network integration**
- **Movement capabilities**

#### `globals.py` - Configuration
- **Centralized parameter management**
- **Screen dimensions and grid settings**
- **Color definitions**
- **Population and generation limits**

## 🐛 Known Issues & Limitations

### Current Limitations
1. **Fixed Architecture**: Neural network structure is hardcoded
2. **Simple Fitness**: Only considers final distance, not path efficiency
3. **Limited Inputs**: Could benefit from additional environmental features
4. **Mutation Strategy**: Relatively simple compared to advanced GA techniques
5. **No Early Stopping**: Simulation runs for full generation count regardless of convergence

### Potential Improvements
1. **Adaptive Architecture**: Allow evolution of network topology
2. **Multi-objective Fitness**: Consider path length, efficiency, and exploration
3. **Advanced Mutation**: Implement Gaussian mutation with adaptive rates
4. **Elitism Strategy**: Preserve more top performers with diversity constraints
5. **Convergence Detection**: Implement early stopping when population converges
6. **Parallel Processing**: Distribute fitness evaluation across multiple cores
7. **Hyperparameter Evolution**: Evolve learning parameters alongside weights

## 🔍 Debugging Features

### Logging Options
- **Generation Progress**: Console output for each generation
- **Error Handling**: Graceful shutdown with informative messages
- **Performance Tracking**: Fitness statistics and improvement trends

### Development Tools
- **Debug Output**: Commented code for position and angle analysis
- **Model Inspection**: Saved models for external analysis
- **Visual Feedback**: Color-coded agent states for behavior observation

## 📈 Expected Learning Behavior

### Early Generations (1-20)
- **Random exploration** with minimal target awareness
- **High failure rate** due to boundary collisions
- **Gradual emergence** of basic directional movement

### Mid Generations (21-60)  
- **Improved target recognition** and basic pathfinding
- **Reduced boundary collisions** 
- **Development of turning behaviors** around obstacles

### Late Generations (61-100)
- **Optimized path planning** with efficient routes
- **Consistent target achievement** across different start/target combinations
- **Robust navigation** handling various environmental configurations

## 🎯 Success Metrics

### Convergence Indicators
- **Fitness Improvement**: Steady increase in average and maximum fitness scores
- **Success Rate**: Higher percentage of agents reaching targets within step limit
- **Path Efficiency**: Shorter average path lengths to reach targets
- **Generalization**: Consistent performance across different start/target positions

## 🛠️ Customization Options

### Easy Modifications
```python
# Adjust population size
population = 100  # in globals.py

# Change mutation rate  
mutation_rate = 0.3  # in globals.py (lower value = more mutation)

# Modify neural network architecture
h1_size = 16  # in globals.py - increase hidden layer size
h2_size = 12  # in globals.py - adjust second hidden layer

# Extend simulation duration
generations = 200  # in globals.py - more evolution cycles

# Fine-tune mutation strength
mutation_strength_kernel = 2.0  # in globals.py - stronger weight mutations
mutation_strength_bias = 0.8    # in globals.py - stronger bias mutations
```

### Advanced Customizations
- **Custom fitness functions** in `gann.py:fitness_update()`
- **Alternative crossover strategies** in `gann.py:dynamic_crossover()`
- **Modified input features** in `main.py:move()`
- **Different activation functions** in `gann.py:GeneticANN.__init__()`

## 📚 References

- **Original Implementation**: Based on Roman Michael Paolucci's Genetic Neural Network
- **TensorFlow Documentation**: https://tensorflow.org/guide
- **Pygame Documentation**: https://pygame.org/docs/
- **Genetic Algorithms**: Introduction to Evolutionary Computing (Eiben & Smith)

## 📄 License

This project is educational and experimental. See original repository for licensing details.

---

*Created by Ronny Milleo based on github.com/RomanMichaelPaolucci/Genetic_Neural_Network*
