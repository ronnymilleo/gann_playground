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
gann_playground/
├── main.py                   # Main game loop and simulation logic
├── gann.py                   # Genetic Algorithm Neural Network implementation
├── player.py                 # Player/Agent class definition
├── config.py                 # Modern configuration management system
├── globals.py                # Legacy global configuration parameters
├── test_genetic_game.py      # Unit tests for the genetic algorithm
├── requirements.txt          # Python dependencies with versions
├── genetic_training.log      # Training progress and error logs
├── CONFIGURATION_EXAMPLES.md # Configuration examples and presets
├── data.npy                  # Saved training data (optional)
├── model_*.keras            # Saved neural network models (auto-generated)
├── font/                    # Font assets
│   └── Roboto-Regular.ttf
├── deep-learning.png        # Game icon
├── start_flag.png           # Start position marker
├── target.png               # Target position marker
└── README.md                # This documentation
```

## 🧠 Neural Network Architecture

### Input Layer (4 neurons) - Simplified and More Effective
The neural network now receives 4 normalized inputs for better performance:
1. **Player X position** (0-1): Current X coordinate normalized by grid width
2. **Player Y position** (0-1): Current Y coordinate normalized by grid height  
3. **Delta X** (-1 to 1): Normalized difference between player and target X positions
4. **Delta Y** (-1 to 1): Normalized difference between player and target Y positions

### Hidden Layers
- **Layer 1**: 8 neurons, tanh activation, Glorot uniform initialization
- **Layer 2**: 8 neurons, tanh activation, Glorot uniform initialization

### Output Layer (4 neurons)
- **Softmax activation** producing probabilities for 4 actions:
  - Index 0: Move Right
  - Index 1: Move Left
  - Index 2: Move Down
  - Index 3: Move Up

## 🧬 Genetic Algorithm Implementation

### Population Structure
- **Population Size**: 100 AI agents (increased from 50)
- **Generations**: 100 (configurable)
- **Elite Selection**: Top 5 performers saved each generation

### Selection Process
1. **Fitness Calculation**: Multi-objective fitness combining distance and path efficiency
   - `fitness = 0.7 * (1 - normalized_distance) + 0.3 * step_efficiency`
   - **Success bonus**: +0.5 points for reaching the target
   - **Step efficiency**: Rewards shorter paths (1.0 - steps/max_steps)
2. **Ranking**: Players sorted by fitness (higher = better)
3. **Elite Preservation**: Best 5 models saved as `.keras` files

### Crossover Strategy
- **Method**: Layer-wise crossover with random crossover point
- **Parents**: Top 5 performers from previous generation
- **Offspring Generation**: 5×5 = 25 combinations, plus random combinations to fill population
- **Layer Selection**: Layers before crossover point from parent 1, layers after from parent 2

### Mutation Parameters
- **Mutation Rate**: 70% chance (when `random.uniform(0,1) >= 0.3`)
- **Selective Mutation**: Only 10% of weights per layer are mutated (masked approach)
- **Gaussian Noise**: Normal distribution with mean=0 for natural weight perturbation
- **Kernel Weight Mutation**: Gaussian noise × `mutation_strength_kernel` (0.3)
- **Bias Weight Mutation**: Gaussian noise × `mutation_strength_bias` (0.15)

## 🎮 Simulation Parameters

### Environment Configuration
```python
# Screen Settings (config.py or globals.py)
screen_width = 1440       # Screen width  
screen_height = 900       # Screen height
cell_size = 32           # Grid cell size (32×32 pixels)
grid_width = 20          # Grid width in cells
grid_height = 20         # Grid height in cells

# Evolution Settings (globals.py - updated values)
generations = 100        # Number of evolution cycles
population = 100         # AI agents per generation (increased from 50)
max_steps = 60          # Steps per agent per generation (increased from 30)

# Neural Network Architecture (config.py)
input_size = 4          # Input layer neurons (simplified: position + deltas)
hidden_layer_1_size = 8 # First hidden layer neurons
hidden_layer_2_size = 8 # Second hidden layer neurons  
output_size = 4         # Output layer neurons (4 directions)

# Genetic Algorithm Settings (globals.py - updated values)
mutation_rate = 0.3              # Mutation threshold (increased exploration)
mutation_strength_kernel = 0.3   # Weight mutation multiplier (increased)
mutation_strength_bias = 0.15    # Bias mutation multiplier (increased)
```

### Objective Randomization
- **Frequency**: Every 10 generations
- **Constraint**: Start and target positions must be at least 5 grid cells apart
- **Purpose**: Prevent overfitting to specific start/target combinations

## 🚀 Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install tensorflow==2.19.0 pygame==2.6.1
```

### Running the Simulation
```bash
python main.py
```

### Running Tests
```bash
python -m pytest test_genetic_game.py -v
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
- **Simplified Architecture**: 4-input design for better convergence
- **Improved Initialization**: Glorot uniform initialization for stable training
- **Multi-objective Fitness**: Distance + path efficiency with success bonus
- **`dynamic_crossover()`**: Layer-wise crossover with random crossover points
- **`mutation()`**: Selective Gaussian mutation affecting only 10% of weights per layer

#### `player.py` - Agent Definition  
- **Player sprite management**
- **Position and state tracking**
- **Neural network integration**
- **Movement capabilities**

#### `config.py` - Modern Configuration System
- **Centralized configuration management** using dataclasses
- **Type safety** with proper type hints
- **Computed properties** for derived values (max_distance, grid dimensions)
- **Legacy compatibility** for backward compatibility with globals.py
- **Dynamic updates** with validation

#### `globals.py` - Legacy Configuration  
- **Backward compatibility** with existing code
- **Direct parameter access** for quick modifications
- **Evolution parameters** and mutation settings

## 🐛 Known Issues & Limitations

### Current Limitations
1. ~~**Fixed Architecture**: Neural network structure is hardcoded~~ **IMPROVED**: Now configurable via config.py
2. ~~**Simple Fitness**: Only considers final distance, not path efficiency~~ **IMPROVED**: Multi-objective fitness
3. ~~**Limited Inputs**: Could benefit from additional environmental features~~ **IMPROVED**: Simplified, more effective inputs
4. ~~**Mutation Strategy**: Relatively simple compared to advanced GA techniques~~ **IMPROVED**: Selective Gaussian mutation
5. **No Early Stopping**: Simulation runs for full generation count regardless of convergence

### Potential Improvements
1. ~~**Adaptive Architecture**: Allow evolution of network topology~~ **PARTIALLY COMPLETE**: Now configurable
2. ~~**Multi-objective Fitness**: Consider path length, efficiency, and exploration~~ **COMPLETE**: Implemented
3. ~~**Advanced Mutation**: Implement Gaussian mutation with adaptive rates~~ **COMPLETE**: Selective Gaussian mutation
4. **Elitism Strategy**: Preserve more top performers with diversity constraints
5. **Convergence Detection**: Implement early stopping when population converges
6. **Parallel Processing**: Distribute fitness evaluation across multiple cores
7. **Hyperparameter Evolution**: Evolve learning parameters alongside weights

## 🔍 Debugging Features

### Logging Options
- **Generation Progress**: Console output for each generation
- **Training Logs**: Detailed logging to `genetic_training.log`
- **Error Handling**: Graceful shutdown with informative messages
- **Performance Tracking**: Fitness statistics and improvement trends
- **Timestamp Tracking**: All events logged with precise timestamps

### Development Tools
- **Unit Testing**: Comprehensive test suite with `pytest`
- **Debug Output**: Commented code for position and angle analysis
- **Model Inspection**: Saved models for external analysis
- **Visual Feedback**: Color-coded agent states for behavior observation
- **Configuration Validation**: Type-safe configuration management

## 🛠️ Customization Options

### Easy Modifications
```python
# Adjust population size
population = 200  # in globals.py (increased from default 100)

# Change mutation rate  
mutation_rate = 0.2  # in globals.py (lower value = more mutation)

# Modify neural network architecture (in config.py)
hidden_layer_1_size = 16  # increase hidden layer size
hidden_layer_2_size = 12  # adjust second hidden layer

# Extend simulation duration
generations = 200  # in globals.py - more evolution cycles
max_steps = 80    # in globals.py - more steps per generation

# Fine-tune mutation strength
mutation_strength_kernel = 0.5  # in globals.py - stronger weight mutations
mutation_strength_bias = 0.2    # in globals.py - stronger bias mutations
```

See `CONFIGURATION_EXAMPLES.md` for pre-configured settings for different scenarios.

### Advanced Customizations
- **Custom fitness functions** in `gann.py:fitness_update()`
- **Alternative crossover strategies** in `gann.py:dynamic_crossover()`
- **Modified input features** in `main.py:move()`
- **Different activation functions** in `gann.py:GeneticANN.__init__()`
- **Configuration management** in `config.py` for type-safe parameter handling

## 🧪 Testing

### Running Tests
```bash
# Run all tests with verbose output
python -m pytest test_genetic_game.py -v

# Run specific test classes
python -m pytest test_genetic_game.py::TestGeneticANN -v
python -m pytest test_genetic_game.py::TestPlayer -v

# Run with coverage (requires pytest-cov)
python -m pytest test_genetic_game.py --cov=. --cov-report=html
```

### Test Coverage
The test suite covers:
- **Neural Network Initialization**: Proper layer setup and parameter validation
- **Fitness Calculation**: Boundary conditions and mathematical correctness
- **Genetic Operations**: Crossover and mutation functionality
- **Player Behavior**: Movement and state management
- **Configuration Validation**: Type safety and parameter ranges

## �📚 References

- **Original Implementation**: Based on Roman Michael Paolucci's Genetic Neural Network
- **TensorFlow Documentation**: https://tensorflow.org/guide
- **Pygame Documentation**: https://pygame.org/docs/
- **Genetic Algorithms**: Introduction to Evolutionary Computing (Eiben & Smith)
- **Testing Framework**: pytest documentation at https://pytest.org

## 📄 License

This project is educational and experimental. Based on the original work by Roman Michael Paolucci.
Enhanced and modernized by Ronny Milleo with improved configuration management, testing framework, and development tools.

---

*Created by Ronny Milleo based on github.com/RomanMichaelPaolucci/Genetic_Neural_Network*
